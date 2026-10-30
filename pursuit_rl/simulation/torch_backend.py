"""Vectorized device-resident pursuit simulator (experimental Torch backend).

The dynamics/reward equations mirror the NumPy surrogate. All rollout tensors
remain on device. Random draws use a dedicated torch.Generator; NumPy and torch
random trajectories are NOT bitwise identical even when seed matches.
"""
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from pursuit_rl.simulation.torch_device import resolve_torch_device


def norm(x, dim=-1, keepdim=False):
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)


def unit(x):
    return x / norm(x, keepdim=True).clamp_min(1e-9)


def cpa(r, v, horizon):
    s = (v*v).sum(-1)
    tc = (-(r*v).sum(-1) / s.clamp_min(1e-9)).clamp(0, horizon)
    tc = torch.where(s > 1e-8, tc, torch.zeros_like(tc))
    return norm(r + v*tc[:, None]), tc


def body_frame(rot, vec):
    return torch.bmm(rot.transpose(1,2), vec[..., None]).squeeze(-1)


def exp_so3(w, eye=None):
    # Rodrigues with stable small-angle formula; returns N,3,3.
    angle = norm(w)
    z = angle*angle
    a = torch.where(angle < 1e-4, 1 - z/6 + z*z/120, torch.sin(angle)/angle.clamp_min(1e-12))
    b = torch.where(angle < 1e-4, .5-z/24+z*z/720, (1-torch.cos(angle))/z.clamp_min(1e-12))
    x,y,q = w.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero,-q,y,q,zero,-x,-y,x,zero), -1).reshape(-1,3,3)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(0) if eye is None else eye
    return eye + a[:,None,None]*k + b[:,None,None]*torch.bmm(k,k)


def segment_hit(r0, r1, radius):
    d = r1-r0
    a = (d*d).sum(-1)
    b = 2*(r0*d).sum(-1)
    c = (r0*r0).sum(-1)-radius*radius
    disc = b*b-4*a*c
    root = torch.sqrt(disc.clamp_min(0))
    frac = (-b-root)/(2*a.clamp_min(1e-12))
    valid = (disc >= 0) & (a > 1e-12) & (frac >= 0) & (frac <= 1)
    inside = c <= 0
    return inside | valid, torch.where(inside, 0., torch.where(valid,frac,1.))


def boundary_risk(pos, p, soft):
    zlo, zhi = -p['max_altitude'], -p['min_altitude']
    margins = torch.stack((p['workspace_xy']-pos[:,0].abs(),p['workspace_xy']-pos[:,1].abs(),pos[:,2]-zlo,zhi-pos[:,2]),1)
    return ((soft-margins.min(1).values)/soft).clamp(0,1).square()


class TorchPursuitBatch:
    """Device-resident batch environment. No per-step NumPy transfers.

    Uses dense masked resets and GPU RNG so varying reset masks do not trigger
    a device->host synchronization every tick. Torch float64 can be selected for
    numerical regression; float32 is recommended for consumer NVIDIA GPUs.
    """
    def __init__(self, cfg, n=None, seed=None, stage=None, randomize=True, device='cpu', dtype=torch.float32):
        self.cfg=cfg
        self.n=int(cfg['training']['num_envs'] if n is None else n)
        self.device=resolve_torch_device(device)
        self.dtype=dtype
        self.gen=torch.Generator(device=self.device).manual_seed(int(cfg['seed'] if seed is None else seed))
        self.stage=int(cfg['training']['initial_stage'] if stage is None else stage)
        self.randomize=randomize
        self.u=SimpleNamespace()
        self.t=SimpleNamespace()
        self.pursuer=self.u; self.target=self.t
        self._eye=torch.eye(3,device=self.device,dtype=dtype).expand(self.n,-1,-1).clone()
        self._rate_limits=torch.tensor([cfg['physics']['rate_xy'],cfg['physics']['rate_xy'],cfg['physics']['rate_yaw']],device=self.device,dtype=dtype)
        for name in ('pos','vel','rate','acceleration'):
            setattr(self.u,name,self.zeros(self.n,3))
        self.u.rot=self._eye.clone()
        self.u.thrust=self.zeros(self.n)
        self.u.last_action=self.zeros(self.n,4)
        for name in ('drag','tau_rate','tau_thrust','mass_ratio'):
            setattr(self.u,name,self.zeros(self.n))
        for name in ('pos','vel','acc','ou_acc','desired_acc'):
            setattr(self.t,name,self.zeros(self.n,3))
        for name in ('heading','speed_setting','max_accel','phase_timer','phase_length','max_observed_speed','max_observed_acc'):
            setattr(self.t,name,self.zeros(self.n))
        for name in ('mode','change_count'):
            setattr(self.t,name,torch.zeros(self.n,device=self.device,dtype=torch.long))
        for name in ('elapsed','initial_distance','min_distance','last_cpa_potential','last_precision_potential','initial_closing','initial_cpa','episode_return','lateral_accel_cost_full','control_variation_full','high_envelope_count_full'):
            setattr(self,name,self.zeros(self.n))
        self.steps=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.initial_rel_pos=self.zeros(self.n,3)
        self.initial_target_vel=self.zeros(self.n,3)
        self.reward_components={}
        self.reset()

    @property
    def stage_config(self):
        return self.cfg['curriculum'][self.stage]

    def zeros(self,*shape):
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)
    def rand(self,*shape):
        return torch.rand(*shape,device=self.device,dtype=self.dtype,generator=self.gen)
    def randn(self,*shape):
        return torch.randn(*shape,device=self.device,dtype=self.dtype,generator=self.gen)
    def uniform(self, bounds, *shape):
        return bounds[0]+(bounds[1]-bounds[0])*self.rand(*shape)
    def _assign(self,obj,key,val,mask):
        old=getattr(obj,key)
        setattr(obj,key,torch.where(mask.reshape((-1,)+(1,)*(old.ndim-1)),val,old))

    def set_stage(self,stage):
        self.stage=int(stage)
        self.reset()

    @torch.no_grad()
    def reset(self,mask=None):
        n=self.n;p=self.cfg['physics'];c=self.cfg['target'];s=self.stage_config
        if mask is None: mask=torch.ones(n,dtype=torch.bool,device=self.device)
        if not isinstance(mask,torch.Tensor): mask=torch.as_tensor(mask,dtype=torch.bool,device=self.device)
        mask=mask.to(device=self.device,dtype=torch.bool)
        modes=torch.tensor([{'straight':0,'singer':1,'dubins':2,'hybrid':3}[x] for x in s['modes']],device=self.device)
        mode=modes[torch.randint(len(modes),(n,),generator=self.gen,device=self.device)]
        speed=self.uniform(s['speed'],n)
        heading=self.uniform((-math.pi,math.pi),n)
        pos=self.zeros(n,3)
        pos[:,:2]=self.uniform((-70.,70.),n,2)
        pos[:,2]=-self.uniform((75.,130.),n)
        vel=torch.stack((torch.cos(heading)*speed,torch.sin(heading)*speed,self.uniform((-.05,.05),n)*speed),dim=1)
        if c.get('strict_speed_ceiling',False): vel=unit(vel)*speed[:,None]
        self._assign(self.t,'mode',mode,mask)
        for key,value in {'speed_setting':speed,'max_accel':torch.full_like(speed,float(s['max_accel'])),'heading':heading,'pos':pos,'vel':vel,'acc':self.zeros(n,3),'ou_acc':self.zeros(n,3),'desired_acc':self.zeros(n,3),'phase_timer':self.zeros(n),'phase_length':self.uniform(c['maneuver_interval'],n),'change_count':torch.zeros(n,device=self.device,dtype=torch.long),'max_observed_speed':norm(vel),'max_observed_acc':self.zeros(n)}.items(): self._assign(self.t,key,value,mask)
        direction_heading=self.uniform((-math.pi,math.pi),n)
        elevation=self.uniform((-.18,.18),n)
        direction=torch.stack((torch.cos(direction_heading)*torch.cos(elevation),torch.sin(direction_heading)*torch.cos(elevation),torch.sin(elevation)),1)
        distance=self.uniform(s['distance'],n)
        initial=self.t.pos-direction*distance[:,None]
        initial[:,2]=initial[:,2].clamp(-p['max_altitude']+20,-p['min_altitude']-20)
        geometry=s.get('initial_geometry','random')
        if geometry!='random':
            if geometry=='overtaking': planar=direction[:,:2]
            elif geometry=='head_on': planar=-direction[:,:2]
            else: planar=torch.stack((-direction[:,1],direction[:,0]),1)
            planar=unit(planar)
            new_vel=torch.cat((planar*self.t.speed_setting[:,None],self.zeros(n,1)),dim=1)
            self._assign(self.t,'vel',new_vel,mask)
            self._assign(self.t,'heading',torch.atan2(planar[:,1],planar[:,0]),mask)
        self._assign(self.u,'pos',initial,mask)
        for key,val in {'vel':self.zeros(n,3),'rot':self._eye,'rate':self.zeros(n,3),'acceleration':self.zeros(n,3),'thrust':torch.full_like(speed,p['hover_thrust']),'last_action':self.zeros(n,4)}.items():self._assign(self.u,key,val,mask)
        fac=self.uniform((.8,1.2),n) if self.randomize else torch.ones_like(speed)
        params={'drag':p['drag']*fac,'tau_rate':p['tau_rate']*(self.uniform((.85,1.15),n) if self.randomize else torch.ones_like(speed)),'tau_thrust':p['tau_thrust']*(self.uniform((.85,1.15),n) if self.randomize else torch.ones_like(speed)),'mass_ratio':self.uniform((.95,1.05),n) if self.randomize else torch.ones_like(speed)}
        for key,val in params.items():self._assign(self.u,key,val,mask)
        rel=self.t.pos-self.u.pos;rv=self.t.vel-self.u.vel
        dc,_=cpa(rel,rv,self.cfg['reward']['cpa_horizon'])
        d=norm(rel)
        scalar={'elapsed':self.zeros(n),'steps':torch.zeros(n,dtype=torch.long,device=self.device),'initial_distance':d,'min_distance':d,'last_cpa_potential':-dc/self.cfg['reward']['distance_scale'],'last_precision_potential':torch.exp(-d/self.cfg['reward']['precision_scale_m']),'initial_closing':-(rel*rv).sum(1)/d.clamp_min(1e-8),'initial_cpa':dc,'episode_return':self.zeros(n),'lateral_accel_cost_full':self.zeros(n),'control_variation_full':self.zeros(n),'high_envelope_count_full':self.zeros(n)}
        for key,val in scalar.items():self._assign(self,key,val,mask)
        self._assign(self,'initial_rel_pos',rel,mask)
        self._assign(self,'initial_target_vel',self.t.vel,mask)
        return self.observe()

    def _target_step(self,dt):
        t=self.t;c=self.cfg['target'];n=self.n
        t.phase_timer=t.phase_timer+dt
        changed=t.phase_timer>=t.phase_length
        t.phase_timer=torch.where(changed,0.,t.phase_timer)
        t.phase_length=torch.where(changed,self.uniform(c['maneuver_interval'],n),t.phase_length)
        desired=self.randn(n,3)
        desired[:,2]*=.45
        desired=unit(desired)*t.max_accel[:,None]
        if c['hybrid_pattern'] in ('switchback','pulse'):
            sign=torch.where(t.change_count%2==0,1.,-1.)
            if c['hybrid_pattern']=='switchback':
                lateral=unit(torch.stack((-t.vel[:,1],t.vel[:,0],torch.zeros_like(sign)),1))
                desired=lateral*sign[:,None]*t.max_accel[:,None]
            else:desired=unit(t.vel)*sign[:,None]*t.max_accel[:,None]
        t.desired_acc=torch.where(changed[:,None],desired,t.desired_acc)
        t.change_count=t.change_count+changed.to(torch.long)
        cmd=self.zeros(n,3)
        singer=t.mode==1; dubins=t.mode==2; hybrid=t.mode==3
        decay=math.exp(-c['singer_alpha']*dt)
        ou=decay*t.ou_acc+c['singer_sigma']*math.sqrt(1-decay**2)*self.randn(n,3)
        ou[:,2]*=.985**(dt/.05)
        t.ou_acc=torch.where(singer[:,None],ou,t.ou_acc)
        cmd=torch.where(singer[:,None],t.ou_acc,cmd)
        turn_sign=torch.sign(t.desired_acc[:,0]);turn_sign=torch.where(turn_sign==0,1.,turn_sign)
        wmax=torch.minimum(torch.full_like(t.max_accel,c['turn_rate']),t.max_accel/t.speed_setting.clamp_min(1.))
        heading=t.heading+turn_sign*wmax*dt
        t.heading=torch.where(dubins,heading,t.heading)
        desired_vel=torch.stack((t.speed_setting*torch.cos(t.heading),t.speed_setting*torch.sin(t.heading),torch.zeros_like(t.heading)),1)
        cmd=torch.where(dubins[:,None],(desired_vel-t.vel)/dt,cmd)
        smoothed=t.ou_acc+(t.desired_acc-t.ou_acc)*min(1.,dt/c['hybrid_smoothing_s'])
        t.ou_acc=torch.where(hybrid[:,None],smoothed,t.ou_acc)
        cmd=torch.where(hybrid[:,None],t.ou_acc,cmd)
        xy=c['target_workspace_xy']
        cmd[:,:2]=cmd[:,:2]-torch.sign(t.pos[:,:2])*((t.pos[:,:2].abs()-.8*xy).clamp_min(0)*.08).clamp(max=c['max_accel'])
        lo,hi=-c['target_max_altitude'],-c['target_min_altitude']
        cmd[:,2]+=torch.where(t.pos[:,2]<lo+20,c['max_accel'],0.)
        cmd[:,2]-=torch.where(t.pos[:,2]>hi-20,c['max_accel'],0.)
        speed=norm(t.vel)
        proj=(cmd*t.vel).sum(1)/speed.square().clamp_min(1e-9)
        cap=torch.minimum(t.speed_setting+1,torch.full_like(speed,c['max_speed']))
        too_fast=(speed>=cap)&((cmd*t.vel).sum(1)>0)
        cmd=torch.where(too_fast[:,None],cmd-proj[:,None]*t.vel,cmd)
        cmd=cmd*torch.minimum(torch.ones_like(speed),t.max_accel/norm(cmd).clamp_min(1e-9))[:,None]
        if c.get('strict_speed_ceiling',False):
            delta=cmd*dt
            vnext=t.vel+delta
            a=delta.square().sum(1)
            b=2*(t.vel*delta).sum(1)
            cc=t.vel.square().sum(1)-c['max_speed']**2
            disc=(b*b-4*a*cc).clamp_min(0)
            alpha=((-b+torch.sqrt(disc))/(2*a.clamp_min(1e-16))).clamp(0,1)
            cmd=torch.where((norm(vnext)>c['max_speed']+1e-10)[:,None],cmd*alpha[:,None],cmd)
        old=t.vel
        t.vel=t.vel+cmd*dt
        t.pos=t.pos+t.vel*dt
        t.acc=(t.vel-old)/dt
        t.max_observed_speed=torch.maximum(t.max_observed_speed,norm(t.vel))
        t.max_observed_acc=torch.maximum(t.max_observed_acc,norm(t.acc))

    def _pursuer_step(self,action,dt):
        u=self.u;p=self.cfg['physics'];old=u.vel
        cmd_thrust=(p['hover_thrust']+p['thrust_delta']*action[:,0]).clamp(p['thrust_min'],p['thrust_max'])
        cmd_rate=action[:,1:]*self._rate_limits
        h=dt/int(p['substeps'])
        for _ in range(int(p['substeps'])):
            u.rate=u.rate+(cmd_rate-u.rate)*(1-torch.exp(-h/u.tau_rate))[:,None]
            u.thrust=u.thrust+(cmd_thrust-u.thrust)*(1-torch.exp(-h/u.tau_thrust))
            u.rot=torch.bmm(u.rot,exp_so3(u.rate*h,self._eye))
            a=-u.rot[:,:,2]*(u.thrust/u.mass_ratio)[:,None]
            a[:,2]+=p['gravity']
            a=a-u.drag[:,None]*u.vel*norm(u.vel)[:,None]
            u.vel=u.vel+a*h
            u.pos=u.pos+u.vel*h
        u.last_action=action
        u.acceleration=(u.vel-old)/dt
        return old

    def _observations(self):
        u,t=self.u,self.t;cfg=self.cfg;n=self.n
        rel=t.pos-u.pos;rv=t.vel-u.vel
        dc,tc=cpa(rel,rv,cfg['reward']['cpa_horizon'])
        obs=cfg['observation'];rew=cfg['reward'];p=cfg['physics']
        arr=[body_frame(u.rot,rel)/rew['distance_scale'],body_frame(u.rot,rv)/rew['speed_scale'],body_frame(u.rot,t.acc)/obs['target_accel_scale'],body_frame(u.rot,u.vel)/rew['speed_scale'],u.rot.reshape(n,9),u.rate/max(p['rate_xy'],p['rate_yaw']),u.last_action,(dc/rew['distance_scale'] if obs.get('use_cpa',True) else self.zeros(n))[:,None],(tc/rew['cpa_horizon'] if obs.get('use_cpa',True) else self.zeros(n))[:,None],((self.stage_config['time']-self.elapsed)/self.stage_config['time'])[:,None]]
        if obs.get('boundary_features',False):
            low,hi=-p['max_altitude'],-p['min_altitude'];xy=p['workspace_xy']
            boundary=torch.cat((torch.where(u.pos[:,:2]>=0,xy-u.pos[:,:2],-xy-u.pos[:,:2]),torch.where(u.pos[:,2:3]>(low+hi)/2,hi-u.pos[:,2:3],low-u.pos[:,2:3])),1)
            arr.append(body_frame(u.rot,boundary)/rew['distance_scale'])
        actor=torch.cat(arr,1).to(torch.float32)
        onehot=F.one_hot(t.mode,4).to(self.dtype)
        privileged=torch.cat(((u.thrust/p['hover_thrust'])[:,None],u.rate/max(p['rate_xy'],1e-6),(u.drag/max(p['drag'],1e-6))[:,None],(u.tau_rate/p['tau_rate'])[:,None],(u.tau_thrust/p['tau_thrust'])[:,None],u.mass_ratio[:,None],onehot),1)
        return actor,torch.cat((actor,privileged.to(torch.float32)),1)

    def observe(self):return self._observations()
    def load_factor(self):return self.u.thrust/(self.u.mass_ratio*self.cfg['physics']['gravity']).clamp_min(1e-8)

    @torch.no_grad()
    def step(self,action,observe=True,finite_horizon=False):
        if not isinstance(action,torch.Tensor): action=torch.as_tensor(action,device=self.device,dtype=self.dtype)
        action=action.to(self.device,self.dtype).clamp(-1,1)
        if action.shape!=(self.n,4):raise ValueError('Expected (N,4) CTBR')
        dt=self.cfg['physics']['dt'];p=self.cfg['physics'];rw=self.cfg['reward'];u=self.u;t=self.t
        r0=t.pos-u.pos
        previous_distance=norm(r0)
        oldpos=u.pos.clone();oldtarget=t.pos.clone();oldaction=u.last_action
        oldvel=self._pursuer_step(action,dt)
        self._target_step(dt)
        # Derive simulation time from the integer integration clock.  Repeated
        # float32 addition of 0.001 drifts by ~8.35 ms at 40,002 steps and
        # can prevent the 40 s timeout from firing within the evaluation budget.
        self.steps=self.steps+1
        self.elapsed=self.steps.to(self.dtype)*dt
        rel=t.pos-u.pos;rv=t.vel-u.vel;distance=norm(rel)
        self.min_distance=torch.minimum(self.min_distance,distance)
        vhat=oldvel/norm(oldvel,keepdim=True).clamp_min(1e-8)
        alat=u.acceleration-(u.acceleration*vhat).sum(1,keepdim=True)*vhat
        self.lateral_accel_cost_full=self.lateral_accel_cost_full+alat.square().sum(1)*dt
        self.control_variation_full=self.control_variation_full+norm(action-oldaction)*(self.steps>1)
        tilt=torch.acos(u.rot[:,2,2].clamp(-1,1));speed=norm(u.vel)
        alt=-u.pos[:,2]
        outside=(u.pos[:,:2].abs()>p['workspace_xy']).any(1)|(alt<p['min_altitude'])|(alt>p['max_altitude'])
        fail=outside|(tilt>math.radians(p['max_tilt_deg']))|(speed>p['max_speed'])
        hit,frac=segment_hit(r0,rel,self.stage_config['success_radius'])
        terminated=fail|hit
        truncated=(self.elapsed>=self.stage_config['time']-1e-8)&~terminated
        tilt_util=tilt/math.radians(p['max_tilt_deg'])
        speed_util=speed/p['max_speed']
        load_util=self.load_factor()/(p['thrust_max']/p['gravity'])
        rates=(u.rate/self._rate_limits).abs()
        high=(tilt_util>=.9)|(speed_util>=.9)|(load_util>=.9)|(rates.amax(1)>=.9)
        self.high_envelope_count_full=self.high_envelope_count_full+high
        dc,_=cpa(rel,rv,rw['cpa_horizon'])
        cp=-dc/rw['distance_scale'];near=torch.exp(-distance/rw['precision_scale_m'])
        closing=-(rel*rv).sum(1)/distance.clamp_min(1e-8)
        closing_w=rw['w_closing']
        if rw['closing_decay_distance_m']>0:closing_w=closing_w*torch.minimum(torch.ones_like(distance),distance/rw['closing_decay_distance_m'])
        speed_risk=((speed_util-rw['soft_speed_frac'])/(1-rw['soft_speed_frac'])).clamp(0,1).square()
        tilt_risk=((tilt_util-rw['soft_tilt_frac'])/(1-rw['soft_tilt_frac'])).clamp(0,1).square()
        shaping_terminal=(terminated|truncated) if finite_horizon else terminated
        potential=lambda prev,current,weight:weight*(self.cfg['training']['gamma']*torch.where(shaping_terminal,0.,current)-prev)
        components={
            'distance':-rw['w_distance']*dt*distance/rw['distance_scale'],
            'progress':rw['w_progress']*(previous_distance-distance)/rw['progress_scale_m'],
            'closing':closing_w*dt*(closing/rw['speed_scale']).clamp(-2,2),
            'cpa':potential(self.last_cpa_potential,cp,rw['w_cpa']) if rw['cpa_mode']=='potential' else -rw['w_cpa']*dt*dc/rw['distance_scale'],
            'action_delta':-rw['w_action_delta']*dt*(1+rw['near_action_delta_factor']*near)*(action-oldaction).square().sum(1),
            'time':torch.full_like(distance,-rw['w_time']*dt),
            'speed_risk':-dt*rw['w_speed_risk']*speed_risk,
            'tilt_risk':-dt*rw['w_tilt_risk']*tilt_risk,
            'boundary_risk':-dt*rw['w_boundary_risk']*boundary_risk(u.pos,p,rw['boundary_soft_distance']),
            'predictive_risk':-dt*rw['w_predictive_risk']*boundary_risk(u.pos+u.vel*rw['predict_horizon_s'],p,rw['boundary_soft_distance']),
            'precision':potential(self.last_precision_potential,near,rw['w_precision_potential']),
            'terminal':rw['success_bonus']*hit.to(self.dtype)-rw['failure_penalty']*(fail&~hit).to(self.dtype),
        }
        self.reward_components=components
        self.last_cpa_potential=cp;self.last_precision_potential=near
        reward=sum(components.values())
        self.episode_return=self.episode_return+reward
        self.last_event={'hit':hit,'frac':frac,'failed':fail,'speed':speed,'tilt':tilt,'outside':outside,'terminated':terminated,'truncated':truncated,'distance':distance,'pursuer_previous_pos':oldpos,'target_previous_pos':oldtarget}
        return self.observe() if observe else (None,None),reward.to(torch.float32),terminated,truncated,[]

    def snapshot(self):
        u,t=self.u,self.t
        rel=t.pos-u.pos;rv=t.vel-u.vel;dc,tc=cpa(rel,rv,self.cfg['reward']['cpa_horizon'])
        fields={'t':self.elapsed,'pursuer_pos':u.pos,'pursuer_vel':u.vel,'pursuer_rot':u.rot,'pursuer_rates':u.rate,'thrust':u.thrust,'pursuer_acc':u.acceleration,'load_factor':self.load_factor(),'target_pos':t.pos,'target_vel':t.vel,'target_acc':t.acc,'relative_distance':norm(rel),'relative_velocity':rv,'cpa_distance':dc,'cpa_time':tc,'action':u.last_action}
        if hasattr(self,'commanded_action'):
            fields['commanded_action']=self.commanded_action
        if hasattr(self,'navigation'):
            fields['observed_relative_distance']=norm(self.navigation.target.pos-self.navigation.pursuer.pos)
        return {k:v.detach().cpu().numpy().copy() for k,v in fields.items()}

    def completed_records(self, dones):
        """Return only completed episodes, with a single synchronization per rollout.

        The high-throughput training loop uses this for JSONL logging; routine
        environment steps never materialize episode records on the host.
        """
        raise NotImplementedError('Use event tensors recorded by the tensor trainer')
