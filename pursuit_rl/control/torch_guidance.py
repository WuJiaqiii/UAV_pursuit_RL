"""GPU-native PN/APN control; mirrors guidance.py equations and normalization."""
import math
import torch
from pursuit_rl.control.guidance import GuidanceParameters
def norm(x):
    return torch.linalg.vector_norm(x,dim=-1)

def unit(x):
    return x/torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-9)


class TorchGuidanceController:
    def __init__(self,algorithm,parameters=None):
        if algorithm not in ('PN','APN','PD'):
            raise ValueError('Torch guidance currently supports PN and APN')
        self.algorithm=algorithm
        self.parameters=parameters or GuidanceParameters()

    @torch.no_grad()
    def _pd(self,env):
        u,t=env.pursuer,env.target;p=env.cfg['physics']
        desired=(.2*(t.pos-u.pos)+.75*(t.vel-u.vel)).clamp(-7,7)
        roll=(desired[:,1]/p['gravity']).clamp(-.6,.6)
        pitch=(-desired[:,0]/p['gravity']).clamp(-.6,.6)
        crot=torch.atan2(u.rot[:,2,1],u.rot[:,2,2])
        cpitch=torch.asin(u.rot[:,0,2].clamp(-1,1))
        rates=torch.stack((3*(roll-crot),3*(pitch-cpitch)),1).clamp(-p['rate_xy'],p['rate_xy'])
        thrust=(p['hover_thrust']-desired[:,2]).clamp(p['thrust_min'],p['thrust_max'])
        return torch.stack(((thrust-p['hover_thrust'])/p['thrust_delta'],rates[:,0]/p['rate_xy'],rates[:,1]/p['rate_xy'],torch.zeros_like(roll)),1).clamp(-1,1).to(torch.float32)

    @torch.no_grad()
    def act(self,env,obs=None):
        param=self.parameters
        p=env.cfg['physics'];u=env.pursuer;t=env.target
        if self.algorithm=='PD':
            return self._pd(env)
        nav=getattr(env,'navigation',None)
        tu,pu=(nav.target,nav.pursuer) if nav is not None else (t,u)
        r=tu.pos-pu.pos;rv=tu.vel-pu.vel
        d=norm(r).clamp_min(1e-8)
        direction=r/d[:,None]
        closing=-(direction*rv).sum(1)
        los=torch.cross(r,rv,dim=1)/d.square()[:,None]
        pn=param.N*closing.clamp_min(0)[:,None]*torch.cross(los,direction,dim=1)
        if self.algorithm=='APN':
            trans=tu.acc-(tu.acc*direction).sum(1,keepdim=True)*direction
            pn=pn+param.Na*trans
        budget=param.speed_fraction*p['max_speed']
        closing_goal=(budget-(tu.vel*direction).sum(1)).clamp(0,param.closing_goal_max)
        desired=tu.vel+closing_goal[:,None]*direction
        desired=desired*torch.minimum(torch.ones_like(d),budget/norm(desired).clamp_min(1e-8))[:,None]
        a=param.velocity_gain*(desired-pu.vel)+pn
        a=a*torch.minimum(torch.ones_like(d),param.accel_limit/norm(a).clamp_min(1e-8))[:,None]
        # acceleration_to_action: nominal drag, target acceleration in NED.
        drag=p['drag']*pu.vel*norm(pu.vel)[:,None]
        f=a+drag
        f=f.clone();f[:,2]-=p['gravity']
        upward=(-f[:,2]).clamp_min(.1*p['gravity'])
        horizontal=norm(f[:,:2])
        tilt_limit=upward*math.tan(math.radians(.9*p['max_tilt_deg']))
        f[:,:2]=f[:,:2]*torch.minimum(torch.ones_like(d),tilt_limit/horizontal.clamp_min(1e-8))[:,None]
        f[:,2]=-upward
        b3=-unit(f)
        yaw=torch.atan2(u.rot[:,1,0],u.rot[:,0,0])
        h=torch.stack((torch.cos(yaw),torch.sin(yaw),torch.zeros_like(yaw)),1)
        b2=unit(torch.cross(b3,h,dim=1))
        b1=torch.cross(b2,b3,dim=1)
        desired_rot=torch.stack((b1,b2,b3),2)
        errmat=.5*(torch.bmm(desired_rot.transpose(1,2),u.rot)-torch.bmm(u.rot.transpose(1,2),desired_rot))
        err=torch.stack((errmat[:,2,1],errmat[:,0,2],errmat[:,1,0]),1)
        rates=-param.attitude_gain*err
        thrust=(f*(-u.rot[:,:,2])).sum(1).clamp(p['thrust_min'],p['thrust_max'])
        action=torch.cat((((thrust-p['hover_thrust'])/p['thrust_delta'])[:,None],rates/torch.tensor([p['rate_xy'],p['rate_xy'],p['rate_yaw']],device=env.device,dtype=env.dtype)),1)
        return action.clamp(-1,1).to(torch.float32)
