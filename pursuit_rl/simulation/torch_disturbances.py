"""Device-resident evaluation noise/delay channels. Uses torch RNG per case."""
from collections import deque
from types import SimpleNamespace
import torch
from pursuit_rl.simulation.torch_backend import body_frame,cpa

class TorchSensorChannel:
    def __init__(self,cfg,spec,seed,device):
        self.cfg=cfg;self.spec=spec
        self.lag=round(spec['observation_delay_s']/cfg['physics']['dt'])
        self.states=deque(maxlen=self.lag+1)
        self.gen=torch.Generator(device=device).manual_seed(int(seed)+460001)
        self.device=device
    def observe(self,env,original):
        state={k:getattr(obj,attr).clone() for k,obj,attr in (
            ('ppos',env.pursuer,'pos'),('pvel',env.pursuer,'vel'),('tpos',env.target,'pos'),('tvel',env.target,'vel'),('tacc',env.target,'acc'))}
        self.states.append(state)
        src=self.states[0]
        def noise(key,std):
            x=src[key]
            if not std:return x
            return x+torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=self.gen)*std
        s=self.spec
        pp=noise('ppos',s['position_std_m']);tp=noise('tpos',s['position_std_m'])
        pv=noise('pvel',s['velocity_std_mps']);tv=noise('tvel',s['velocity_std_mps'])
        ta=noise('tacc',s['target_accel_std_mps2'])
        env.navigation=SimpleNamespace(pursuer=SimpleNamespace(pos=pp,vel=pv),target=SimpleNamespace(pos=tp,vel=tv,acc=ta))
        rel=tp-pp;rv=tv-pv;rot=env.pursuer.rot
        actor=original.clone()
        actor[:,0:3]=body_frame(rot,rel)/env.cfg['reward']['distance_scale']
        actor[:,3:6]=body_frame(rot,rv)/env.cfg['reward']['speed_scale']
        actor[:,6:9]=body_frame(rot,ta)/env.cfg['observation']['target_accel_scale']
        actor[:,9:12]=body_frame(rot,pv)/env.cfg['reward']['speed_scale']
        if env.cfg['observation']['use_cpa']:
            dc,tc=cpa(rel,rv,env.cfg['reward']['cpa_horizon'])
            actor[:,28]=dc/env.cfg['reward']['distance_scale']
            actor[:,29]=tc/env.cfg['reward']['cpa_horizon']
        return actor

class TorchCommandChannel:
    def __init__(self,cfg,spec,n,device):
        self.lag=round(spec['command_delay_s']/cfg['physics']['dt'])
        self.hist=deque(maxlen=self.lag+1)
    def apply(self,action):
        self.hist.append(action.clone())
        if len(self.hist)<=self.lag:return torch.zeros_like(action)
        return self.hist[0]
