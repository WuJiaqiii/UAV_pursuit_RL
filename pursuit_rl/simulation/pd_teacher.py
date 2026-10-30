"""Training-only PD teacher based on current pursuer/target state.

Same CTBR heuristic as control.guidance.pd_intercept and
control.torch_guidance.TorchGuidanceController('PD'), but located in the
simulation layer to preserve package dependency direction. No privileged
future state, no teacher action mixing at environment execution or evaluation.
"""
import numpy as np
import torch


def pd_teacher(env):
    u, t, p = env.pursuer, env.target, env.cfg["physics"]
    if isinstance(u.pos, torch.Tensor):
        with torch.no_grad():
            desired=(.2*(t.pos-u.pos)+.75*(t.vel-u.vel)).clamp(-7,7)
            roll=(desired[:,1]/p['gravity']).clamp(-.6,.6)
            pitch=(-desired[:,0]/p['gravity']).clamp(-.6,.6)
            crot=torch.atan2(u.rot[:,2,1],u.rot[:,2,2])
            cpitch=torch.asin(u.rot[:,0,2].clamp(-1,1))
            rates=torch.stack((3*(roll-crot),3*(pitch-cpitch)),1).clamp(-p['rate_xy'],p['rate_xy'])
            thrust=(p['hover_thrust']-desired[:,2]).clamp(p['thrust_min'],p['thrust_max'])
            return torch.stack(((thrust-p['hover_thrust'])/p['thrust_delta'],rates[:,0]/p['rate_xy'],rates[:,1]/p['rate_xy'],torch.zeros_like(roll)),1).clamp(-1,1).to(torch.float32)
    # NumPy match for original guidance PD.
    desired=np.clip(.2*(t.pos-u.pos)+.75*(t.vel-u.vel),-7.,7.)
    roll=np.clip(desired[:,1]/p['gravity'],-.6,.6)
    pitch=np.clip(-desired[:,0]/p['gravity'],-.6,.6)
    crot=np.arctan2(u.rot[:,2,1],u.rot[:,2,2])
    cpitch=np.arcsin(np.clip(u.rot[:,0,2],-1,1))
    rates=np.stack((3*(roll-crot),3*(pitch-cpitch)),1)
    rates=np.clip(rates,-p['rate_xy'],p['rate_xy'])
    thrust=np.clip(p['hover_thrust']-desired[:,2],p['thrust_min'],p['thrust_max'])
    return np.stack(((thrust-p['hover_thrust'])/p['thrust_delta'],rates[:,0]/p['rate_xy'],rates[:,1]/p['rate_xy'],np.zeros_like(roll)),1).clip(-1.,1.).astype(np.float32)
