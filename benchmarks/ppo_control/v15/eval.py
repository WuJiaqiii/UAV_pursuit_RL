import numpy as np
import torch
from .envs import Bench

def evaluate(name,policy,seed=10000,episodes=8,gamma=.99):
    rows=[]
    for k in range(episodes):
        env=Bench(name,seed+k); obs=env.reset(); total=disc=err=eff=0.; acts=[]
        for t in range(env.horizon):
            a=policy(obs); obs,r,term,trunc,info=env.step(a); total+=r; disc+=gamma**t*r; err+=info['error']; eff+=info['effort']; acts.append(info['action'])
            if term or trunc: break
        rows.append(dict(seed=seed+k,return_=total,discounted_return=disc,mean_error=err/(t+1),effort=eff/(t+1),action_rms=float(np.sqrt(np.mean(np.square(acts)))),steps=t+1,terminated=term,truncated=trunc))
    return {'episodes':rows,**{k:float(np.mean([r[k] for r in rows])) for k in ('return_','discounted_return','mean_error','effort','action_rms')}}

def model_action(model,device):
    @torch.no_grad()
    def fn(obs): return model(torch.as_tensor(obs,dtype=torch.float32,device=device)).cpu().numpy()
    return fn
