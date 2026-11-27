import numpy as np
import torch
from .envs import make_env

def evaluate(name,policy,seed=10000,episodes=8,gamma=.99):
    rows=[]
    for k in range(episodes):
        env=make_env(name,seed+k)
        try:
            obs=env.reset(seed+k); total=disc=0.; errors=[]; efforts=[]; acts=[]; errors_dim=[]
            for t in range(env.horizon):
                obs,r,term,trunc,info=env.step(policy(obs)); total+=r; disc+=gamma**t*r
                if info['error'] is not None: errors.append(info['error'])
                if 'error_by_dim' in info: errors_dim.append(info['error_by_dim'])
                efforts.append(info['effort']); acts.append(np.atleast_1d(info['action']))
                if term or trunc: break
            actions=np.stack(acts)
            rows.append(dict(seed=seed+k,return_=total,discounted_return=disc,mean_error=float(np.mean(errors)) if errors else None,effort=float(np.mean(efforts)),action_rms=float(np.sqrt(np.mean(actions**2))),action_mean_by_dim=actions.mean(0).tolist(),action_std_by_dim=actions.std(0).tolist(),action_saturation_by_dim=(np.abs(actions)>=.98).mean(0).tolist(),error_by_dim=np.mean(errors_dim,axis=0).tolist() if errors_dim else None,steps=t+1,terminated=bool(term),truncated=bool(trunc)))
        finally: env.close()
    result={'episodes':rows}
    for k in ('return_','discounted_return','mean_error','effort','action_rms','action_mean_by_dim','action_std_by_dim','action_saturation_by_dim','error_by_dim'):
        v=[r[k] for r in rows if r[k] is not None]; result[k]=np.mean(v,axis=0).tolist() if v else None
    return result

def model_action(model,device):
    @torch.no_grad()
    def fn(obs): return model(torch.as_tensor(obs,dtype=torch.float32,device=device)).cpu().numpy()
    return fn
