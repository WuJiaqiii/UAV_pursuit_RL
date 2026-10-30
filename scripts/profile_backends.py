"""Measure full policy+plant simulation throughput, not just Actor forward latency.

Uses short vectorized trajectories. Results are device/hardware specific and
must not be confused with hard real-time 1ms CTBR latency guarantees.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from pursuit_rl.config import load_config
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.simulation.observations import observation_dimensions


def measure(cfg,backend,device,steps,warmup):
    cfg=copy.deepcopy(cfg)
    cfg['device']=device
    cfg['simulation_backend']=backend
    cfg['training']['num_envs']=max(cfg['training']['num_envs'],1)
    torch.manual_seed(cfg['seed'])
    n=cfg['training']['num_envs']
    model=AsymmetricPPO(*observation_dimensions(cfg),cfg['training'],device=device,create_optimizers=False)
    model.eval()
    if backend=='torch':
        env=TorchPursuitBatch(cfg,device=device)
        obs,priv=env.observe()
        def tick():
            nonlocal obs,priv
            with torch.no_grad():
                action=model.predict_tensor(obs)
                (obs,priv),_,done,trunc,_=env.step(action)
                if bool((done|trunc).any().item()):
                    env.reset(done|trunc)
                    obs,priv=env.observe()
    else:
        env=PursuitBatch(cfg)
        obs,priv=env.observe()
        def tick():
            nonlocal obs,priv
            action=model.predict(obs)
            (obs,priv),_,done,trunc,_=env.step(action)
            if (done|trunc).any():
                env.reset(done|trunc)
                obs,priv=env.observe()
    for _ in range(warmup):tick()
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(steps):tick()
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return {'backend':backend,'device':device,'num_envs':n,'policy_steps':steps,
            'physical_dt_s':cfg['physics']['dt'],'wall_seconds':elapsed,
            'policy_batch_updates_per_second':steps/elapsed,
            'environment_steps_per_second':steps*n/elapsed,
            'meets_1ms_batch_budget_on_this_host':elapsed/steps<=.001}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/high_frequency/ppo_1khz.yaml')
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    ap.add_argument('--backend',choices=('numpy','torch'),default='torch')
    ap.add_argument('--num-envs',type=int,default=64)
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--warmup',type=int,default=30)
    ap.add_argument('--out',default='runs/backend_profile.json')
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    cfg=load_config(args.config)
    cfg['training']['num_envs']=args.num_envs
    result=measure(cfg,args.backend,args.device,args.steps,args.warmup)
    result.update(torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(0) if args.device=='cuda' else None)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
