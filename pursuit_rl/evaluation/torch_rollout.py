"""GPU-native PPO/PN/APN evaluation with optional stride-decimated CPU trace export."""
import math
import numpy as np
import torch
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.simulation.disturbances import validate_disturbance
from pursuit_rl.simulation.torch_disturbances import TorchSensorChannel,TorchCommandChannel
from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.control.torch_guidance import TorchGuidanceController
from pursuit_rl.artifacts.trace import TraceRecorder
from pursuit_rl.config import config_hash
from pursuit_rl.simulation.targets import MODES
from pursuit_rl.evaluation.metrics.statistics import mission_summary
MODE_NAMES={v:k for k,v in MODES.items()}
EVALUATION_VERSION='pursuit-evaluation-v6-batch64'

def task_hash(cfg,stage):
    return config_hash({k:cfg[k] for k in ('physics','target')} | {'stage':cfg['curriculum'][stage],'simulation_backend':cfg.get('simulation_backend','numpy')})

def summarize(results,stage):
    stats=mission_summary([{'success':r['success'],'reason':r['reason'],'capture_time':r['reach_time'],'capped_time':r['capped_time']} for r in results])
    bymode={}
    for name in sorted({r['target_mode_name'] for r in results}):
        sub=[r for r in results if r['target_mode_name']==name]
        bymode[name]={'episodes':len(sub),'success_rate':sum(r['success'] for r in sub)/len(sub),
                      'mean_capped_time':float(np.mean([r['capped_time'] for r in sub]))}
    return {'stage':stage,'episodes':len(results),'success_rate':stats['success_rate'],
            'mean_capped_time':stats['mean_capped_time'],
            'mean_reach_time_success':stats['mean_capture_time_success'],
            'modes':bymode,'failures':{k:stats['failures_'+k] for k in ('speed','outside','tilt','timeout')},
            'episodes_detail':results}

from pursuit_rl.learning.contracts import policy_contract

@torch.no_grad()
def evaluate_torch(model,cfg,stage=None,episodes=32,seed=25000,trace_dir=None,disturbance=None):
    disturbance=validate_disturbance(disturbance,cfg['physics']['dt'])
    stage=cfg['training']['max_stage'] if stage is None else int(stage)
    n=min(episodes,int(cfg['training'].get('num_envs',64)))
    from pursuit_rl.simulation.torch_device import resolve_torch_device
    device=resolve_torch_device(cfg['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    results=[]
    for start in range(0,episodes,n):
        count=min(n,episodes-start)
        env=TorchPursuitBatch(cfg,n=count,seed=seed+start,stage=stage,randomize=True,device=device)
        obs,priv=env.observe()
        disturbed=any(disturbance.values())
        sensor=TorchSensorChannel(cfg,disturbance,seed+start,device) if disturbed else None
        channel=TorchCommandChannel(cfg,disturbance,count,device)
        if sensor:obs=sensor.observe(env,obs)
        env.commanded_action=torch.zeros(count,4,device=device)
        if isinstance(model,GuidanceController):
            policy=TorchGuidanceController(model.algorithm,model.parameters)
        else:
            policy=model
            if not hasattr(policy,'predict_tensor'):
                raise ValueError('Tensor evaluator requires PPO model or PN/APN baseline')
            actual_policy_device = next(policy.actor.parameters()).device
            if actual_policy_device != device:
                raise ValueError(
                    f"Policy and environment must reside on same device: "
                    f"policy={actual_policy_device}, environment={device}"
                )
        recorder=TraceRecorder(env,trace_dir) if trace_dir is not None else None
        finished=torch.zeros(count,dtype=torch.bool,device=device)
        # First completion values kept entirely on device until final export.
        fields={}
        def frozen(key,value,new):
            prev=fields.get(key)
            if prev is None:prev=torch.zeros_like(value)
            fields[key]=torch.where(new.reshape((-1,)+(1,)*(value.ndim-1)),value,prev)
        interval=int(cfg['physics'].get('policy_interval_steps',1))
        horizon=int(math.ceil(env.stage_config['time']/cfg['physics']['dt']))+2
        held=None
        for tick in range(horizon):
            if held is None or tick%interval==0:
                held=policy.act(env,obs) if isinstance(policy,TorchGuidanceController) else policy.predict_tensor(obs)
            env.commanded_action=held
            executed=channel.apply(held)
            (nextobs,_),r,term,trunc,_=env.step(executed)
            if sensor: nextobs=sensor.observe(env,nextobs)
            done=term|trunc
            new=done&~finished
            e=env.last_event
            for k,value in {
                'success':e['hit'],'reason_speed':e['speed']>cfg['physics']['max_speed'],
                'reason_outside':e['outside'],'reason_failed':e['failed'],
                'reach_time':env.elapsed-cfg['physics']['dt']+e['frac']*cfg['physics']['dt'],
                'time':env.elapsed,'distance':e['distance'],'minimum_distance':env.min_distance,
                'speed':e['speed'],'tilt':e['tilt'],'return':env.episode_return,
                'target_mode':env.target.mode,'maneuvers':env.target.change_count,
                'target_speed_peak':env.target.max_observed_speed,'target_acc_peak':env.target.max_observed_acc,
                'control_variation_full':env.control_variation_full,
                'lateral_accel_cost_full':env.lateral_accel_cost_full,
                'high_envelope_any_ratio_full':env.high_envelope_count_full/env.steps.clamp_min(1),
                'capture_pursuer_pos':e['pursuer_previous_pos']+e['frac'][:,None]*(env.pursuer.pos-e['pursuer_previous_pos']),
                'capture_target_pos':e['target_previous_pos']+e['frac'][:,None]*(env.target.pos-e['target_previous_pos']),
            }.items():frozen(k,value,new)
            # TraceRecorder transfers states only at the configured stride and at completion.
            if recorder is not None:
                finished_cpu=finished.cpu().numpy();done_cpu=done.cpu().numpy()
                recorder.record(env,finished_cpu,done_cpu)
            finished=finished|done
            obs=nextobs
            if tick%64==0 and bool(finished.all().item()):break
        if not bool(finished.all().item()):
            pending=(~finished).nonzero(as_tuple=True)[0]
            pending_elapsed=env.elapsed[pending].detach().cpu().tolist()
            pending_steps=env.steps[pending].detach().cpu().tolist()
            raise RuntimeError(
                f'Tensor evaluation failed to finish {len(pending_elapsed)} of {count} '
                f'episodes (stage={stage}, limit={env.stage_config["time"]} s, '
                f'dt={cfg["physics"]["dt"]} s, horizon={horizon} ticks, '
                f'first unfinished local indices={pending[:8].cpu().tolist()}, '
                f'elapsed={pending_elapsed[:8]}, steps={pending_steps[:8]}). '
                f'Check simulation clock and termination masks.'
            )
        field_cpu={k:v.cpu().numpy() for k,v in fields.items()}
        ini_rel=env.initial_rel_pos.cpu().numpy()
        ini_tvel=env.initial_target_vel.cpu().numpy()
        ini_dist=env.initial_distance.cpu().numpy()
        ini_closing=env.initial_closing.cpu().numpy()
        ini_cpa=env.initial_cpa.cpu().numpy()
        for i in range(count):
            success=bool(field_cpu['success'][i]);f=field_cpu
            if success:reason='success'
            elif f['reason_speed'][i]:reason='speed'
            elif f['reason_outside'][i]:reason='outside'
            elif f['reason_failed'][i]:reason='tilt'
            else:reason='timeout'
            reach=float(f['reach_time'][i]) if success else None
            record={'episode':start+i,'seed':seed+start,'target_mode_name':MODE_NAMES[int(f['target_mode'][i])],
                    'success':success,'reason':reason,'stage':stage,'time':float(f['time'][i]),
                    'reach_time':reach,'start_distance':float(ini_dist[i]),
                    'start_closing_speed':float(ini_closing[i]),'start_cpa_distance':float(ini_cpa[i]),
                    'start_rel_pos':ini_rel[i].tolist(),'start_target_vel':ini_tvel[i].tolist(),
                    'initial_geometry':env.stage_config.get('initial_geometry','random'),
                    'terminal_distance':float(f['distance'][i]),'minimum_distance':float(f['minimum_distance'][i]),
                    'speed':float(f['speed'][i]),'tilt_deg':float(np.rad2deg(f['tilt'][i])),
                    'return':float(f['return'][i]),'control_variation_full':float(f['control_variation_full'][i]),
                    'lateral_accel_cost_full':float(f['lateral_accel_cost_full'][i]),
                    'high_envelope_any_ratio_full':float(f['high_envelope_any_ratio_full'][i]),
                    'capture_pursuer_pos':f['capture_pursuer_pos'][i].tolist() if success else None,
                    'capture_target_pos':f['capture_target_pos'][i].tolist() if success else None,
                    'target_speed_peak':float(f['target_speed_peak'][i]),'target_acc_peak':float(f['target_acc_peak'][i]),
                    'target_maneuvers':int(f['maneuvers'][i]),'target_mode':int(f['target_mode'][i]),
                    'is_truncated':reason=='timeout','time_limit':float(env.stage_config['time']),
                    'capped_time':reach if success else float(env.stage_config['time'])}
            results.append(record)
            if recorder:recorder.save(i,record,cfg,stage,seed+start)
    report=summarize(sorted(results,key=lambda x:x['episode']),stage)
    report['evaluation_protocol']={
        'version':EVALUATION_VERSION,'seed':seed,'batch_size':n,'episodes':episodes,
        'stage':stage,'task_hash':task_hash(cfg,stage),
        'pursuer_max_speed':cfg['physics']['max_speed'],'policy_contract':policy_contract(cfg),
        'disturbance':disturbance,'physics_dt_s':cfg['physics']['dt'],
        'policy_dt_s':cfg['physics']['dt']*cfg['physics'].get('policy_interval_steps',1),
        'policy_frequency_hz':1/(cfg['physics']['dt']*cfg['physics'].get('policy_interval_steps',1)),
        'trace_stride':cfg.get('evaluation',{}).get('trace_stride',1),
        'simulation_backend':'torch','device':str(device),
    }
    return report
