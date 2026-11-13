"""Physical trim and short-horizon steady-offset diagnostics for cruise tasks."""

import argparse
import copy
import hashlib
import math
from pathlib import Path

import torch

from .behavior import policy_action
from .checkpoint import load as load_model, source_fingerprint, write_json
from .config import interface, load
from .control import teacher, thrust_to_action
from .diagnostics import profiles
from .environment import CTBRBatch
from .evaluation import evaluate


@torch.no_grad()
def set_cruise_trim(env, gap=0):
    """Match target velocity and balance gravity and quadratic drag in NED."""
    u, t = env.plant.pursuer, env.plant.target
    p = env.config['physics']
    velocity = t.vel.to(u.vel)
    speed = velocity.norm(dim=1, keepdim=True)
    if env.config['task']['kind'] != 'cruise' or bool((speed < 0.2).any()):
        raise ValueError('Trim requires a cruise task and target speeds of at least 0.2 m/s')
    direction = velocity / speed
    force = u.drag[:, None] * velocity * speed
    force[:, 2] -= p['gravity']
    force_norm = force.norm(dim=1)
    thrust = u.mass_ratio * force_norm
    if bool(((thrust <= p['thrust_min']) | (thrust >= p['thrust_max'])).any()):
        raise ValueError('Trim thrust exceeds the plant limits')

    yaw = torch.atan2(u.rot[:, 1, 0], u.rot[:, 0, 0])
    heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=1)
    b3 = -force / force_norm[:, None]
    cross = torch.linalg.cross(b3, heading, dim=1)
    b2 = torch.nn.functional.normalize(cross, dim=1)
    b1 = torch.linalg.cross(b2, b3, dim=1)
    u.rot.copy_(torch.stack((b1, b2, b3), dim=2))
    u.pos.copy_(t.pos - gap * direction)
    u.vel.copy_(velocity)
    u.acc.zero_()
    u.rate.zero_()
    u.thrust.copy_(thrust)
    t.acc.zero_()

    action = torch.zeros_like(env.raw_history[:, 0])
    action[:, 0] = thrust_to_action(thrust, p).clamp(-0.999999, 0.999999)
    raw = action.atanh()
    env.raw_history.copy_(raw[:, None].expand_as(env.raw_history))
    history = env.target_history
    age = torch.arange(history.shape[1] - 1, -1, -1, device=history.device)
    history[:, :, :3] = t.pos.float()[:, None] - (
        age[None, :, None] * env.decision_dt * velocity[:, None]
    )
    history[:, :, 3:6] = velocity[:, None]
    history[:, :, 6:9] = 0
    history[:, :, 9] = 1
    return action, raw, direction


@torch.no_grad()
def response(model, env, direction, *, fixed_action=None, horizons=(0.1, 0.3, 1.0)):
    """Record time, actual plant response and termination from an isolated copy."""
    branch = copy.deepcopy(env)
    u, t = branch.plant.pursuer, branch.plant.target
    initial_velocity = u.vel.clone()
    start_gap = ((t.pos-u.pos) * direction).sum(1)
    start_steps = branch.steps.clone()
    sample_ticks = {round(h/branch.decision_dt) for h in horizons}
    if min(sample_ticks) < 1 or max(sample_ticks) > branch.max_decision_steps:
        raise ValueError('Probe horizon is outside the episode')
    terminal = {}
    samples = []
    first_action = None
    for tick in range(1, max(sample_ticks)+1):
        if fixed_action is not None:
            action, raw = fixed_action
        elif model is None:
            action, raw = teacher(branch), None
        else:
            action, raw = policy_action(model, branch)
        if first_action is None:
            first_action = action.tolist()
        _, _, _, _, completed = branch.step(action, raw)
        terminal.update({row['slot']:row['reason'] for row in completed})
        if tick in sample_ticks:
            gap = ((t.pos-u.pos)*direction).sum(1)
            samples.append(dict(
                nominal_time_s=tick*branch.decision_dt,
                elapsed_s=((branch.steps-start_steps)*branch.dt).tolist(),
                gap_m=gap.tolist(), gap_change_m=(gap-start_gap).tolist(),
                speed_change_mps=((u.vel-initial_velocity)*direction).sum(1).tolist(),
                along_acceleration_mps2=(u.acc*direction).sum(1).tolist(),
                rotation=u.rot.tolist(), thrust=u.thrust.tolist(),
                body_rate=u.rate.tolist(), finished=branch.finished.tolist()))
    return dict(first_action=first_action, samples=samples, terminal=terminal)


@torch.no_grad()
def sensitivity(model, env, direction, delta=0.05):
    """Central differences in physical units; synthetic inputs are offline only."""
    obs, _ = env.observe()
    base, _ = model.distribution_parameters(obs)
    result = {}
    for label, start, scale in (
        ('position_error_m', 0, 30), ('relative_velocity_mps', 3, 15),
        ('own_velocity_mps', 9, 15),
    ):
        lo, hi = obs.clone(), obs.clone()
        lo[:, start:start+3] -= delta*direction/scale
        hi[:, start:start+3] += delta*direction/scale
        lo_mean, _ = model.distribution_parameters(lo)
        hi_mean, _ = model.distribution_parameters(hi)
        lo_raw = model.actor(model.bound_actor_input(model.actor_features(lo)))
        hi_raw = model.actor(model.bound_actor_input(model.actor_features(hi)))
        result[label] = dict(
            actor_raw_slope=((hi_raw-lo_raw)/(2*delta)).tolist(),
            executed_action_slope=((hi_mean.tanh()-lo_mean.tanh())/(2*delta)).tolist())
    no_history = obs.clone()
    history_start = model.memory_start
    history_width = env.raw_history.shape[1]*env.raw_history.shape[2]
    no_history[:, history_start:history_start+history_width] = 0
    changed, _ = model.distribution_parameters(no_history)
    result['zero_raw_history_action_change'] = (changed.tanh()-base.tanh()).tolist()
    return result


def run(args):
    if (args.scenarios < 1 or args.full_episodes < 0 or not args.checkpoint
            or any(not math.isfinite(x) or x < 0 for x in args.drag_factors)
            or any(not math.isfinite(x) or x < 0.2 for x in args.speeds)):
        raise ValueError('Invalid scene count, speed, or drag factor')
    base = load(args.config, device=args.device)
    models, provenance = {}, {}
    for spec in args.checkpoint:
        name, separator, path = spec.partition('=')
        if not separator or not name or name in models or name in ('teacher', 'fixed'):
            raise ValueError('Use unique NAME=PATH checkpoint arguments')
        model, config, _ = load_model(path, args.device)
        if interface(base) != interface(config):
            raise ValueError(f'Checkpoint interface differs from nominal config: {name}')
        models[name] = model
        provenance[name] = dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Use an empty diagnostic output directory')
    write_json(root/'manifest.json', dict(arguments=vars(args), checkpoints=provenance,
               config=base, source_fingerprint=source_fingerprint(), torch_version=str(torch.__version__)))
    summary = []
    for speed in args.speeds:
        for factor in args.drag_factors:
            config = profiles(base, [speed])[f'cruise_{speed:g}']
            config['num_envs'] = args.scenarios
            config['physics']['drag'] *= factor
            directory = root / f'cruise_{speed:g}' / f'drag_{factor:g}'
            for gap in (-0.2, 0, 0.2, 0.5, 0.8):
                env = CTBRBatch(config, seed=args.seed)
                action, raw, direction = set_cruise_trim(env, gap)
                fixed = response(None, env, direction, fixed_action=(action, raw), horizons=(0.3,))
                sample = fixed['samples'][0]
                trim_error = max(map(abs, sample['speed_change_mps']))
                gap_error = max(map(abs, sample['gap_change_m']))
                record = dict(speed_mps=speed, drag_factor=factor, gap_m=gap,
                    config=config, initial=dict(
                        target_velocity=env.plant.target.vel.tolist(),
                        pursuer_position=env.plant.pursuer.pos.tolist(),
                        target_position=env.plant.target.pos.tolist(),
                        mass_ratio=env.plant.pursuer.mass_ratio.tolist(),
                        drag=env.plant.pursuer.drag.tolist(),
                        rotation=env.plant.pursuer.rot.tolist(),
                        thrust=env.plant.pursuer.thrust.tolist(), action=action.tolist()),
                    trim_check=fixed)
                output = directory / f'gap_{gap:g}.json'
                if trim_error >= 0.005 or gap_error >= 0.005 or fixed['terminal']:
                    write_json(output, record)
                    raise RuntimeError(f'Physical trim check failed: {output}')
                record['teacher'] = response(None, env, direction)
                record['policies'] = {name:dict(response=response(model, env, direction),
                    sensitivity=sensitivity(model, env, direction)) for name,model in models.items()}
                write_json(output, record)
                summary.append(dict(speed_mps=speed, drag_factor=factor, gap_m=gap,
                    mean_speed_change_03={name:sum(value['samples'][1]['speed_change_mps'])/env.n
                        for name,value in [('teacher',record['teacher']), *((n,r['response']) for n,r in record['policies'].items())]}))
            if args.full_episodes:
                for index, (name, model) in enumerate([('teacher',None), *models.items()]):
                    report, _ = evaluate(model, config, args.full_episodes, args.seed+100000,
                                         reference=model is None)
                    write_json(directory / f'full_{index}.json', dict(controller=name, report=report))
            print(f'{speed:g} m/s, drag {factor:g}: completed', flush=True)
    write_json(root/'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--config', default='configs/ctbr_v701/near_mixed.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--speeds', type=float, nargs='+', default=[1])
    parser.add_argument('--drag-factors', type=float, nargs='+', default=[0, 0.5, 1])
    parser.add_argument('--scenarios', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1700042)
    parser.add_argument('--full-episodes', type=int, default=0)
    torch.set_num_threads(1)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
