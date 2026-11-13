"""Compare cumulative imitation updates with fixed states and paired rollouts."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .behavior import acceptable, collect, fit, policy_action, reports, scan
from .checkpoint import load as load_model, save, source_fingerprint, write_json
from .config import interface, load
from .control import teacher
from .diagnostics import profiles
from .environment import CTBRBatch
from .evaluation import evaluate
from .trim import set_cruise_trim


@torch.no_grad()
def actions(model, observations):
    return torch.cat([
        model.distribution_parameters(batch)[0].tanh().cpu()
        for batch in observations.split(2048)
    ])


@torch.no_grad()
def recovery_buckets(model, task, episodes, seed, *, trim=False):
    """Branch from A at two late times and label teacher recovery."""
    env = CTBRBatch(dict(task, num_envs=episodes), seed=seed)
    snapshots = sorted({round(env.max_decision_steps * fraction) for fraction in (0.75, 0.85)})
    horizon = round(0.8 / env.decision_dt)
    if horizon < 2 or snapshots[-1] + horizon > env.max_decision_steps:
        raise ValueError('Cruise duration must allow a full 0.8 s recovery after the late snapshot')
    gaps = (-0.2, 0, 0.2) if trim else (0.3, 0.5, 0.8)
    buckets = {(gap, phase): ([], [], []) for gap in gaps for phase in (0, 1)}
    tick = 0
    for snapshot in snapshots:
        while tick < snapshot:
            env.step(*policy_action(model, env))
            tick += 1
        target_v = env.plant.target.vel.float()
        direction = torch.nn.functional.normalize(target_v, dim=1)
        rel_v = ((target_v - env.plant.pursuer.vel) * direction).sum(1)
        eligible = (~env.finished) & (rel_v.abs() <= 0.1)
        error = (env.plant.target.pos - env.plant.pursuer.pos).float()
        lag = (error * direction).sum(1)
        lateral = (error - lag[:, None] * direction).norm(dim=1)
        for gap in gaps:
            branch = copy.deepcopy(env)
            if trim:
                set_cruise_trim(branch, gap)
            else:
                branch.plant.pursuer.pos += (lag - gap)[:, None] * direction
            shift = ((branch.plant.pursuer.pos - env.plant.pursuer.pos) * direction).sum(1)
            actual_relative_velocity = ((target_v - branch.plant.pursuer.vel) * direction).sum(1)
            initial_velocity = branch.plant.pursuer.vel.clone()
            initial_position = branch.plant.pursuer.pos.clone()
            initial_target = branch.plant.target.pos.clone()
            terminal = {}
            collected = []
            for step in range(horizon):
                slots = (eligible & ~branch.finished).nonzero(as_tuple=True)[0]
                obs, _ = branch.observe()
                label = teacher(branch)
                if len(slots):
                    phase = int(step >= horizon // 2)
                    xs, ys, metadata = buckets[gap, phase]
                    xs.append(obs[slots].clone())
                    ys.append(label[slots].clone())
                    for slot in slots.tolist():
                        record = dict(snapshot_tick=snapshot, slot=slot, recovery_tick=step,
                                      initial_lag_m=float(lag[slot]), imposed_lag_m=gap,
                                      position_shift_m=float(shift[slot]),
                                      lateral_error_m=0.0 if trim else float(lateral[slot]),
                                      initial_relative_velocity_mps=float(actual_relative_velocity[slot]),
                                      trimmed=trim)
                        metadata.append(record)
                        collected.append(record)
                _, _, _, _, completed = branch.step(label)
                for row in completed:
                    terminal[row['slot']] = row['reason']
            dv = ((branch.plant.pursuer.vel-initial_velocity)*direction).sum(1)
            closure = (((branch.plant.pursuer.pos-initial_position)
                        -(branch.plant.target.pos-initial_target))*direction).sum(1)
            for record in collected:
                slot = record['slot']
                record.update(terminal_reason=terminal.get(slot),
                              teacher_speed_change_mps=float(dv[slot]),
                              teacher_closure_m=float(closure[slot]))
    return buckets


def collect_arm(model, tasks, args):
    mixed, mixed_info = collect(model, tasks['mixed'], args.episodes, 710042, stride=5)
    budget = args.samples
    if budget is None and args.data_mix == 'mixed':
        return mixed, dict(samples=len(mixed[0]), sources={'mixed': mixed_info})
    if budget is None or budget < 12 or budget % 12:
        raise ValueError('Matched comparisons require --samples divisible by 12')
    recovery_arm = args.data_mix in ('recovery', 'trim_recovery')
    if recovery_arm and budget % 144:
        raise ValueError('Recovery comparisons require --samples divisible by 144')
    chunks, sources = [], {}

    def add(name, data, info, count, seed):
        available = len(data[0])
        if available < count:
            raise ValueError(f'{name}: collected {available} samples for quota {count}; increase rollout episodes')
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(available, generator=rng)[:count].to(data[0].device)
        chunks.append(tuple(t[indices] for t in data))
        info = dict(info)
        rows = info.pop('rows', None)
        sources[name] = dict(info, quota=count, subsample_seed=seed, indices=indices.tolist())
        if rows is not None:
            sources[name]['selected_rows'] = [rows[i] for i in sources[name]['indices']]

    add('mixed', mixed, mixed_info, budget if args.data_mix == 'mixed' else budget // 2, 800)
    if args.data_mix != 'mixed':
        for index, task in enumerate(('cruise_0.5', 'cruise_1', 'cruise_2')):
            cruise, info = collect(model, tasks[task], args.cruise_episodes, 720042+index*1000, stride=5)
            add(task, cruise, info, budget // (6 if args.data_mix == 'cruise' else 12), 801+index)
            if args.data_mix == 'near_zero' or recovery_arm:
                near, info = collect(model, tasks[task], args.cruise_episodes,
                                     720042+index*1000, stride=5,
                                     teacher_rollout=True, max_label_distance=0.2)
                near_quota = budget // (24 if recovery_arm else 12)
                add(task+'_teacher_near', near, info, near_quota, 901+index)
                if recovery_arm:
                    buckets = recovery_buckets(model, tasks[task], args.recovery_episodes,
                                               740042+index*1000, trim=args.data_mix == 'trim_recovery')
                    for bucket, ((gap, phase), (xs, ys, rows)) in enumerate(buckets.items()):
                        name = f'{task}_recovery_{gap}_phase{phase}'
                        if not xs:
                            raise ValueError(f'{name}: no eligible recovery states; increase --recovery-episodes')
                        add(name, (torch.cat(xs), torch.cat(ys)), dict(rows=rows, available=len(rows)),
                            budget // 144, 1001+index*10+bucket)
    return tuple(torch.cat([chunk[i] for chunk in chunks]) for i in range(2)), dict(
        samples=budget, sources=sources)


@torch.no_grad()
def baseline_probe(model, task):
    env = CTBRBatch(dict(task, num_envs=8), seed=710042)
    ticks = min(round(35 / env.decision_dt), env.max_decision_steps - 1)
    for _ in range(ticks):
        env.step(*policy_action(model, env))
    obs, priv = env.observe()
    digest = hashlib.sha256(priv.cpu().contiguous().numpy().tobytes()).hexdigest()
    return env, digest, ticks * env.decision_dt


def run(args):
    if min(args.milestones) < 1 or sorted(set(args.milestones)) != args.milestones:
        raise ValueError('Milestones must be positive and strictly increasing')
    if min(args.episodes, args.screen_episodes, args.num_envs, args.validation_episodes) < 1:
        raise ValueError('Episode and environment counts must be positive')
    if min(args.cruise_episodes, args.recovery_episodes) < 1:
        raise ValueError('Collection episode counts must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Learning rate must be finite and positive')
    if not math.isfinite(args.anchor_weight) or args.anchor_weight < 0:
        raise ValueError('Anchor weight must be finite and nonnegative')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Use an empty experiment directory')
    model, original, saved = load_model(args.checkpoint, args.device)
    config = load(args.config, device=args.device)
    tasks = profiles(config, [0.5, 1, 2])
    for name in ('hover', 'acquisition30'):
        tasks[name] = load(f'configs/ctbr_v701/{name}.json', device=args.device)
    for task in tasks.values():
        if interface(original) != interface(task):
            raise ValueError('Checkpoint interface does not match an evaluation task')
        task['num_envs'] = args.num_envs
    screen = {name: dict(tasks[name], num_envs=args.screen_episodes)
              for name in ('hover', 'cruise_1')}
    seeds = [810042, 910042]
    write_json(root / 'manifest.json', dict(
        arguments=vars(args), tasks=tasks, screening_tasks=screen, train_seed=710042, fixed_state_seed=760042,
        validation_seeds=seeds, source_fingerprint=source_fingerprint(),
        torch_version=str(torch.__version__),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()))
    data, collection = collect_arm(model, tasks, args)
    probe_env, probe_digest, probe_time = baseline_probe(model, tasks['cruise_1'])
    torch.save(tuple(t.cpu() for t in data), root / 'training_samples.pt')
    write_json(root / 'collection.json', collection)
    fixed = {}
    for name, task in screen.items():
        (obs, labels), info = collect(model, task, args.screen_episodes, 760042, stride=20)
        fixed[name] = (obs, labels.cpu(), actions(model, obs))
        torch.save(dict(observations=obs.cpu(), teacher=labels.cpu(), baseline=fixed[name][2],
                        collection=info), root / f'fixed_{name}.pt')

    progress = dict(ppo_updates=0, physics_steps=0, critic_needs_calibration=True,
                    cumulative_ppo_physics_steps=saved['progress'].get('cumulative_ppo_physics_steps', 0))
    results = []

    def record_candidate(update):
        directory = root / f'update_{update:04d}'
        directory.mkdir()
        save(directory / 'candidate.pt', model, tasks['mixed'],
             dict(progress, behavior_updates=update), 'staged_behavior_candidate')
        write_json(directory / 'shared_probe.json', dict(
            initial_state_digest=probe_digest, start_time_s=probe_time, horizon_s=0.3,
            rows=scan(model, probe_env, [-0.5, 0, 0.5], 0.3)))
        screening = {}
        for name, task in screen.items():
            report, _ = evaluate(model, task, args.screen_episodes, seeds[0],
                                 trace_dir=directory / name, trace_count=args.screen_episodes)
            screening[name] = report
            obs, labels, baseline = fixed[name]
            predicted = actions(model, obs)
            torch.save(predicted, directory / f'actions_{name}.pt')
            report['fixed_state_action_error'] = dict(
                baseline_teacher_rmse=(baseline-labels).square().mean(0).sqrt().tolist(),
                candidate_teacher_rmse=(predicted-labels).square().mean(0).sqrt().tolist(),
                signed_change_from_baseline=(predicted-baseline).mean(0).tolist(),
                change_rmse=(predicted-baseline).square().mean(0).sqrt().tolist())
        write_json(directory / 'screening.json', screening)
        results.append(dict(update=update, tasks={
            name: dict(success_rate=r['success_rate'], rmse=r['mean_tail_position_rmse'],
                       joint_precision=r['joint_precision_success_rate'])
            for name, r in screening.items()}))
        write_json(root / 'screening_summary.json', results)

    def on_update(update, metrics):
        with (root / 'updates.jsonl').open('a') as log:
            log.write(json.dumps(dict(update=update, **metrics), allow_nan=False) + '\n')
        if update in args.milestones:
            record_candidate(update)

    record_candidate(0)
    torch.manual_seed(720042)
    fit(model, data, updates=args.milestones[-1], lr=args.lr,
        on_update=on_update, anchor_weight=args.anchor_weight)

    baseline_model, _, _ = load_model(args.checkpoint, args.device)
    baseline = reports(baseline_model, tasks, args.validation_episodes, seeds)
    write_json(root / 'baseline_validation.json', baseline)
    stable = [r for r in results if r['update'] and all(
        r['tasks'][k]['success_rate'] >= results[0]['tasks'][k]['success_rate']
        and r['tasks'][k]['rmse'] is not None for k in screen)]
    selected = None
    if stable:
        best = min(stable, key=lambda r: (
            -r['tasks']['cruise_1']['joint_precision'] if args.rank_by_precision else 0,
            r['tasks']['cruise_1']['rmse'], r['update']))
        path = root / f"update_{best['update']:04d}" / 'candidate.pt'
        candidate, _, _ = load_model(path, args.device)
        validation = reports(candidate, tasks, args.validation_episodes, seeds)
        write_json(root / 'candidate_validation.json', validation)
        if acceptable(baseline, validation):
            selected = str(path)
    write_json(root / 'summary.json', dict(
        nominated_update=best['update'] if stable else None, selected=selected,
        independent_test_seeds=[1210042, 1310042]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='configs/ctbr_v701/near_mixed.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--screen-episodes', type=int, default=8)
    parser.add_argument('--validation-episodes', type=int, default=32)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--milestones', type=int, nargs='+', default=[10, 25, 50, 100])
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--anchor-weight', type=float, default=0)
    parser.add_argument('--data-mix', choices=['mixed', 'cruise', 'near_zero', 'recovery', 'trim_recovery'], default='mixed')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--cruise-episodes', type=int, default=8)
    parser.add_argument('--recovery-episodes', type=int, default=12)
    parser.add_argument('--rank-by-precision', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
