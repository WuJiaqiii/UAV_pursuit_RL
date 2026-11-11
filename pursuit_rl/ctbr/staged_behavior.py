"""Compare cumulative imitation updates with fixed states and paired rollouts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .behavior import acceptable, collect, fit, reports
from .checkpoint import load as load_model, save, source_fingerprint, write_json
from .config import interface, load
from .diagnostics import profiles
from .evaluation import evaluate


@torch.no_grad()
def actions(model, observations):
    return torch.cat([
        model.distribution_parameters(batch)[0].tanh().cpu()
        for batch in observations.split(2048)
    ])


def run(args):
    if min(args.milestones) < 1 or sorted(set(args.milestones)) != args.milestones:
        raise ValueError('Milestones must be positive and strictly increasing')
    if min(args.episodes, args.screen_episodes, args.num_envs, args.validation_episodes) < 1:
        raise ValueError('Episode and environment counts must be positive')
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
    data, collection = collect(model, tasks['mixed'], args.episodes, 710042, stride=5)
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
            name: dict(success_rate=r['success_rate'], rmse=r['mean_tail_position_rmse'])
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
        best = min(stable, key=lambda r: (r['tasks']['cruise_1']['rmse'], r['update']))
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
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
