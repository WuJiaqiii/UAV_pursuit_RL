"""Same-state controller probes and full-episode deterministic DAgger experiments."""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import torch
from .checkpoint import load as load_model, save, write_json, source_fingerprint
from .config import load, interface
from .control import teacher
from .environment import CTBRBatch
from .evaluation import evaluate
from .diagnostics import profiles


@torch.no_grad()
def policy_action(model, env):
    obs, _ = env.observe()
    raw = model.distribution_parameters(obs)[0]
    return raw.tanh(), raw


@torch.no_grad()
def scan(model, env, offsets, horizon=0.3):
    """Perturb pursuer position only, preserving causal target/action history."""
    speed = torch.linalg.vector_norm(env.plant.target.vel.float(), dim=1)
    direction = env.plant.target.vel.float() / speed[:, None].clamp_min(0.2)
    valid = (~env.finished) & (speed >= 0.2)
    result = []
    for offset in offsets:
        trial = copy.deepcopy(env)
        trial.plant.pursuer.pos -= float(offset) * direction
        obs, _ = trial.observe()
        action, _ = policy_action(model, trial)
        target = teacher(trial)
        future = {}
        for name in ("policy", "teacher"):
            branch = copy.deepcopy(trial)
            initial_velocity = branch.plant.pursuer.vel.clone()
            for _ in range(math.ceil(horizon / branch.decision_dt)):
                a, raw = policy_action(model, branch) if name == "policy" else (teacher(branch), None)
                branch.step(a, raw)
            delta_velocity = branch.plant.pursuer.vel - initial_velocity
            future[name] = dict(
                along_velocity_change=(delta_velocity * direction).sum(1),
                final_distance=torch.linalg.vector_norm(branch.plant.target.pos - branch.plant.pursuer.pos, dim=1),
                finished=branch.finished)
        for slot in valid.nonzero(as_tuple=True)[0].tolist():
            result.append(dict(slot=slot, added_along_error_m=float(offset),
                position_error_m=(trial.plant.target.pos[slot]-trial.plant.pursuer.pos[slot]).tolist(),
                target_velocity=trial.plant.target.vel[slot].tolist(),
                actor_position_feature=model.bound_actor_input(model.actor_features(obs))[slot, :3].tolist(),
                raw_memory=obs[slot, 32:36].tolist(),
                policy_action=action[slot].tolist(), teacher_action=target[slot].tolist(),
                actual_thrust=float(trial.plant.pursuer.thrust[slot]),
                actual_rate=trial.plant.pursuer.rate[slot].tolist(),
                branches={name: {key: bool(value[slot]) if key == 'finished' else float(value[slot])
                                 for key, value in data.items()} for name, data in future.items()}))
    return result


@torch.no_grad()
def collect(model, config, episodes, seed, stride=5, *, teacher_rollout=False, max_label_distance=None):
    """Collect labeled states from complete policy or teacher rollouts."""
    states, labels, records = [], [], []
    interactions = 0
    for start in range(0, episodes, config['num_envs']):
        env = CTBRBatch(config, n=min(config['num_envs'], episodes-start), seed=seed+start)
        for tick in range(env.max_decision_steps):
            active = ~env.finished
            obs, _ = env.observe()
            eligible = active
            if max_label_distance is not None:
                distance = torch.linalg.vector_norm(env.plant.target.pos - env.plant.pursuer.pos, dim=1)
                eligible = active & (distance <= max_label_distance)
            if tick % stride == 0 and bool(eligible.any()):
                states.append(obs[eligible].clone())
                labels.append(teacher(env)[eligible].clone())
            action, raw = (teacher(env), None) if teacher_rollout else policy_action(model, env)
            _, _, _, _, completed = env.step(action, raw)
            records.extend(completed)
            interactions += env.last_physics_steps
            if bool(env.finished.all()):
                break
    if len(records) != episodes:
        raise RuntimeError('Incomplete collection')
    if not states:
        raise ValueError('No rollout states satisfied the sampling criteria')
    return (torch.cat(states), torch.cat(labels)), dict(
        episodes=episodes, samples=sum(len(x) for x in states), physics_steps=interactions,
        successes=sum(r['success'] for r in records))


def fit(model, data, updates=500, batch_size=2048, lr=3e-5, *, on_update=None, anchor_weight=0):
    """Fit executed conditional mean, with frozen predictor and exploration std."""
    obs, labels = data
    with torch.no_grad():
        features = torch.cat([model.actor_features(x) for x in obs.split(batch_size)])
    reference = None
    if anchor_weight:
        with torch.no_grad():
            reference = torch.cat([model.distribution_parameters(x, prepared=True)[0].tanh()
                                   for x in features.split(batch_size)])
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    loss_value = None
    for update in range(1, updates + 1):
        ids = torch.randint(len(obs), (min(batch_size, len(obs)),), device=model.device)
        mean, _ = model.distribution_parameters(features[ids], prepared=True)
        actions = mean.tanh()
        imitation_loss = (actions - labels[ids]).square().mean()
        anchor_loss = (actions - reference[ids]).square().mean() if reference is not None else actions.new_zeros(())
        loss = imitation_loss + anchor_weight * anchor_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite full-observation imitation loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5, error_if_nonfinite=True)
        opt.step()
        loss_value = float(loss.detach())
        if on_update is not None:
            on_update(update, dict(loss=loss_value, imitation_loss=float(imitation_loss.detach()),
                                   anchor_loss=float(anchor_loss.detach()), grad_norm=float(grad_norm)))
    return dict(updates=updates, loss=loss_value, samples=len(obs))


def acceptable(before, after, relative_slack=0.1, absolute_slack=0.02):
    """Protect every task/seed; require a mean current-task precision improvement."""
    for key, old in before.items():
        new = after[key]
        if new['success_rate'] < old['success_rate']:
            return False
        if new['joint_precision_success_rate'] < old['joint_precision_success_rate']:
            return False
        a, b = old['mean_tail_position_rmse'], new['mean_tail_position_rmse']
        if a is None or b is None or b > a * (1+relative_slack) + absolute_slack:
            return False
    current = [k for k in before if k.startswith('mixed/')]
    return sum(after[k]['mean_tail_position_rmse'] for k in current) < sum(
        before[k]['mean_tail_position_rmse'] for k in current) - 0.01 * len(current)


def reports(model, tasks, episodes, seeds):
    return {f'{task}/{seed}': evaluate(model, c, episodes, seed)[0]
            for task, c in tasks.items() for seed in seeds}


def run(args):
    if args.validation_episodes < 1 or not args.validation_seeds or args.warmup < 0 or args.episodes < 1 or args.num_envs < 1 or args.rounds < 1 or args.updates < 1 or args.stride < 1:
        raise ValueError('Counts must be positive')
    if args.horizon <= 0 or args.lr <= 0 or len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError('Positive horizon/lr and distinct validation seeds required')
    if any(s in args.validation_seeds for s in [args.train_seed + 100003*i for i in range(args.rounds)]):
        raise ValueError('Training and validation seed bases must differ')
    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    model, original, saved = load_model(args.checkpoint, args.device, optimizers=True)
    c = load(args.config, device=args.device)
    if interface(c) != interface(original):
        raise ValueError('Incompatible model/config interface')
    c['num_envs'] = args.num_envs
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(arguments=vars(args), config=c, source_fingerprint=source_fingerprint(),
                    source_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    write_json(root/'manifest.json', manifest)
    torch.manual_seed(args.train_seed)
    if args.command == 'probe':
        task = profiles(c, [args.speed])[f'cruise_{args.speed:g}']
        env = CTBRBatch(task, seed=args.train_seed)
        warmup = min(round(args.warmup / env.decision_dt), env.max_decision_steps-1)
        for _ in range(warmup):
            env.step(*policy_action(model, env))
        rows = scan(model, env, args.offsets, args.horizon)
        write_json(root/'probe.json', dict(rows=rows, config=task,
            nominal_horizon_s=math.ceil(args.horizon/env.decision_dt)*env.decision_dt,
            note='Added error is target-pursuer along velocity. Branches use closed-loop actions, not held commands. Finished branches may be shorter.'))
        return
    tasks = profiles(c, [0.5, 1, 2])
    hover = load('configs/ctbr_v701/hover.json', device=args.device)
    if interface(hover) != interface(c):
        raise ValueError('Hover retention interface mismatch')
    hover['num_envs'] = args.num_envs
    tasks['hover'] = hover
    acquisition = load('configs/ctbr_v701/acquisition30.json', device=args.device)
    if interface(acquisition) != interface(c):
        raise ValueError('Acquisition retention interface mismatch')
    acquisition['num_envs'] = args.num_envs
    tasks['acquisition30'] = acquisition
    before = reports(model, tasks, args.validation_episodes, args.validation_seeds)
    write_json(root/'baseline_validation.json', before)
    initial_std = model.log_std.detach().clone()
    total_updates = 0
    aggregate = None
    for index in range(args.rounds):
        backup = copy.deepcopy(model.actor.state_dict())
        current, collection = collect(model, c, args.episodes, args.train_seed+100003*index, args.stride)
        data = current if aggregate is None else tuple(torch.cat((old, new)) for old, new in zip(aggregate, current))
        fit_report = fit(model, data, args.updates, lr=args.lr)
        candidate = reports(model, tasks, args.validation_episodes, args.validation_seeds)
        accepted = acceptable(before, candidate)
        write_json(root/f'round_{index+1}.json', dict(collection=collection, fit=fit_report,
                   accepted=accepted, baseline=before, candidate=candidate))
        if accepted:
            before = candidate
            aggregate = data
            total_updates += args.updates
        else:
            model.actor.load_state_dict(backup)
        del current, data
    assert torch.equal(initial_std, model.log_std)
    # Fresh optimizers are used by train --init-from; critic must be recalibrated.
    progress = dict(ppo_updates=0, physics_steps=0,
        cumulative_ppo_physics_steps=saved['progress'].get('cumulative_ppo_physics_steps', 0),
        critic_needs_calibration=True, behavior_updates=total_updates, parent=manifest['source_sha256'])
    model.critic_policy_match = False
    save(root/'selected.pt', model, c, progress,
         'full_episode_dagger' if total_updates else 'unchanged_behavior_baseline')
    write_json(root/'summary.json', dict(accepted_updates=total_updates,
        selected=str(root/'selected.pt'), validation=before,
        note='Validation selected candidates; reserve new final-test seeds. No PPO was performed.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['probe', 'distill'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default='configs/ctbr_v701/near_mixed.json')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--train-seed', type=int, default=710042)
    p.add_argument('--validation-seeds', type=int, nargs='+', default=[810042, 910042])
    p.add_argument('--validation-episodes', type=int, default=32)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--updates', type=int, default=500)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--speed', type=float, default=1)
    p.add_argument('--warmup', type=float, default=35)
    p.add_argument('--offsets', type=float, nargs='+', default=[-0.5, 0, 0.5])
    p.add_argument('--horizon', type=float, default=0.3)
    a = p.parse_args()
    torch.set_num_threads(1)
    run(a)


if __name__ == '__main__':
    main()
