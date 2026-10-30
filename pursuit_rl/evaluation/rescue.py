"""Frozen-policy rescue evaluation and complete-episode Monte Carlo targets."""

import collections
import math
import torch
from pursuit_rl.simulation.rescue_environment import RescueBatch
from pursuit_rl.simulation.rescue_control import reference_action


@torch.no_grad()
def evaluate_rescue(
    model,
    cfg,
    profile,
    episodes,
    seed,
    reference=False,
    stochastic=False,
    collect_mc=False,
    environment_factory=None,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    device = cfg["device"]
    all_records = []
    all_prediction = []
    all_returns = []
    data_priv = []
    data_targets = []
    cuda = bool(str(device).startswith("cuda"))
    devices = list(range(torch.cuda.device_count())) if cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, episodes, profile["num_envs"]):
            count = min(profile["num_envs"], episodes - start)
            factory = (
                RescueBatch if environment_factory is None else environment_factory
            )
            env = factory(cfg, profile, n=count, seed=seed + start, device=device)
            obs, priv = env.observe()
            prediction = (
                model.values_tensor(priv)
                if model is not None
                else torch.zeros(count, device=env.device)
            )
            reward_history = []
            done_history = []
            samples = []
            total = torch.zeros(count, device=env.device, dtype=torch.float64)
            discount = 1.0
            stride = max(1, math.ceil(0.02 / env.decision_dt))
            for tick in range(
                math.ceil(env.env.stage_config["time"] / env.decision_dt) + 2
            ):
                active = ~env.finished
                if collect_mc and tick % stride == 0:
                    samples.append((tick, priv[active].clone(), active.clone()))
                if reference:
                    action = reference_action(env.env, profile)
                    action = torch.where(active[:, None], action, 0.0)
                    raw = torch.atanh(action.clamp(-0.999999, 0.999999))
                else:
                    action, raw, _, _ = model.sample_tensor(
                        obs, priv, deterministic=not stochastic
                    )
                (obs, priv), reward, term, _, records = env.step(action, raw)
                total += discount * torch.where(active, reward, 0.0)
                discount *= env.gamma
                if collect_mc:
                    reward_history.append(reward.clone())
                    done_history.append(term.clone())
                for record in records:
                    record = dict(
                        record, episode=start + record["slot"], seed=seed + start
                    )
                    record["start_rel_pos"] = (
                        env.env.initial_rel_pos[record["slot"]].cpu().tolist()
                    )
                    record["start_target_vel"] = (
                        env.env.initial_target_vel[record["slot"]].cpu().tolist()
                    )
                    all_records.append(record)
                if bool(env.finished.all()):
                    break
            if not bool(env.finished.all()):
                raise RuntimeError("Rescue evaluation did not finish")
            all_prediction.append(prediction.cpu())
            all_returns.append(total.cpu())
            if collect_mc:
                lookup = {k: (state, mask) for k, state, mask in samples}
                future = torch.zeros(count, device=env.device)
                targets = []
                states = []
                for k in range(len(reward_history) - 1, -1, -1):
                    future = reward_history[k] + env.gamma * torch.where(
                        done_history[k], 0.0, future
                    )
                    if k in lookup:
                        state, mask = lookup[k]
                        states.append(state)
                        targets.append(future[mask].clone())
                data_priv.extend(states)
                data_targets.extend(targets)
    records = sorted(all_records, key=lambda r: r["episode"])
    if len(records) != episodes or len({r["episode"] for r in records}) != episodes:
        raise RuntimeError("Duplicate/missing rescue episode")
    success = sum(r["success"] for r in records)
    predicted = torch.cat(all_prediction).double()
    actual = torch.cat(all_returns).double()
    error = predicted - actual
    successful = [r["time"] for r in records if r["success"]]
    report = {
        "schema": "pursuit-rescue-evaluation-v1",
        "action_mode": profile["action_mode"],
        "decision_hz": 1 / (cfg["physics"]["dt"] * profile["decision_interval_steps"]),
        "physics_hz": 1 / cfg["physics"]["dt"],
        "stage": profile["stage"],
        "capture_radius_m": cfg["curriculum"][profile["stage"]]["success_radius"],
        "success_definition": "first entrance into capture sphere before failure/deadline",
        "controller": "reference" if reference else "learned_actor",
        "policy": "stochastic" if stochastic else "deterministic",
        "seed": seed,
        "episodes": episodes,
        "successes": success,
        "success_rate": success / episodes,
        "failures": dict(
            collections.Counter(r["reason"] for r in records if not r["success"])
        ),
        "mean_capture_time_success": (
            sum(successful) / len(successful) if successful else None
        ),
        "value_units": "learning_reward = task_reward * reward_scale",
        "reward_scale": profile["reward_scale"],
        "initial_value_mean": float(predicted.mean()),
        "mc_return_mean": float(actual.mean()),
        "critic_rmse": float(error.square().mean().sqrt()),
        "critic_bias": float(error.mean()),
        "episodes_detail": records,
    }
    data = (torch.cat(data_priv), torch.cat(data_targets)) if collect_mc else None
    return report, data
