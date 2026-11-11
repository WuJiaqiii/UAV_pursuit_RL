"""Paired frozen-policy full episodes, true MC returns and optional traces."""

import collections
import hashlib
import json
import math
from pathlib import Path
import torch
from .environment import CTBRBatch
from .control import teacher
from pursuit_rl.learning.prediction import PredictionMetrics
from .config import decision_dt


@torch.no_grad()
def evaluate(
    model,
    c,
    episodes,
    seed,
    stochastic=False,
    collect_mc=False,
    reference=False,
    trace_dir=None,
    trace_count=2,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    records = []
    states = []
    targets = []
    signatures = []
    forecast_reports = []
    cuda = str(c["device"]).startswith("cuda")
    devices = list(range(torch.cuda.device_count())) if cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, episodes, c["num_envs"]):
            count = min(c["num_envs"], episodes - start)
            env = CTBRBatch(c, n=count, seed=seed + start)
            obs, priv = env.observe()
            signatures.extend(priv.cpu().tolist())
            initial = (
                model.values_tensor(priv)
                if model is not None
                else torch.zeros(count, device=env.device)
            )
            total = torch.zeros(count, device=env.device, dtype=torch.float64)
            discount = 1.0
            chunks = []
            samples = []
            block_reward = torch.zeros(count, device=env.device)
            length = 0
            trace = []
            forecast = (
                PredictionMetrics(model, env.decision_dt)
                if model is not None and model.predictor is not None
                else None
            )
            stride = max(1, round(0.02 / env.decision_dt))
            for tick in range(env.max_decision_steps + 1):
                active = ~env.finished
                if forecast is not None:
                    forecast.begin(env.target_history, active)
                if collect_mc and length == 0:
                    samples.append((priv[active].clone(), active.clone()))
                if reference:
                    action = teacher(env)
                    raw = torch.atanh(action)
                else:
                    action, raw, _, _ = model.sample_tensor(
                        obs, priv, deterministic=not stochastic
                    )
                if trace_dir is not None and start == 0 and tick % stride == 0:
                    k = min(trace_count, count)
                    trace.append(
                        (
                            tick * env.decision_dt,
                            env.plant.pursuer.pos[:k].cpu().clone(),
                            env.plant.target.pos[:k].cpu().clone(),
                            action[:k].cpu().clone(),
                            active[:k].cpu().clone(),
                            env.plant.pursuer.rot[:k].cpu().clone(),
                            env.plant.pursuer.rate[:k].cpu().clone(),
                            env.plant.pursuer.thrust[:k].cpu().clone(),
                        )
                    )
                (obs, priv), reward, done, _, completed = env.step(action, raw)
                if forecast is not None:
                    forecast.advance(env.plant.target.pos, done)
                total += discount * reward.double()
                discount *= env.gamma
                if collect_mc:
                    block_reward += (env.gamma**length) * reward
                    length += 1
                    if length == stride or bool(env.finished.all()):
                        chunks.append((block_reward.clone(), length))
                        block_reward.zero_()
                        length = 0
                for rec in completed:
                    slot = rec["slot"]
                    rec = dict(
                        rec,
                        episode=start + slot,
                        seed=seed + start,
                        initial_value=float(initial[slot]),
                        mc_return=float(total[slot]),
                        start_rel_pos=env.initial_rel_pos[slot].cpu().tolist(),
                        start_target_velocity=env.initial_target_vel[slot]
                        .cpu()
                        .tolist(),
                    )
                    records.append(rec)
                if bool(env.finished.all()):
                    break
            if not bool(env.finished.all()):
                raise RuntimeError("Evaluation failed to terminate")
            if forecast is not None:
                forecast_reports.append(forecast.report())
            if collect_mc:
                future = torch.zeros(count, device=env.device)
                for (chunk, size), (state, mask) in zip(
                    reversed(chunks), reversed(samples)
                ):
                    future = chunk + (env.gamma**size) * future
                    states.append(state)
                    targets.append(future[mask].clone())
            if trace:
                import numpy as np

                directory = Path(trace_dir)
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    directory / f"trace_seed_{seed}.npz",
                    time=np.array([x[0] for x in trace]),
                    pursuer=np.stack([x[1].numpy() for x in trace]),
                    target=np.stack([x[2].numpy() for x in trace]),
                    action=np.stack([x[3].numpy() for x in trace]),
                    active=np.stack([x[4].numpy() for x in trace]),
                    rotation=np.stack([x[5].numpy() for x in trace]),
                    body_rate=np.stack([x[6].numpy() for x in trace]),
                    thrust=np.stack([x[7].numpy() for x in trace]),
                )
    records.sort(key=lambda r: r["episode"])
    if len(records) != episodes or len({r["episode"] for r in records}) != episodes:
        raise RuntimeError("Missing/duplicate completed episodes")
    values = torch.tensor([r["initial_value"] for r in records], dtype=torch.float64)
    returns = torch.tensor([r["mc_return"] for r in records], dtype=torch.float64)

    def mean(key):
        return sum(r[key] for r in records) / episodes

    full_tail = [r for r in records if r["tail_complete"]]
    observed_tail = [r for r in records if r["tail_steps_observed"] > 0]

    def tail_mean(key, population=full_tail):
        return sum(r[key] for r in population) / len(population) if population else None

    report = dict(
        schema="pursuit-direct-ctbr-evaluation-v4",
        task=c["task"],
        seed=seed,
        num_envs=c["num_envs"],
        physics_hz=1000,
        decision_hz=1 / decision_dt(c),
        decision_steps=c["policy"]["decision_steps"],
        actor_position_error_cap_m=(
            model.position_error_cap_m if model is not None else None
        ),
        action_std=(
            model.log_std.detach().exp().cpu().tolist() if model is not None else None
        ),
        policy=(
            "geometric_reference"
            if reference
            else "stochastic" if stochastic else "deterministic"
        ),
        success_definition="Safe until full deadline; tail within-radius fraction and relative-velocity RMSE pass configured thresholds",
        episodes=episodes,
        successes=sum(r["success"] for r in records),
        success_rate=sum(r["success"] for r in records) / episodes,
        failures=dict(
            collections.Counter(r["reason"] for r in records if not r["success"])
        ),
        mean_position_rmse=mean("position_rmse"),
        mean_tail_position_rmse=tail_mean("tail_position_rmse"),
        precision_rmse_limit_m=c["task"]["maximum_tail_rmse"],
        precision_passed=(
            tail_mean("tail_position_rmse") is not None
            and tail_mean("tail_position_rmse") <= c["task"]["maximum_tail_rmse"]
        ),
        mean_tail_velocity_rmse=tail_mean("tail_velocity_rmse"),
        mean_tail_within_fraction=tail_mean("tail_within_fraction"),
        full_tail_episodes=len(full_tail),
        observed_tail_episodes=len(observed_tail),
        mean_observed_tail_position_rmse=tail_mean("tail_position_rmse", observed_tail),
        tail_metric_population="Only episodes with the complete configured tail window; absent data is null",
        mean_mc_return=float(returns.mean()),
        critic_bias=float((values - returns).mean()),
        critic_error_standard_error=float(
            (values - returns).std(unbiased=False) / math.sqrt(episodes)
        ),
        critic_mc_rmse=float((values - returns).square().mean().sqrt()),
        critic_matches_policy=stochastic
        and not reference
        and getattr(model, "critic_policy_match", True),
        reward_scale=c["reward"]["scale"],
        initial_state_digest=hashlib.sha256(
            json.dumps(signatures, sort_keys=True).encode()
        ).hexdigest(),
        episodes_detail=records,
    )
    report["joint_precision_success_rate"] = sum(
        r["success"] and r["tail_complete"]
        and r["tail_position_rmse"] <= c["task"]["maximum_tail_rmse"]
        for r in records
    ) / episodes
    report["mean_tail_axis_rmse_m"] = (
        [sum(r["tail_axis_rmse_m"][i] for r in full_tail) / len(full_tail)
         for i in range(3)] if full_tail else None
    )
    report["tail_position_rmse_p95_m"] = (
        float(torch.quantile(torch.tensor([r["tail_position_rmse"] for r in full_tail]), 0.95))
        if full_tail else None
    )
    report["velocity_projection_definition"] = (
        "Mean dot(target-pursuer, target_velocity)/speed^2 over tail physics ticks "
        "with target speed >=0.2 m/s; positive means behind along velocity; "
        "a geometric proxy, not identified latency. Per-episode values include partial tails."
    )
    if forecast_reports:
        horizons = c["prediction"]["horizons_s"]
        counts = [
            sum(r["samples_by_horizon"][k] for r in forecast_reports)
            for k in range(len(horizons))
        ]
        errors = {}
        for name in ("learned", "constant_velocity", "constant_acceleration"):
            errors[name] = [
                (
                    math.sqrt(
                        sum(
                            (r["rmse_by_horizon_m"][name][k] or 0) ** 2
                            * r["samples_by_horizon"][k]
                            for r in forecast_reports
                        )
                        / counts[k]
                    )
                    if counts[k]
                    else None
                )
                for k in range(len(horizons))
            ]
        report["forecast"] = dict(
            mode=c["prediction"]["mode"],
            horizons_s=horizons,
            samples_by_horizon=counts,
            rmse_by_horizon_m=errors,
        )
    data = (torch.cat(states), torch.cat(targets)) if collect_mc else None
    return report, data


def score(report):
    records = report["episodes_detail"]
    completed = sum(
        r["reason"] in ("success", "tracking_failed") for r in records
    ) / len(records)
    error = sum(
        (
            r["tail_position_rmse"]
            if r["reason"] in ("success", "tracking_failed")
            else max(r["position_rmse"], r["tail_position_rmse"])
            + report["task"]["radius"]
        )
        for r in records
    ) / len(records)
    return (report["success_rate"], completed, -error)


def improved(report, best):
    candidate = score(report)
    if candidate[:2] != tuple(best[:2]):
        return candidate[:2] > tuple(best[:2])
    return candidate[2] - best[2] > max(0.01, 0.01 * abs(best[2]))
