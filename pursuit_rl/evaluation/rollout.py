"""Independent deterministic seeded evaluation and optional full trajectory capture."""

import numpy as np
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.artifacts.trace import TraceRecorder
from pursuit_rl.control.guidance import Controller
from pursuit_rl.simulation.targets import MODES
from pursuit_rl.config import config_hash
from pursuit_rl.evaluation.metrics.statistics import mission_summary
from pursuit_rl.learning.contracts import policy_contract

MODE_NAMES = {value: key for key, value in MODES.items()}
EVALUATION_VERSION = "pursuit-evaluation-v6-batch64"


def evaluate(
    model: Controller, cfg, stage=None, episodes=32, seed=25000, trace_dir=None,
    disturbance=None,
):
    if cfg.get('simulation_backend','numpy') == 'torch':
        from pursuit_rl.evaluation.torch_rollout import evaluate_torch
        return evaluate_torch(model,cfg,stage,episodes,seed,trace_dir,disturbance)
    from pursuit_rl.simulation.disturbances import (
        SensorChannel, CommandChannel, validate_disturbance,
    )
    disturbance = validate_disturbance(disturbance, cfg["physics"]["dt"])
    stage = cfg["training"]["max_stage"] if stage is None else int(stage)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    # Use a separate evaluation environment to avoid affecting training RNG.
    n = min(episodes, 64)
    results = []
    # Sequential batches when episodes > n.
    for start in range(0, episodes, n):
        active = min(n, episodes - start)
        env = PursuitBatch(
            cfg, n=active, seed=seed + start, stage=stage, randomize=True
        )
        obs, priv = env.observe()
        sensor = SensorChannel(cfg, disturbance, seed + start)
        command_channel = CommandChannel(cfg, disturbance, active)
        if any(disturbance.values()):
            obs = sensor.observe(env, obs)
        env.commanded_action = np.zeros((active, 4), dtype=np.float32)
        finished = np.zeros(active, bool)
        recorder = TraceRecorder(env, trace_dir) if trace_dir is not None else None
        horizon = int(np.ceil(env.stage_config["time"] / cfg["physics"]["dt"])) + 2
        interval = int(cfg["physics"].get("policy_interval_steps", 1))
        held_action = None
        for tick in range(horizon):
            if held_action is None or tick % interval == 0:
                held_action = model.act(env, obs)
            actions = held_action
            env.commanded_action = np.asarray(actions, dtype=np.float32).copy()
            executed = command_channel.apply(actions)
            (nobs, npriv), _, term, trunc, info = env.step(executed)
            if any(disturbance.values()):
                nobs = sensor.observe(env, nobs)
            done = term | trunc
            if recorder is not None:
                recorder.record(env, finished, done)
            for i in np.where(done & ~finished)[0]:
                rec = {
                    "episode": int(start + i),
                    "seed": int(seed + start),
                    "target_mode_name": MODE_NAMES[info[i]["target_mode"]],
                    **info[i],
                }
                rec["capped_time"] = float(
                    rec["reach_time"] if rec["success"] else env.stage_config["time"]
                )
                results.append(rec)
                if recorder is not None:
                    recorder.save(i, rec, cfg, stage, seed + start)
            finished |= done
            if finished.all():
                break
            obs = nobs
            priv = npriv
    if len(results) != episodes:
        raise RuntimeError(
            f"Evaluation ended with {len(results)} of {episodes} episodes"
        )
    report = summarize(sorted(results, key=lambda item: item["episode"]), stage)
    report["evaluation_protocol"] = {
        "version": EVALUATION_VERSION,
        "seed": seed,
        "batch_size": n,
        "episodes": episodes,
        "stage": stage,
        "task_hash": task_hash(cfg, stage),
        "pursuer_max_speed": cfg["physics"]["max_speed"],
        "policy_contract": policy_contract(cfg),
        "disturbance": disturbance,
        "physics_dt_s": cfg["physics"]["dt"],
        "policy_dt_s": cfg["physics"]["dt"] * cfg["physics"].get("policy_interval_steps", 1),
        "policy_frequency_hz": 1.0 / (cfg["physics"]["dt"] * cfg["physics"].get("policy_interval_steps", 1)),
        "trace_stride": cfg.get("evaluation", {}).get("trace_stride", 1),
    }
    return report


def summarize(results, stage=None):
    stats = mission_summary(
        [
            {
                "success": row["success"],
                "reason": row["reason"],
                "capture_time": row["reach_time"],
                "capped_time": row["capped_time"],
            }
            for row in results
        ]
    )
    bymode = {}
    for name in sorted(set(v["target_mode_name"] for v in results)):
        subset = [v for v in results if v["target_mode_name"] == name]
        bymode[name] = {
            "episodes": len(subset),
            "success_rate": sum(v["success"] for v in subset) / len(subset),
            "mean_capped_time": float(np.mean([v["capped_time"] for v in subset])),
        }
    return {
        "stage": stage,
        "episodes": len(results),
        "success_rate": stats["success_rate"],
        "mean_capped_time": stats["mean_capped_time"],
        "mean_reach_time_success": stats["mean_capture_time_success"],
        "modes": bymode,
        "failures": {
            k: stats["failures_" + k] for k in ("speed", "outside", "tilt", "timeout")
        },
        "episodes_detail": results,
    }


def task_hash(cfg, stage):
    return config_hash(
        {key: cfg[key] for key in ("physics", "target")}
        | {"stage": cfg["curriculum"][stage], "simulation_backend":cfg.get('simulation_backend','numpy')}
    )
