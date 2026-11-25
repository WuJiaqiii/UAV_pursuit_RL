"""Audit whether zero CTBR, observed-state PD and trained PPO can close distance.

This is EVALUATION ONLY. PD is never used as a training teacher or blended
with PPO. Matching task/seed/episode counts expose passive head-on successes,
effective closed-loop controllability and failures of policy exploration.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint
from pursuit_rl.ctbr.config import load as load_config
from pursuit_rl.ctbr.evaluation import evaluate
from scripts.run_pure_ppo import TASKS
from scripts.run_pure_ppo_campaign import _atomic_json


class HoverPolicy:
    """Deterministic zero CTBR action: neutral thrust and zero body rates."""
    predictor = None

    def values_tensor(self, priv):
        return torch.zeros(len(priv), device=priv.device)

    def sample_tensor(self, obs, priv, deterministic=True):
        action = torch.zeros(len(obs), 4, device=obs.device)
        logp = torch.zeros(len(obs), device=obs.device)
        return action, action, logp, self.values_tensor(priv)


def summarize(report):
    rows = report["episodes_detail"]
    distances = [
        math.dist(r["start_rel_pos"], (0, 0, 0))
        - r["minimum_distance"]
        for r in rows
    ]
    n = len(rows)
    return dict(
        episodes=n,
        successes=report["successes"],
        success_rate=report["success_rate"],
        failures=report["failures"],
        mean_minimum_distance_m=report["mean_minimum_distance_m"],
        mean_initial_to_closest_progress_m=sum(distances) / n,
        active_closing_fraction=sum(d > 0.15 for d in distances) / n,
        mean_maximum_tilt_deg=report.get("mean_maximum_tilt_deg"),
        mean_entry_time_s=report.get("mean_entry_time_s"),
    )


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS),
                        default=["near_static", "static", "tail", "head_on", "random"])
    parser.add_argument("--checkpoint",
                        help="Optional PPO checkpoint to compare with zero and PD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=960001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("Expected at least one episode")
    result = dict(schema="v13-pursuit-control-audit-v1",
                  tasks=args.tasks, episode_seed=args.seed,
                  episodes_per_task=args.episodes, methods={})
    checkpoint_model = None
    checkpoint_config = None
    if args.checkpoint:
        checkpoint_model, checkpoint_config, info = load_checkpoint(
            args.checkpoint, device=args.device
        )
        if checkpoint_config["prediction"]["enabled"]:
            parser.error("Use a pure PPO policy checkpoint for this baseline")
        result["checkpoint"] = str(Path(args.checkpoint).resolve())
        result["checkpoint_origin"] = info["origin"]
    for task_index, task in enumerate(args.tasks):
        name = TASKS[task]
        c = copy.deepcopy(checkpoint_config) if checkpoint_config else load_config(
            name, device=args.device
        )
        c["task"] = copy.deepcopy(load_config(name)["task"])
        c["training"]["fast_terminal_records"] = True
        c["safety"]["predictive_enabled"] = False
        c["safety"]["constraint_enabled"] = False
        c["reward"]["actuator_risk"] = 0
        seed = args.seed + task_index * 10007
        procedures = {}
        empty = HoverPolicy()
        report, _ = evaluate(empty, c, args.episodes, seed, stochastic=False)
        procedures["zero_ctbr"] = summarize(report)
        pd_config = copy.deepcopy(c)
        pd_config["training"]["teacher_method"] = "pd_observed"
        report, _ = evaluate(None, pd_config, args.episodes, seed, reference=True)
        procedures["pd_observed"] = summarize(report)
        if checkpoint_model is not None:
            report, _ = evaluate(
                checkpoint_model, c, args.episodes, seed, stochastic=False
            )
            procedures["ppo_deterministic"] = summarize(report)
        result["methods"][task] = procedures
        print(json.dumps({task: procedures}, ensure_ascii=False), flush=True)
    _atomic_json(args.out, result)
    return result


if __name__ == "__main__":
    run()
