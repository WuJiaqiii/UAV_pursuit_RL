"""Probe control feasibility without PPO, teacher training or BC.

Compare a normalized zero-action (hover setpoint) baseline against the
repository's geometric observed-PD reference on *identical* seeded starts.
Any geometry that the zero policy can complete should not be cited as a
learned pursuit result. The reference is evaluation-only; it is never
queried by the PPO actor or used for supervised loss.
"""
import argparse
import json
from pathlib import Path
import time
import torch

from pursuit_rl.ctbr.config import load as load_config
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.checkpoint import write_json
from scripts.run_pure_ppo import TASKS


class ZeroCTBRPolicy:
    predictor = None
    position_error_cap_m = 0.0
    critic_policy_match = False

    def __init__(self, device):
        self.log_std = torch.full((4,), -2.0, device=device)

    def values_tensor(self, priv):
        return torch.zeros(len(priv), device=priv.device)

    def sample_tensor(self, obs, priv, deterministic=True):
        action = obs.new_zeros((len(obs), 4))
        return (
            action, action, obs.new_zeros(len(obs)),
            obs.new_zeros(len(obs)),
        )


def diagnose(task, device="cuda", episodes=16, seed=960001, repaired=False):
    c = load_config(TASKS[task], device=device, seed=42)
    c["num_envs"] = 32
    c["prediction"]["enabled"] = False
    c["safety"]["predictive_enabled"] = False
    c["safety"]["constraint_enabled"] = False
    if repaired:
        c["policy"]["relative_position_mode"] = "asinh"
        c["policy"]["relative_position_scale_m"] = 8.0
        c["policy"]["relative_velocity_scale_mps"] = 15.0
        c["reward"]["progress_mode"] = "pursuer_closure_normalized"
    c["training"]["teacher_method"] = "pd_observed"
    zero = ZeroCTBRPolicy(torch.device(device))
    beginning = time.perf_counter()
    zero_report, _ = evaluate(zero, c, episodes, seed, stochastic=False)
    pd_report, _ = evaluate(None, c, episodes, seed, reference=True)
    return {
        "task": task,
        "seed": seed,
        "episodes": episodes,
        "pursuit_fixes": bool(repaired),
        "wall_s": time.perf_counter() - beginning,
        "zero": {
            key: zero_report[key]
            for key in (
                "successes", "success_rate", "mean_minimum_distance_m",
                "mean_fractional_closure", "mean_maximum_tilt_deg",
                "failures",
            )
        },
        "pd_observed": {
            key: pd_report[key]
            for key in (
                "successes", "success_rate", "mean_minimum_distance_m",
                "mean_fractional_closure", "mean_maximum_tilt_deg",
                "failures",
            )
        },
        "interpretation": (
            "PD is an evaluation-only controllability/reference baseline, "
            "NOT a teacher used for PPO training. Zero-action successes "
            "can occur passively (e.g. head-on target approaches)."
        ),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=tuple(TASKS), default="near_static")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=960001)
    ap.add_argument("--pursuit-fixes", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        ap.error("Episodes must be positive and seed non-negative")
    result = diagnose(
        args.task, args.device, args.episodes, args.seed, args.pursuit_fixes
    )
    if args.out:
        path = Path(args.out)
        if path.exists():
            ap.error("Refusing to overwrite existing audit output")
        write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
