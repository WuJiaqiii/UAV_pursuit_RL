#!/usr/bin/env python3
"""Run frozen PPO and an explicit reference in paired challenge scenarios."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pursuit_rl.config.challenge import SCENARIOS
from pursuit_rl.evaluation.challenge import run_challenge


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=[
            "baseline",
            "singer_a05",
            "singer_a1",
            "singer_a2",
            "dubins_a2",
            "hybrid_a2",
        ],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1000000, 1010000, 1020000])
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-reference", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=1)
    args = p.parse_args(argv)
    if (
        args.episodes < 1
        or len(set(args.scenarios)) != len(args.scenarios)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        p.error("Positive episode count and unique scenarios/seeds required")
    torch.set_num_threads(args.cpu_threads)
    run_challenge(
        args.checkpoint,
        args.scenarios,
        args.episodes,
        args.seeds,
        args.out,
        args.device,
        num_envs=args.num_envs,
        stochastic=args.stochastic,
        include_reference=not args.no_reference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
