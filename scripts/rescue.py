#!/usr/bin/env python3
"""Standalone entry point: legacy main.py checkpoints remain a separate interface."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pursuit_rl.config.rescue import load_rescue_profile, rescue_contract
from pursuit_rl.learning.rescue_checkpoint import load_rescue_checkpoint
from pursuit_rl.evaluation.rescue import evaluate_rescue
from pursuit_rl.training.rescue import train_rescue, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "reference"))
    parser.add_argument("--profile")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--physics-steps", type=int, default=None)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume")
    initialization.add_argument(
        "--init-from", help="Transfer actor to a new task; reset critic and optimizers"
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--capture-radius",
        type=float,
        default=None,
        help="Explicit transfer evaluation at another capture radius",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.command != "train" and (args.resume or args.init_from):
        parser.error("Initialization flags apply only to training")
    if args.command == "evaluate":
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        expected = None
        if args.profile:
            expected_cfg, expected_profile = load_rescue_profile(
                args.profile, device=args.device
            )
            expected = rescue_contract(expected_cfg, expected_profile)
        model, cfg, profile, saved = load_rescue_checkpoint(
            args.checkpoint, device=args.device, expected_contract=expected
        )
        if args.capture_radius is not None:
            if args.capture_radius <= 0:
                parser.error("--capture-radius must be positive")
            cfg["curriculum"][profile["stage"]]["success_radius"] = args.capture_radius
        report, _ = evaluate_rescue(
            model,
            cfg,
            profile,
            args.episodes or profile["test_episodes"],
            profile["test_seed"] if args.seed is None else args.seed,
            stochastic=args.stochastic,
        )
        report.update(
            checkpoint=str(Path(args.checkpoint).resolve()),
            checkpoint_origin=saved["origin"],
            ppo_updates=saved["progress"]["ppo_updates"],
            physics_steps=saved["progress"]["physics_steps"],
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "episodes_detail"},
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )
        return 0
    if args.capture_radius is not None:
        parser.error("Use a profile for training/reference capture radius")
    if not args.profile:
        parser.error("train/reference requires --profile")
    cfg, profile = load_rescue_profile(args.profile, device=args.device)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.physics_steps is not None:
        if args.physics_steps < 0:
            parser.error("--physics-steps cannot be negative")
        profile["budget_physics_steps"] = args.physics_steps
    if args.command == "reference":
        report, _ = evaluate_rescue(
            None,
            cfg,
            profile,
            args.episodes or profile["test_episodes"],
            profile["test_seed"] if args.seed is None else args.seed,
            reference=True,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "episodes_detail"},
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )
        return 0
    print(
        json.dumps(
            train_rescue(
                cfg, profile, args.out, resume=args.resume, init_from=args.init_from
            ),
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
