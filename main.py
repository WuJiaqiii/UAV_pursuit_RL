"""Train, evaluate and compare high-speed first-entry CTBR policies."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from pursuit_rl.ctbr.config import load
from pursuit_rl.ctbr.checkpoint import load as checkpoint_load, write_json
from pursuit_rl.ctbr.training import train
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.curriculum import run
from pursuit_rl.ctbr.errors import ExperimentGateError


DEFAULT_ENTRY_CONFIG = "configs/first_entry_highspeed/01_static.json"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="8.0.0")
    commands = p.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--config", default=DEFAULT_ENTRY_CONFIG)
    t.add_argument("--out", required=True)
    init = t.add_mutually_exclusive_group()
    init.add_argument("--resume")
    init.add_argument("--init-from")
    t.add_argument("--physics-steps", type=int)
    e = commands.add_parser("evaluate")
    e.add_argument("--checkpoint", help="Required unless --reference is used")
    e.add_argument("--out", required=True)
    e.add_argument("--config", help="First-entry task profile; must match checkpoint interface")
    e.add_argument("--episodes", type=int, default=64)
    e.add_argument("--stochastic", action="store_true")
    e.add_argument(
        "--reference",
        action="store_true",
        help="Geometric controller only; never mixes actions",
    )
    e.add_argument(
        "--reference-method", choices=("pd", "pn", "pn_hybrid"),
        help="Select teacher baseline when evaluating with --reference",
    )
    e.add_argument("--trace-dir")
    e.add_argument("--trace-count", type=int, default=2)
    e.add_argument("--num-envs", type=int)
    q = commands.add_parser("curriculum")
    q.add_argument("--plan", default="configs/first_entry_highspeed/plan.json")
    q.add_argument("--out", required=True)
    q.add_argument("--physics-steps", type=int)
    q.add_argument("--episodes", type=int, default=32)
    q.add_argument("--minimum-success", type=float, default=0.95)
    q.add_argument("--phases", nargs="+")
    q.add_argument("--init-from", help="Initialize from a checkpoint with the current first-entry interface")
    for sub in (t, e, q):
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--cpu-threads", type=int, default=1)
    return p


def execute(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.command == "train":
        c = load(args.config, args.device, args.seed)
        if args.physics_steps is not None:
            if args.physics_steps < 0:
                raise ValueError("Budget must be nonnegative")
            c["training"]["physics_steps"] = args.physics_steps
        return train(c, args.out, args.resume, args.init_from)
    if args.command == "curriculum":
        return run(
            args.plan,
            args.out,
            args.device,
            args.seed,
            args.physics_steps,
            args.episodes,
            args.minimum_success,
            args.phases,
            init_from=args.init_from,
        )
    if args.reference_method and not args.reference:
        raise ValueError("--reference-method requires --reference")
    if not args.checkpoint:
        if not args.reference:
            raise ValueError("Evaluation requires --checkpoint or --reference")
        model, saved = None, None
        c = load(args.config or DEFAULT_ENTRY_CONFIG, args.device, args.seed)
    else:
        model, c, saved = checkpoint_load(args.checkpoint, args.device)
    if args.checkpoint and args.config:
        from pursuit_rl.ctbr.config import interface

        requested = load(args.config, args.device, c["seed"])
        if interface(requested) != interface(c):
            raise ValueError(
                "Evaluation changes the control/observation/reward interface"
            )
        c = requested
    if args.num_envs is not None:
        if args.num_envs < 1:
            raise ValueError("num-envs must be positive")
        c["num_envs"] = args.num_envs
    if args.trace_count < 1:
        raise ValueError("--trace-count must be positive")
    if args.reference_method:
        c["training"]["teacher_method"] = args.reference_method
    report, _ = evaluate(
        model,
        c,
        args.episodes,
        args.seed,
        args.stochastic,
        reference=args.reference,
        trace_dir=args.trace_dir,
        trace_count=args.trace_count,
    )
    if saved is not None:
        report.update(
            checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            checkpoint_origin=saved["origin"],
            progress=saved["progress"],
        )
    write_json(args.out, report)
    print(
        f"{report['successes']}/{report['episodes']} first safe entries "
        f"(radius={report['entry_radius_m']:g}m, relative-speed limit="
        f"{report['entry_relative_speed_limit_mps']:g}m/s)"
    )
    return report


def main(argv=None):
    try:
        return execute(argv)
    except ExperimentGateError as exc:
        print(
            json.dumps(exc.details, ensure_ascii=False, allow_nan=False),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
