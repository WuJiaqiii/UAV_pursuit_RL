"""Single direct-CTBR CLI. Old entry points are kept in Git history."""

import argparse
import hashlib
from pathlib import Path
import torch
from pursuit_rl.ctbr.config import load
from pursuit_rl.ctbr.checkpoint import load as checkpoint_load, write_json
from pursuit_rl.ctbr.training import train
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.curriculum import run


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="6.9.0")
    commands = p.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--out", required=True)
    init = t.add_mutually_exclusive_group()
    init.add_argument("--resume")
    init.add_argument("--init-from")
    t.add_argument("--physics-steps", type=int)
    e = commands.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--config", help="Explicit same-interface task transfer evaluation")
    e.add_argument("--episodes", type=int, default=64)
    e.add_argument("--stochastic", action="store_true")
    e.add_argument(
        "--reference",
        action="store_true",
        help="Geometric controller only; never mixes actions",
    )
    e.add_argument("--trace-dir")
    e.add_argument("--num-envs", type=int)
    q = commands.add_parser("curriculum")
    q.add_argument("--plan", default="configs/ctbr/plan.json")
    q.add_argument("--out", required=True)
    q.add_argument("--physics-steps", type=int)
    q.add_argument("--episodes", type=int, default=32)
    q.add_argument("--minimum-success", type=float, default=0.9)
    q.add_argument("--phases", nargs="+")
    for sub in (t, e, q):
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--cpu-threads", type=int, default=1)
    return p


def main(argv=None):
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
        )
    model, c, saved = checkpoint_load(args.checkpoint, args.device)
    if args.config:
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
    report, _ = evaluate(
        model,
        c,
        args.episodes,
        args.seed,
        args.stochastic,
        reference=args.reference,
        trace_dir=args.trace_dir,
    )
    report.update(
        checkpoint_sha256=hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        checkpoint_origin=saved["origin"],
        progress=saved["progress"],
    )
    write_json(args.out, report)
    print(
        f"{report['successes']}/{report['episodes']} sustained successes; tail RMSE {report['mean_tail_position_rmse']:.3f} m"
    )
    return report


if __name__ == "__main__":
    main()
