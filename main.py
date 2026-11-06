"""Single direct-CTBR CLI. Old entry points are kept in Git history."""

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
from pursuit_rl.ctbr.experiments import run as run_experiment
from pursuit_rl.ctbr.checkpoint import migrate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="7.0.1")
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
    q.add_argument("--minimum-success", type=float, default=0.95)
    q.add_argument("--phases", nargs="+")
    q.add_argument("--init-from", help="Explicit same-interface v701 initialization")
    m = commands.add_parser(
        "migrate", help="Explicitly migrate v700 weights into a new v701 contract"
    )
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--config", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--preserve-std", action="store_true")
    x = commands.add_parser(
        "experiment", help="Run the paired frozen-A / PPO-B / low-noise-C protocol"
    )
    x.add_argument("--plan", default="configs/ctbr_v701/experiment.json")
    x.add_argument("--source-checkpoint", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--seeds", type=int, nargs="+", default=[42])
    x.add_argument("--test-episodes", type=int)
    x.add_argument("--device", default="cpu")
    x.add_argument("--cpu-threads", type=int, default=1)
    for sub in (t, e, q, m):
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--cpu-threads", type=int, default=1)
    return p


def execute(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.command == "experiment":
        return run_experiment(
            args.plan,
            args.source_checkpoint,
            args.out,
            args.device,
            args.seeds,
            args.test_episodes,
        )
    if args.command == "migrate":
        _, receipt = migrate(
            args.checkpoint,
            load(args.config, args.device, args.seed),
            args.out,
            reset_std=not args.preserve_std,
        )
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
        return receipt
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
    error = report["mean_tail_position_rmse"]
    tail = f"{error:.3f} m" if error is not None else "N/A (no complete tail window)"
    print(
        f"{report['successes']}/{report['episodes']} sustained successes; tail RMSE {tail}; "
        f"precision <= {report['precision_rmse_limit_m']:g} m: {'passed' if report['precision_passed'] else 'failed'}"
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
