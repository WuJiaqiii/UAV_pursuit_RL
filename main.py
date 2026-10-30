"""Unified CLI: argument parsing, configuration, input/output and workflow dispatch."""

import argparse
import json
import os
import sys
from pathlib import Path

from pursuit_rl import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pursuit RL: training, evaluation and analysis"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train, initialize or resume PPO")
    train.add_argument(
        "--config", help="Packaged defaults when omitted; saved config for resume"
    )
    train.add_argument(
        "--out", help="Explicit experiment directory; default is dated under runs"
    )
    source = train.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--init-from")
    train.add_argument("--seed", type=int)
    train.add_argument("--total-steps", type=int)
    train.add_argument("--num-envs", type=int)
    train.add_argument("--device", choices=["cpu", "cuda", "auto"])
    train.add_argument("--simulation-backend", choices=["numpy", "torch"])

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate PPO, PN, APN or PD online"
    )
    evaluate.add_argument(
        "--algorithm", choices=["PPO", "PN", "APN", "PD"], default="PPO"
    )
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--device", choices=["cpu", "cuda", "auto"])
    evaluate.add_argument("--simulation-backend", choices=["numpy", "torch"])
    evaluate.add_argument("--config")
    evaluate.add_argument(
        "--out", help="Explicit report JSON; default is inside the experiment"
    )
    evaluate.add_argument("--stage", type=int)
    evaluate.add_argument("--episodes", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=84000)
    evaluate.add_argument("--trace-dir")
    evaluate.add_argument(
        "--save-traces",
        action="store_true",
        help="Save trajectories in this evaluation result directory",
    )

    benchmark = commands.add_parser(
        "benchmark", help="Paired methods/scenarios with complete reports"
    )
    benchmark.add_argument(
        "--algorithms", nargs="+", choices=["PPO", "PN", "APN", "PD"], default=["PPO"]
    )
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--config")
    benchmark.add_argument("--episodes", type=int, default=32)
    benchmark.add_argument("--seed", type=int, default=1234)
    scenario = benchmark.add_mutually_exclusive_group()
    scenario.add_argument("--stage", type=int)
    from pursuit_rl.evaluation.scenarios import SCENARIOS

    scenario.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS))
    benchmark.add_argument(
        "--out", help="Explicit comparison directory; default is inside the experiment"
    )
    benchmark.add_argument("--no-plots", action="store_true")
    benchmark.add_argument("--device", choices=["cpu", "cuda", "auto"])
    benchmark.add_argument("--simulation-backend", choices=["numpy", "torch"])

    experiments = commands.add_parser(
        "experiments", help="E1–E5: maneuver, limit map, calibrated PN/APN, robustness, Pareto"
    )
    experiments.add_argument("--checkpoint", help="Required when PPO is included")
    experiments.add_argument("--config")
    experiments.add_argument("--device", choices=["cpu", "cuda", "auto"])
    experiments.add_argument("--simulation-backend", choices=["numpy", "torch"])
    experiments.add_argument("--suites", nargs="+", choices=["E1", "E2", "E3", "E4", "E5"],
                              default=["E1", "E2", "E3", "E4", "E5"])
    experiments.add_argument("--algorithms", nargs="+", choices=["PPO", "PN", "APN"],
                              default=["PPO", "PN", "APN"])
    experiments.add_argument("--episodes", type=int, default=128)
    experiments.add_argument("--calibration-episodes", type=int, default=16)
    experiments.add_argument("--seed", type=int, default=930000)
    experiments.add_argument("--out", default="runs/e1e5")
    experiments.add_argument("--quick", action="store_true", help="Small E1–E4 task subsets for smoke testing")
    experiments.add_argument("--no-plots", action="store_true")

    trace = commands.add_parser(
        "trace-metrics", help="Compute offline metrics from a saved episode"
    )
    trace.add_argument("trace")
    trace.add_argument("--out", help="Explicit metrics JSON")
    trace.add_argument("--capture-radius", type=float)

    visualize = commands.add_parser(
        "visualize", help="Trajectory, controls, capture details and optional animation"
    )
    visualize.add_argument("--trace", required=True)
    visualize.add_argument("--out", help="Explicit trajectory PNG")
    visualize.add_argument("--title", default="Pursuit trajectory")
    visualize.add_argument("--capture-radius", type=float)
    visualize.add_argument("--gif", action="store_true")
    visualize.add_argument("--mp4", action="store_true")
    visualize.add_argument("--fps", type=int, default=12)
    visualize.add_argument("--duration", type=float, default=12.0)
    visualize.add_argument("--max-frames", type=int, default=240)
    visualize.add_argument("--camera", choices=["global", "follow"], default="global")
    visualize.add_argument("--trail-seconds", type=float, default=0.0)
    visualize.add_argument("--rotate-degrees", type=float, default=0.0)

    report = commands.add_parser(
        "report", help="Export benchmark summary to CSV/LaTeX/figure"
    )
    report.add_argument("--input", required=True)
    report.add_argument("--out", help="Explicit CSV output")
    report.add_argument("--plot", help="Optional PNG output")

    compare = commands.add_parser("compare", help="Compare paired episode reports")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--out", help="Explicit comparison JSON")

    export = commands.add_parser(
        "export", help="Export lightweight experimental records"
    )
    export.add_argument("--source", help="Defaults to --run; otherwise required")
    export.add_argument("--out", help="Explicit ZIP output")
    export.add_argument("--max-file-mb", type=float, default=15.0)

    feasibility = commands.add_parser(
        "feasibility", help="Analyze optimistic interception geometry"
    )
    feasibility.add_argument("inputs", nargs="+")
    feasibility.add_argument("--out", help="Explicit CSV output")
    feasibility.add_argument("--max-speed", type=float)

    for command in [evaluate, trace, visualize, report, compare, export, feasibility]:
        command.add_argument(
            "--log-dir", help="Default: <output stem>.run beside the output"
        )
    for command in [
        train,
        evaluate,
        benchmark,
        trace,
        visualize,
        report,
        compare,
        export,
        feasibility,
    ]:
        command.add_argument(
            "--runs-dir",
            default=os.environ.get("PURSUIT_RUNS_DIR", "runs"),
            help="Root directory for new experiments",
        )
        command.add_argument("--name", help="Optional experiment/operation label")
    for command in [
        evaluate,
        benchmark,
        trace,
        visualize,
        report,
        compare,
        export,
        feasibility,
    ]:
        command.add_argument("--run", help="Attach results to an existing experiment")
    for command in [train, evaluate, benchmark]:
        command.add_argument(
            "--set",
            dest="set_values",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Validated dotted YAML override; repeatable",
        )
    return parser


def run_train(args):
    from pursuit_rl.config import Config
    from pursuit_rl.artifacts.layout import Experiment, find_experiment
    from pursuit_rl.training.trainer import Trainer

    config = Config.from_args(args)
    if args.resume:
        owner = find_experiment(args.resume)
        if not owner:
            raise ValueError("Resume checkpoint must belong to an existing experiment")
        experiment = Experiment(args.out) if args.out else owner
    else:
        experiment = Experiment.plan(
            "train", args.runs_dir, args.name, config.seed, args.out
        )
    args.out = str(experiment.directory)
    # Let Trainer perform its guard before writing experiment identity metadata.
    was_empty = not experiment.directory.exists() or not any(
        experiment.directory.iterdir()
    )
    try:
        trainer = Trainer(
            config,
            args.out,
            resume=args.resume,
            init_from=args.init_from,
            invocation={
                **vars(args),
                "effective_config": args.config
                or (
                    str(Path(args.resume).resolve().parent.parent / "config.yaml")
                    if args.resume
                    else "packaged defaults"
                ),
            },
        )
    except BaseException:
        if was_empty and (experiment.directory / "logs").is_dir():
            experiment.register("train", args.name, config.seed)
        raise
    experiment.register(
        "train",
        args.name,
        config.seed,
        sources=(
            [args.init_from or args.resume] if (args.init_from or args.resume) else []
        ),
    )
    trainer.exp_logger.metadata(experiment=str(experiment.directory))
    return trainer.run()


def run_benchmark(args):
    from pursuit_rl.evaluation.benchmark import run_benchmark as benchmark

    # A checkpoint fallback is resolved by the shared benchmark loader before overrides.
    return benchmark(
        None,
        args.algorithms,
        args.checkpoint,
        args.episodes,
        args.seed,
        args.out,
        args.stage,
        args.scenarios,
        not args.no_plots,
        invocation=vars(args),
        config_args=args,
    )


def run_evaluate(args, logger):
    from pursuit_rl.config import Config
    from pursuit_rl.evaluation.rollout import evaluate
    from pursuit_rl.artifacts.io import write_json

    if args.algorithm == "PPO":
        if not args.checkpoint:
            raise ValueError("PPO requires --checkpoint")
        from pursuit_rl.learning.checkpoints import load_model
        from pursuit_rl.learning.contracts import require_compatible_policy

        model, data = load_model(args.checkpoint)
        config = Config.from_args(args, fallback=data["config"])
        require_compatible_policy(data["config"], config.to_dict())
        import torch
        model.to(config['device'])
        model.device=next(model.actor.parameters()).device
        model.eval()
    else:
        from pursuit_rl.control.guidance import GuidanceController

        model, config = GuidanceController(args.algorithm), Config.from_args(args)
    logger.set_config(config)
    logger.metadata(device=str(getattr(model, "device", "cpu")))
    result = evaluate(
        model, config.to_dict(), args.stage, args.episodes, args.seed, args.trace_dir
    )
    write_json(args.out, result)
    summary = {key: value for key, value in result.items() if key != "episodes_detail"}
    logger.record("evaluation.jsonl", {"algorithm": args.algorithm, **summary})
    logger.logger.info(
        "%s: %.1f%% success over %s episodes",
        args.algorithm,
        result["success_rate"] * 100,
        args.episodes,
    )
    return {"report": args.out, **summary}


def run_analysis(args, logger, operation_directory=None):
    from pursuit_rl.artifacts.io import write_json

    if args.command == "trace-metrics":
        from pursuit_rl.artifacts.trace import load_trace
        from pursuit_rl.evaluation.metrics import evaluate_trace

        result = evaluate_trace(load_trace(args.trace), args.capture_radius)
        write_json(args.out, result)
        return result
    if args.command == "visualize":
        from pursuit_rl.artifacts.trace import load_trace
        from pursuit_rl.visualization.trajectory import plot_trace

        trace = load_trace(args.trace)
        if args.capture_radius is not None:
            from pursuit_rl.evaluation.metrics.mission import mission_metrics

            mission_metrics(trace, args.capture_radius)
            trace["capture_radius"] = args.capture_radius
        if (args.gif or args.mp4) and "capture_radius" not in trace:
            raise ValueError("Animation requires --capture-radius or saved metadata")
        outputs = plot_trace(trace, args.out, args.title)
        if args.gif or args.mp4:
            from pursuit_rl.visualization.animation import render_animation

            outputs += render_animation(
                trace,
                Path(args.out).with_suffix(""),
                gif=args.gif,
                mp4=args.mp4,
                fps=args.fps,
                duration=args.duration,
                max_frames=args.max_frames,
                camera=args.camera,
                trail_seconds=args.trail_seconds,
                rotate_degrees=args.rotate_degrees,
                title=args.title,
            )
        return {"outputs": [str(path) for path in outputs]}
    if args.command == "report":
        from pursuit_rl.evaluation.metrics.report import write_summary, write_latex

        rows = json.loads(Path(args.input).read_text(encoding="utf8"))
        if not isinstance(rows, list) or not rows or "success_rate" not in rows[0]:
            raise ValueError("Expected completed benchmark summary rows")
        write_summary(rows, args.out)
        write_latex(rows, Path(args.out).with_suffix(".tex"))
        if args.plot:
            from pursuit_rl.visualization.summary import plot_summary

            plot_summary(rows, args.plot)
        return {
            "rows": len(rows),
            "csv": args.out,
            "latex": str(Path(args.out).with_suffix(".tex")),
            "plot": args.plot,
        }
    if args.command == "compare":
        from pursuit_rl.evaluation.comparison import compare

        result = compare(args.a, args.b)
        write_json(args.out, result)
        logger.logger.info("Paired comparison: %s", result)
        return result
    if args.command == "export":
        from pursuit_rl.artifacts.export import export

        count = export(
            args.source,
            args.out,
            args.max_file_mb,
            exclude_paths=[
                operation_directory or logger.session_dir,
                logger.session_dir,
            ],
        )
        return {"files": count, "archive": args.out}
    if args.command == "feasibility":
        from pursuit_rl.evaluation.feasibility import produce

        rows = produce(args.inputs, args.out, args.max_speed)
        return {"episodes": len(rows), "csv": args.out}
    raise ValueError(f"Unknown analysis command: {args.command}")


def source_paths(args):
    paths = []
    for name in ("checkpoint", "trace", "input", "source", "a", "b", "inputs"):
        value = getattr(args, name, None)
        if value:
            paths.extend(value if isinstance(value, list) else [value])
    return paths


def execute(args):
    if args.command == "experiments":
        from pursuit_rl.evaluation.experiment_suite import run_suite
        return run_suite(checkpoint=args.checkpoint, suites=args.suites,
                         episodes=args.episodes, seed=args.seed,
                         out=args.out, calibration_episodes=args.calibration_episodes,
                         quick=args.quick, algorithms=args.algorithms,
                         config_path=args.config, plots=not args.no_plots,
                         device=args.device, backend=args.simulation_backend)
    if args.command == "train":
        return run_train(args)
    from pursuit_rl.artifacts.layout import select_experiment
    from pursuit_rl.artifacts.logging import ExpLogger

    if args.command == "export" and not args.source:
        if not args.run:
            raise ValueError("export requires --source or --run")
        args.source = args.run
    sources = source_paths(args)
    experiment = select_experiment(args, sources)
    explicit_output = Path(args.out).expanduser().resolve() if args.out else None
    if (
        args.command == "benchmark"
        and explicit_output
        and explicit_output.exists()
        and any(explicit_output.iterdir())
    ):
        raise FileExistsError(f"Benchmark output must be empty: {explicit_output}")
    directory = (
        explicit_output
        if args.command == "benchmark"
        else (
            explicit_output.with_name(explicit_output.stem + ".run")
            if explicit_output
            else None
        )
    )
    operation = experiment.operation(args.command, args.name, directory, sources)
    filenames = {
        "evaluate": "report.json",
        "trace-metrics": "metrics.json",
        "visualize": "trajectory.png",
        "report": "summary.csv",
        "compare": "comparison.json",
        "export": "records.zip",
        "feasibility": "feasibility.csv",
    }
    args.out = str(
        explicit_output
        or (
            operation.directory
            if args.command == "benchmark"
            else operation.directory / filenames[args.command]
        )
    )
    if args.command == "evaluate" and args.save_traces and not args.trace_dir:
        args.trace_dir = str(operation.directory / "traces")
    experiment.register(args.command, args.name, getattr(args, "seed", None), sources)
    status = "failed"
    try:
        if args.command == "benchmark":
            result = run_benchmark(args)
        else:
            with ExpLogger(
                args.log_dir or operation.directory, args.command, invocation=vars(args)
            ) as logger:
                logger.metadata(
                    experiment=str(experiment.directory),
                    operation=str(operation.directory),
                )
                inputs = {
                    name: getattr(args, name)
                    for name in (
                        "checkpoint",
                        "config",
                        "trace",
                        "input",
                        "source",
                        "a",
                        "b",
                        "inputs",
                    )
                    if getattr(args, name, None) is not None
                }
                logger.record_inputs(inputs)
                result = (
                    run_evaluate(args, logger)
                    if args.command == "evaluate"
                    else run_analysis(args, logger, operation.directory)
                )
                logger.complete(result)
        status = "completed"
        return result
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        operation.register(status, args.out)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        execute(args)
    except KeyboardInterrupt:
        print("Interrupted; session logs record the interruption.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
