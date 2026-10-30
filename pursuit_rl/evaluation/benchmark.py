"""Shared seeded PPO/PN/APN rollouts and explicit episode-level metrics."""

import copy
import hashlib
import platform
from dataclasses import asdict
from pathlib import Path
from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.config import (
    load_config,
    validate,
    config_hash,
    dump_config,
    as_dict,
    Config,
)
from pursuit_rl.learning.contracts import require_compatible_policy
from pursuit_rl.evaluation.rollout import evaluate, task_hash
from pursuit_rl.artifacts.io import write_json
from pursuit_rl.evaluation.metrics import evaluate_trace
from pursuit_rl.evaluation.metrics.report import aggregate, write_summary, write_latex
from pursuit_rl.visualization.summary import plot_summary
from pursuit_rl.evaluation.scenarios import (
    SCENARIOS,
    configuration,
    evaluation_stage,
    BENCHMARK_VERSION,
)
from pursuit_rl.artifacts.trace import load_trace
from pursuit_rl import __version__
from pursuit_rl.artifacts.provenance import source_fingerprint
from pursuit_rl.artifacts.logging import ExpLogger

BENCHMARK_SCHEMA = "pursuit-benchmark-v6"


def rollout_controller(
    cfg, controller, episodes=32, seed=1234, out="benchmark_results", stage=None, disturbance=None
):
    if isinstance(controller, str):
        controller = GuidanceController(controller)
    return evaluate(
        controller, cfg, stage=stage, episodes=episodes, seed=seed, trace_dir=out,
        disturbance=disturbance,
    )


def evaluate_folder(folder):
    paths = sorted(Path(folder).glob("episode_*.npz"))
    if not paths:
        raise ValueError(f"No episode traces in {folder}")
    records = []
    for path in paths:
        trace = load_trace(path)
        records.append(
            {
                "file": path.name,
                "episode": int(trace["episode"]),
                **evaluate_trace(trace),
            }
        )
    return records


def _run_benchmark(
    cfg=None,
    algorithms=("PN", "APN"),
    checkpoint=None,
    episodes=32,
    seed=1234,
    out="benchmark_results",
    stage=None,
    scenarios=None,
    plots=True,
    logger=None,
    config_args=None,
    scenario_specs=None,
    baseline_parameters=None,
    disturbance=None,
):
    algorithms = tuple(algorithms)
    if (
        not algorithms
        or len(set(algorithms)) != len(algorithms)
        or any(a not in ("PPO", "PN", "APN", "PD") for a in algorithms)
    ):
        raise ValueError("Choose distinct algorithms from PPO, PN, APN, PD")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if scenario_specs and scenarios:
        raise ValueError("Custom scenario specifications and named scenarios are exclusive")
    if (scenarios or scenario_specs) and stage is not None:
        raise ValueError("Choose --stage or --scenarios, not both")
    model, checkpoint_data, digest = None, None, None
    if "PPO" in algorithms:
        if checkpoint is None:
            raise ValueError("PPO requires --checkpoint")
        from pursuit_rl.learning.checkpoints import load_model

        # Load exactly once; cfg is never passed as the device parameter.
        model, checkpoint_data = load_model(checkpoint)
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    base = copy.deepcopy(
        cfg
        if cfg is not None
        else (
            checkpoint_data["config"] if checkpoint_data is not None else load_config()
        )
    )
    if config_args is not None:
        fallback = base if checkpoint_data is not None or cfg is not None else None
        base = Config.from_args(config_args, fallback=fallback).to_dict()
    validate(base)
    if model is not None:
        import torch
        model.to(base['device'])
        model.device=next(model.actor.parameters()).device
        model.eval()
    if model is not None:
        require_compatible_policy(checkpoint_data["config"], base)
    selected_stage = base["training"]["max_stage"] if stage is None else stage
    if not 0 <= selected_stage < len(base["curriculum"]):
        raise ValueError("Invalid evaluation stage")
    if scenario_specs:
        from pursuit_rl.evaluation.scenario_grid import apply_scenario_spec
        if not isinstance(scenario_specs, dict) or not scenario_specs:
            raise ValueError("scenario_specs must be a nonempty map")
        jobs = [(name, *apply_scenario_spec(base, spec)) for name, spec in scenario_specs.items()]
    elif scenarios:
        if len(scenarios) != len(set(scenarios)) or any(
            name not in SCENARIOS for name in scenarios
        ):
            raise ValueError("Choose distinct registered scenarios")
        jobs = []
        for name in scenarios:
            case = configuration(base, name)
            jobs.append((name, case, evaluation_stage(case)))
    else:
        jobs = [("configured_stage", base, selected_stage)]
    root = Path(out)
    logger.set_config(base)
    protocol = {
        "schema": BENCHMARK_SCHEMA,
        "package_version": __version__,
        "source_sha256": source_fingerprint(),
        "python": platform.python_version(),
        "algorithms": list(algorithms),
        "episodes": episodes,
        "seed": seed,
        "batch_size": min(64, episodes),
        "stage": stage,
        "scenarios": scenarios,
        "base_config_hash": config_hash(base),
        "checkpoint_sha256": digest,
        "checkpoint_steps": checkpoint_data["steps"] if checkpoint_data else None,
        "information": "Exact simulated state, including target acceleration",
        "benchmark_version": BENCHMARK_VERSION,
        "baseline_parameters": baseline_parameters or GuidanceController("PN").configuration(),
        "disturbance": disturbance,
        "physics_dt_s": base["physics"]["dt"],
        "simulation_backend": base['simulation_backend'],
        "device": base['device'],
        "policy_interval_steps": base["physics"].get("policy_interval_steps", 1),
        "policy_frequency_hz": 1. / (base["physics"]["dt"] * base["physics"].get("policy_interval_steps", 1)),
        "trace_stride": base.get("evaluation", {}).get("trace_stride", 1),
        "custom_scenario_specs": scenario_specs,
        "tasks": {name: task_hash(case, case_stage) for name, case, case_stage in jobs},
    }
    write_json(root / "protocol.json", protocol)
    dump_config(base, root / "config.yaml")
    summaries = []
    for scenario, case_cfg, case_stage in jobs:
        if model is not None:
            require_compatible_policy(checkpoint_data["config"], case_cfg)
        for algorithm in algorithms:
            folder = root / algorithm / scenario
            if algorithm == "PPO":
                ctrl = model
            else:
                from pursuit_rl.control.guidance import GuidanceParameters
                params = (baseline_parameters or {}).get(algorithm, {})
                ctrl = GuidanceController(algorithm, GuidanceParameters(**params))
            logger.event("evaluation_started", algorithm=algorithm, scenario=scenario)
            result = rollout_controller(
                case_cfg, ctrl, episodes, seed, folder, case_stage, disturbance
            )
            result["evaluation_protocol"]["scenario"] = scenario
            metrics = evaluate_folder(folder)
            if len(metrics) != episodes:
                raise RuntimeError("Trace count does not match requested episodes")
            events = {r["episode"]: r for r in result["episodes_detail"]}
            for row in metrics:
                ep = row["episode"]
                rec = events[ep]
                if row["success"] != rec["success"] or (
                    rec["success"]
                    and abs(row["capture_time"] - rec["reach_time"]) > 1e-8
                ):
                    raise RuntimeError("Online and offline mission metrics disagree")
            write_json(folder / "episodes.json", result)
            write_json(folder / "metrics.json", metrics)
            row = {
                "algorithm": algorithm,
                "scenario": scenario,
                "stage": case_stage,
                "capture_radius": case_cfg["curriculum"][case_stage]["success_radius"],
                "config_hash": config_hash(case_cfg),
                "task_hash": task_hash(case_cfg, case_stage),
                "disturbance": disturbance,
                "baseline_parameters": (baseline_parameters or {}).get(algorithm, {}),
                **aggregate(metrics),
            }
            write_json(folder / "summary.json", row)
            dump_config(case_cfg, folder / "config.yaml")
            summaries.append(row)
            logger.record("benchmark.jsonl", row)
            logger.logger.info(
                "%s / %s: success=%.1f%%",
                algorithm,
                scenario,
                row["success_rate"] * 100,
            )
            logger.event(
                "evaluation_completed",
                algorithm=algorithm,
                scenario=scenario,
                episodes=episodes,
            )
    write_json(root / "summary.json", summaries)
    write_summary(summaries, root / "summary.csv")
    write_latex(summaries, root / "summary_table.tex")
    if plots:
        plot_summary(summaries, root / "summary.png")
    write_json(
        root / "complete.json", {"schema": BENCHMARK_SCHEMA, "rows": len(summaries)}
    )
    return summaries


def run_benchmark(
    cfg=None,
    algorithms=("PN", "APN"),
    checkpoint=None,
    episodes=32,
    seed=1234,
    out="benchmark_results",
    stage=None,
    scenarios=None,
    plots=True,
    invocation=None,
    config_args=None,
    scenario_specs=None,
    baseline_parameters=None,
    disturbance=None,
):
    cfg = as_dict(cfg)
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Benchmark output must be empty: {root}")
    with ExpLogger(
        root,
        "benchmark",
        invocation=invocation
        or {
            "algorithms": list(algorithms),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "episodes": episodes,
            "seed": seed,
            "stage": stage,
            "scenarios": scenarios,
        },
    ) as logger:
        logger.record_inputs({"checkpoint": checkpoint} if checkpoint else {})
        rows = _run_benchmark(
            cfg,
            algorithms,
            checkpoint,
            episodes,
            seed,
            out,
            stage,
            scenarios,
            plots,
            logger,
            config_args,
            scenario_specs,
            baseline_parameters,
            disturbance,
        )
        logger.complete({"rows": len(rows), "summary": str(root / "summary.json")})
        return rows
