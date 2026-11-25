"""V12 stage-to-stage pure PPO curriculum and held-out difficulty matrix.

Unlike run_pure_ppo_campaign.py (independent from-scratch runs), each
curriculum stage warm-starts from the preceding stage's best checkpoint.
No LSTM, DAgger, PD teacher, safety Critic or other auxiliary optimization.
Per-stage *held-out* evaluation uses fixed disjoint seeds and original task
profiles, preserving comparable success-rate denominators.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint
from pursuit_rl.ctbr.checkpoint import source_fingerprint
from pursuit_rl.ctbr.config import load as load_config, transfer_interface
from pursuit_rl.ctbr.evaluation import evaluate
from scripts.run_pure_ppo import TASKS
from scripts import run_pure_ppo as pure_runner
from scripts import run_pure_ppo_campaign as base

SCHEMA = "v12-pure-ppo-curriculum-v1"
DEFAULT_PLAN = "configs/experiments/11_v12_pure_ppo_curriculum.json"
EVAL_KEYS = frozenset(("profiles", "episodes", "seed", "gate_success_rate", "gate_mode"))
GATE_MODES = frozenset(("record_only", "stop"))
PLAN_KEYS = frozenset((
    "schema", "description", "defaults", "stages", "evaluation",
    "compare_direct_baseline", "start_from_required", "initial_parent_task",
))
FIELD_NAMES = (
    "arm", "stage", "stage_index", "seed", "train_task", "evaluated_task",
    "episodes", "successes", "success_rate",
    "success_ci95_lower", "success_ci95_upper",
    "before_success_rate", "learning_gain_pp",
    "tilt_failure_rate",
    "timeout_rate", "unsafe_entry_rate", "mean_maximum_tilt_deg",
    "maximum_tilt_p95_deg", "checkpoint", "gate_met",
)


def wilson95(successes, episodes):
    """95% binomial Wilson score interval (independent episodes)."""
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be positive")
    p = successes / episodes
    z = 1.959963984540054
    denominator = 1 + z*z / episodes
    center = (p + z*z / (2*episodes)) / denominator
    spread = (z / denominator) * math.sqrt(
        p*(1-p)/episodes + z*z/(4*episodes*episodes)
    )
    return max(0., center - spread), min(1., center + spread)


def validate_plan(plan, device):
    """Expand and freeze a full matrix; all seeds require same prior-stage seed."""
    if plan.get("schema") != SCHEMA or set(plan) - PLAN_KEYS:
        raise ValueError("Unsupported curriculum schema or unknown plan field")
    eval_cfg = plan.get("evaluation", {})
    if not isinstance(eval_cfg, dict) or set(eval_cfg) - EVAL_KEYS:
        raise ValueError("Invalid curriculum evaluation configuration")
    profiles = eval_cfg.get("profiles", list(TASKS))
    if (not isinstance(profiles, list) or not profiles or
            any(p not in TASKS for p in profiles) or
            len(profiles) != len(set(profiles))):
        raise ValueError("Unknown or repeated evaluation profile")
    episodes = eval_cfg.get("episodes", 64)
    seed = eval_cfg.get("seed", 920001)
    threshold = eval_cfg.get("gate_success_rate", 0.80)
    mode = eval_cfg.get("gate_mode", "record_only")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("Evaluation episodes must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("Evaluation gate success threshold must lie in [0,1]")
    if mode not in GATE_MODES:
        raise ValueError("Invalid curriculum gate_mode")
    requires_parent = plan.get("start_from_required", False)
    if type(requires_parent) is not bool:
        raise ValueError("start_from_required must be boolean")
    expected_parent = plan.get("initial_parent_task")
    if expected_parent is not None and expected_parent not in TASKS:
        raise ValueError("Unknown initial_parent_task")
    if expected_parent is not None and not requires_parent:
        raise ValueError("initial_parent_task requires start_from_required")
    if expected_parent is not None and expected_parent not in profiles:
        raise ValueError(
            "Continuation evaluation.profiles must include initial_parent_task "
            "to compute a fixed-seed pre-training success baseline"
        )
    compare = plan.get("compare_direct_baseline", True)
    if type(compare) is not bool:
        raise ValueError("compare_direct_baseline must be boolean")
    core = dict(
        schema=base.SCHEMA,
        defaults=plan.get("defaults", {}),
        stages=plan.get("stages", []),
    )
    stage_items = base.expand_plan(core, device)
    if len(stage_items) < 2:
        raise ValueError("A curriculum needs two or more stages")
    stages = list(dict.fromkeys(i["stage_index"] for i in stage_items))
    if stages != list(range(1, len(stages) + 1)):
        raise ValueError("Non-consecutive stage numbers")
    seeds_per_stage = {}
    for item in stage_items:
        seeds_per_stage.setdefault(item["stage_index"], set()).add(item["seed"])
    first = seeds_per_stage[1]
    for stage, seeds in seeds_per_stage.items():
        if seeds != first:
            raise ValueError(
                f"Stage {stage} seeds must exactly match stage 1; "
                "transfer is paired by training seed"
            )
    # Keep stage-major ordering, with optional scratch control for each
    # higher stage; stage 1 is shared by both arms and not trained twice.
    runs = []
    previous = {}
    for item in stage_items:
        curr = dict(item)
        curr["arm"] = "curriculum"
        curr["parent_id"] = previous.get(item["seed"])
        curr["id"] = f"curr_{item['id']}"
        runs.append(curr)
        if compare and item["stage_index"] > 1:
            direct = dict(item)
            direct["arm"] = "direct"
            direct["parent_id"] = None
            direct["id"] = f"direct_{item['id']}"
            runs.append(direct)
        previous[item["seed"]] = curr["id"]
    return runs, dict(
        profiles=profiles, episodes=episodes, seed=seed,
        gate_success_rate=float(threshold), gate_mode=mode,
        compare_direct_baseline=compare,
    )


def _matrix_report(model_path, training_config, item, evaluation, device):
    """One independently seeded deterministic policy test per difficulty."""
    model, trained, saved = load_checkpoint(model_path, device=device)
    if trained["prediction"]["enabled"] or trained["training"]["bc_updates"]:
        raise ValueError("Curriculum checkpoint is not a pure-PPO model")
    results = []
    for idx, task in enumerate(evaluation["profiles"]):
        cfg = copy.deepcopy(training_config)
        cfg["task"] = copy.deepcopy(load_config(TASKS[task])["task"])
        # Task-dependent nominal evaluation is deterministic for the
        # same model and profile across all curriculum stages and controls.
        report, _ = evaluate(
            model, cfg,
            episodes=evaluation["episodes"],
            seed=evaluation["seed"] + idx * 10007,
        )
        failures = report.get("failures", {})
        n = int(report["episodes"])
        lower, upper = wilson95(int(report["successes"]), n)
        results.append(dict(
            evaluated_task=task,
            episodes=n,
            successes=int(report["successes"]),
            success_rate=float(report["success_rate"]),
            success_ci95_lower=lower,
            success_ci95_upper=upper,
            tilt_failure_rate=int(failures.get("tilt", 0)) / n,
            timeout_rate=int(failures.get("timeout", 0)) / n,
            unsafe_entry_rate=float(report.get("unsafe_entry_rate", 0)),
            mean_maximum_tilt_deg=report.get("mean_maximum_tilt_deg"),
            maximum_tilt_p95_deg=report.get("maximum_tilt_p95_deg"),
            failures=failures,
            evaluation_seed=evaluation["seed"] + idx * 10007,
        ))
    own = next((r for r in results if r["evaluated_task"] == item["task"]), None)
    if own is None:
        raise ValueError("Own-stage task must be in evaluation profiles")
    gate_met = own["success_rate"] >= evaluation["gate_success_rate"]
    return dict(
        schema="v12-curriculum-stage-evaluation-v1",
        id=item["id"],
        arm=item["arm"],
        stage=item["stage"],
        stage_index=item["stage_index"],
        seed=item["seed"],
        task=item["task"],
        checkpoint=str(model_path.resolve()),
        checkpoint_sha256=base._hash_file(model_path),
        checkpoint_source_fingerprint=saved["source_fingerprint"],
        gate_success_rate=evaluation["gate_success_rate"],
        own_task_success_rate=own["success_rate"],
        gate_met=gate_met,
        results=results,
    )


def _stage_best(item, root):
    # The pre-existing trainer writes both best_current and latest; the
    # curriculum explicitly inherits the best *validation-selected* actor.
    checkpoint = root / item["id"] / "checkpoints" / "best_current.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing stage best checkpoint: {checkpoint}")
    return checkpoint


def _validate_saved_evaluation(report, item, model_path):
    if (report.get("id") != item["id"]
            or report.get("checkpoint_sha256") != base._hash_file(model_path)
            or report.get("checkpoint_source_fingerprint") != source_fingerprint()):
        raise ValueError(f"Invalid or stale curriculum evaluation: {item['id']}")


def _get_run_summary(item, root):
    result = root / item["id"] / "training_summary.json"
    if not result.exists():
        raise RuntimeError(f"Missing training completion: {item['id']}")
    data = base._read_json(result)
    if (data.get("source_fingerprint") != source_fingerprint()
            or int(data.get("physics_steps", 0)) < item["budget"]):
        raise ValueError(f"Training summary incompatible: {result}")
    return data


def _results_csv(reports, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for report in reports:
            if report is None:
                continue
            for entry in report["results"]:
                writer.writerow(dict(
                    arm=report["arm"], stage=report["stage"],
                    stage_index=report["stage_index"], seed=report["seed"],
                    train_task=report["task"],
                    evaluated_task=entry["evaluated_task"],
                    episodes=entry["episodes"],
                    successes=entry["successes"],
                    success_rate=entry["success_rate"],
                    success_ci95_lower=entry.get("success_ci95_lower"),
                    success_ci95_upper=entry.get("success_ci95_upper"),
                    before_success_rate=entry.get("before_success_rate"),
                    learning_gain_pp=entry.get("learning_gain_pp"),
                    tilt_failure_rate=entry["tilt_failure_rate"],
                    timeout_rate=entry["timeout_rate"],
                    unsafe_entry_rate=entry["unsafe_entry_rate"],
                    mean_maximum_tilt_deg=entry["mean_maximum_tilt_deg"],
                    maximum_tilt_p95_deg=entry["maximum_tilt_p95_deg"],
                    checkpoint=report["checkpoint"],
                    gate_met=report["gate_met"],
                ))


def _compare(reports, runs):
    """Paired, equal-budget curriculum vs scratch on the current difficulty."""
    by_id = {r["id"]: r for r in runs}
    groups = {}
    for report in reports.values():
        if report is None:
            continue
        key = report["stage_index"], report["seed"]
        groups.setdefault(key, {})[report["arm"]] = report
    contrasts = []
    for (stage, seed), sides in sorted(groups.items()):
        if "curriculum" not in sides or "direct" not in sides:
            continue
        c, d = sides["curriculum"], sides["direct"]
        # Both use exactly the same evaluation task and fixed episode seeds.
        c_rate, d_rate = c["own_task_success_rate"], d["own_task_success_rate"]
        contrasts.append(dict(
            stage=stage, seed=seed, task=c["task"],
            curriculum_success_rate=c_rate,
            direct_success_rate=d_rate,
            difference_pp=100 * (c_rate - d_rate),
            episodes_per_arm=next(
                x["episodes"] for x in c["results"]
                if x["evaluated_task"] == c["task"]
            ),
        ))
    return contrasts


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--start-from",
        help="Existing pure-PPO best_current.pt for the FIRST curriculum stage. "
             "Used for C3-to-C4 or high-to-extreme continuation, never in-place "
             "modification of an already-running campaign.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--log-interval", type=float, default=10.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    root = Path(args.out).resolve()
    state_file = root / "campaign_progress.json"
    if args.status:
        if not state_file.is_file():
            parser.error(f"No campaign progress file at {state_file}")
        data = base._read_json(state_file)
        print("\n".join(base.progress_lines(data)))
        for stage in data.get("stages", []):
            print(f"Stage {stage['name']}: {stage['completed']}/{stage['runs']} "
                  f"({stage['percent']:.1f}%)")
        results_path = root / "curriculum_results.json"
        if results_path.is_file():
            saved = base._read_json(results_path)
            print("Difficulty matrix (fixed heldout episodes, best checkpoint):")
            for record in saved.get("records", []):
                scores = "  ".join(
                    f"{x['evaluated_task']}={x['successes']}/{x['episodes']}"
                    for x in record.get("results", [])
                )
                print(
                    f"  {record['arm']} {record['stage']} seed={record['seed']} | "
                    f"{scores} | gate={'PASS' if record['gate_met'] else 'FAIL'}"
                )
        return data
    if args.poll_seconds <= 0 or args.log_interval <= 0 or args.cpu_threads <= 0:
        parser.error("Invalid polling/logging/CPU thread settings")
    plan_path = Path(args.plan).resolve()
    plan = base._read_json(plan_path)
    runs, evaluation = validate_plan(plan, args.device)
    if plan.get("start_from_required", False) and not args.start_from and not args.dry_run:
        parser.error(
            "This continuation curriculum requires --start-from PATH to the "
            "preceding stage's best_current.pt"
        )
    external_parent = None
    if args.start_from:
        # A single checkpoint cannot correctly initialize different train seeds.
        seeds = {x["seed"] for x in runs}
        if len(seeds) != 1:
            parser.error(
                "--start-from supports one training seed per campaign; "
                "use a separate campaign for each distinct parent seed"
            )
        checkpoint_path = Path(args.start_from).resolve()
        if not args.dry_run:
            if not checkpoint_path.is_file():
                parser.error(f"Missing --start-from checkpoint: {checkpoint_path}")
            model, prior_config, parent_data = load_checkpoint(
                checkpoint_path, device="cpu"
            )
            if (
                prior_config["prediction"]["enabled"]
                or prior_config["safety"]["predictive_enabled"]
                or prior_config["safety"]["constraint_enabled"]
                or prior_config["training"]["bc_updates"] != 0
                or prior_config["training"]["aux_pd_coef"] != 0
            ):
                parser.error("External parent must be a pure PPO checkpoint")
            expected_parent = plan.get("initial_parent_task")
            if expected_parent is not None:
                expected_name = load_config(TASKS[expected_parent])["task"]["name"]
                if prior_config["task"]["name"] != expected_name:
                    parser.error(
                        f"Expected parent task {expected_name}, but checkpoint "
                        f"was trained on {prior_config['task']['name']}"
                    )
            if transfer_interface(prior_config) != transfer_interface(runs[0]["config"]):
                parser.error(
                    "Parent Actor/Critic architecture, action/observation "
                    "or physical dynamics are incompatible with C4 training"
                )
            external_parent = dict(
                path=str(checkpoint_path),
                sha256=base._hash_file(checkpoint_path),
                task=prior_config["task"]["name"],
                origin=parent_data.get("origin"),
                source_fingerprint=parent_data.get("source_fingerprint"),
            )
            if external_parent["origin"] == "random_initial":
                print(
                    "[CURRICULUM] WARNING: external checkpoint is random_initial; "
                    "higher difficulty transfer may have no learned tracking skill.",
                    flush=True,
                )
            del model
    if any(i["task"] not in evaluation["profiles"] for i in runs):
        parser.error("Each curriculum task must be in evaluation.profiles")
    manifest = dict(
        schema=SCHEMA, plan_sha256=base._hash_file(plan_path),
        runner_sha256=base._hash_file(__file__),
        parent_runner_sha256=base._hash_file(base.__file__),
        pure_runner_sha256=base._hash_file(pure_runner.__file__),
        source_fingerprint=source_fingerprint(),
        device=args.device,
        evaluation=evaluation,
        external_parent=external_parent,
        # A plan marked continuation cannot silently become from-scratch.
        start_from_required=bool(plan.get("start_from_required", False)),
        runs=[dict(
            id=x["id"], arm=x["arm"], stage=x["stage"],
            task=x["task"], seed=x["seed"], budget=x["budget"],
            parent_id=x["parent_id"], config_fingerprint=x["config_fingerprint"],
        ) for x in runs],
    )
    if args.dry_run:
        print(json.dumps(dict(
            manifest=manifest,
            total_runs=len(runs),
            total_budget_steps=sum(x["budget"] for x in runs),
        ), ensure_ascii=False, indent=2))
        return None
    manifest_path = root / "campaign_manifest.json"
    if args.resume_existing:
        if not manifest_path.is_file() or base._read_json(manifest_path) != manifest:
            parser.error("Changed/missing curriculum manifest; cannot resume")
    else:
        if root.exists() and any(root.iterdir()):
            parser.error("Output directory exists; choose --resume-existing or a new path")
        root.mkdir(parents=True, exist_ok=True)
        base._atomic_json(manifest_path, manifest)
    lock = (root / ".campaign.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another curriculum campaign holds the run directory")
    prior = base._read_json(state_file) if args.resume_existing and state_file.exists() else {}
    elapsed_before = float(prior.get("elapsed_wall_s", 0))
    started = time.monotonic()
    states = {i["id"]: base.default_state(i) for i in runs}
    reports = {}
    external_parent_report = None
    checkpoints = {}
    active_id = None
    display = base.CampaignDisplay(interval=args.log_interval)
    (root / "logs").mkdir(exist_ok=True)
    previous_sigterm = signal.signal(signal.SIGTERM, base._handle_termination)

    def snapshot(status="running", force=False):
        data = base.compute_snapshot(
            runs, states, elapsed_before + time.monotonic() - started,
            status, active_id,
        )
        data["curriculum_evaluated"] = sum(r is not None for r in reports.values())
        data["curriculum_gates"] = [
            dict(id=k, gate_met=v["gate_met"],
                 success_rate=v["own_task_success_rate"])
            for k, v in reports.items() if v is not None
        ]
        base._atomic_json(state_file, data)
        display.render(data, force=force)
        return data

    try:
        old_child = prior.get("active_run") or {}
        if args.resume_existing and old_child.get("status") == "running":
            if base._pid_alive(old_child.get("child_pid")):
                raise RuntimeError("Training child still alive; refuse concurrent resume")
        for item in runs:
            state, resume_checkpoint = base.inspect_run(item, root)
            states[item["id"]] = state
            checkpoints[item["id"]] = resume_checkpoint
            matrix_path = root / item["id"] / "difficulty_matrix.json"
            if matrix_path.exists():
                checkpoint = _stage_best(item, root)
                report = base._read_json(matrix_path)
                _validate_saved_evaluation(report, item, checkpoint)
                reports[item["id"]] = report
                if state["status"] != "completed":
                    raise RuntimeError("Difficulty matrix exists before training completed")
            else:
                reports[item["id"]] = None
                if state["status"] == "completed":
                    state["status"] = "evaluating"
        if external_parent:
            parent_report_path = root / "external_parent_difficulty_matrix.json"
            if parent_report_path.is_file():
                external_parent_report = base._read_json(parent_report_path)
                if (
                    external_parent_report.get("checkpoint_sha256")
                    != external_parent["sha256"]
                    or external_parent_report.get("checkpoint_source_fingerprint")
                    != external_parent["source_fingerprint"]
                ):
                    raise ValueError(
                        "External C3 checkpoint/evaluation provenance changed"
                    )
            else:
                prior_item = dict(
                    id="external_parent",
                    arm="external",
                    stage="Previous curriculum",
                    stage_index=0,
                    seed=runs[0]["seed"],
                    task=plan["initial_parent_task"],
                )
                external_parent_report = _matrix_report(
                    Path(external_parent["path"]), runs[0]["config"],
                    prior_item, evaluation, args.device
                )
                base._atomic_json(parent_report_path, external_parent_report)
                if args.device.startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()
        snapshot(force=True)
        for item in runs:
            state = states[item["id"]]
            active_id = item["id"]
            if reports[item["id"]] is not None and state["status"] == "completed":
                continue
            parent_id = item["parent_id"]
            parent_ckpt = None
            if parent_id:
                parent_report = reports.get(parent_id)
                if parent_report is None:
                    raise RuntimeError(f"Missing previous curriculum stage {parent_id}")
                parent_ckpt = _stage_best(
                    next(x for x in runs if x["id"] == parent_id), root
                )
                if (evaluation["gate_mode"] == "stop"
                        and not parent_report["gate_met"]):
                    state["status"] = "blocked_by_gate"
                    snapshot("gate_failed", force=True)
                    return base._read_json(state_file)
            # Only the first curriculum course may inherit from a
            # previously completed and separate C3 (or high random) campaign.
            if (
                parent_id is None and item["arm"] == "curriculum"
                and item["stage_index"] == 1 and external_parent is not None
            ):
                parent_ckpt = Path(external_parent["path"])
            directory = root / item["id"]
            # Parent checkpoint ancestry is persistent across interruption.
            # Place this outside the run dir because the standalone PPO
            # trainer refuses a nonempty directory for a fresh run.
            if parent_ckpt:
                dependency = dict(
                    parent_id=parent_id if parent_id is not None else "external",
                    parent_sha256=base._hash_file(parent_ckpt),
                    parent_path=str(parent_ckpt.resolve()),
                )
                ancestry = root / "dependencies" / (item["id"] + ".json")
                if ancestry.exists():
                    if base._read_json(ancestry) != dependency:
                        raise RuntimeError(
                            "Curriculum parent checkpoint was replaced; "
                            "refusing unsafe resume"
                        )
                elif (directory / "config.json").exists():
                    raise RuntimeError(
                        "Resumed transferred stage lacks checkpoint ancestry"
                    )
                else:
                    base._atomic_json(ancestry, dependency)
            state["status"] = "running"
            snapshot(force=True)
            if checkpoints[item["id"]] is None and not (directory / "training_summary.json").exists():
                command = base.run_command(item, directory, args.device)
                if parent_ckpt:
                    command += ["--init-from", str(parent_ckpt)]
                logfile = root / "logs" / f"{item['id']}.log"
                with logfile.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n# {base._now()} cmd={command!r}\n")
                    stream.flush()
                    tail = base.MetricsTail(directory / "metrics.jsonl")
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
                    child = subprocess.Popen(
                        command, stdout=stream, stderr=subprocess.STDOUT,
                        env=env,
                    )
                    state["child_pid"] = getattr(child, "pid", None)
                    snapshot(force=True)
                    try:
                        while True:
                            base.update_from_metrics(state, tail.poll())
                            snapshot()
                            code = child.poll()
                            if code is not None:
                                break
                            time.sleep(args.poll_seconds)
                    except KeyboardInterrupt:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
                base.update_from_metrics(state, tail.poll())
                if code:
                    state["status"] = "failed"
                    state.pop("child_pid", None)
                    snapshot("failed", force=True)
                    raise RuntimeError(f"Child exited {code}: {logfile}")
            elif state["status"] != "completed" and checkpoints[item["id"]] is not None:
                command = base.run_command(
                    item, directory, args.device,
                    resume_checkpoint=checkpoints[item["id"]],
                )
                logfile = root / "logs" / f"{item['id']}.log"
                with logfile.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n# {base._now()} resumed cmd={command!r}\n")
                    stream.flush()
                    tail = base.MetricsTail(directory / "metrics.jsonl", resume=True)
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
                    child = subprocess.Popen(
                        command, stdout=stream, stderr=subprocess.STDOUT,
                        env=env,
                    )
                    state["child_pid"] = getattr(child, "pid", None)
                    snapshot(force=True)
                    try:
                        while True:
                            base.update_from_metrics(state, tail.poll())
                            snapshot()
                            code = child.poll()
                            if code is not None:
                                break
                            time.sleep(args.poll_seconds)
                    except KeyboardInterrupt:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
                base.update_from_metrics(state, tail.poll())
                if code:
                    state["status"] = "failed"
                    state.pop("child_pid", None)
                    snapshot("failed", force=True)
                    raise RuntimeError(f"Resumed child exited {code}: {logfile}")
            summary = _get_run_summary(item, root)
            state["steps"] = item["budget"]
            state["ppo_updates"] = int(summary["ppo_updates"])
            state["completed_episodes"] = int(summary["completed_episodes"])
            state["successful_episodes"] = int(summary["successful_episodes"])
            if state["completed_episodes"]:
                state["training_success_rate"] = (
                    state["successful_episodes"] / state["completed_episodes"]
                )
            state.pop("child_pid", None)
            state["status"] = "evaluating"
            snapshot(force=True)
            checkpoint = _stage_best(item, root)
            matrix_path = directory / "difficulty_matrix.json"
            if matrix_path.exists():
                report = base._read_json(matrix_path)
                _validate_saved_evaluation(report, item, checkpoint)
            else:
                try:
                    report = _matrix_report(
                        checkpoint, item["config"], item, evaluation, args.device
                    )
                    base._atomic_json(matrix_path, report)
                finally:
                    # Evaluation is in the scheduler process. Release cached
                    # GPU tensors before launching the next PPO child.
                    if args.device.startswith("cuda"):
                        import torch
                        torch.cuda.empty_cache()
            reports[item["id"]] = report
            # Compare BEFORE and AFTER the same stage on identical, held-out
            # episode seeds. The previous stage's full difficulty matrix
            # supplies the pre-training baseline with no extra evaluation.
            prior_report = (
                reports.get(parent_id) if parent_id else
                external_parent_report if item["arm"] == "curriculum"
                else None
            )
            if prior_report:
                before_by_task = {
                    entry["evaluated_task"]: entry["success_rate"]
                    for entry in prior_report.get("results", [])
                }
                for entry in report["results"]:
                    before = before_by_task.get(entry["evaluated_task"])
                    entry["before_success_rate"] = before
                    entry["learning_gain_pp"] = (
                        100 * (entry["success_rate"] - before)
                        if before is not None else None
                    )
                base._atomic_json(matrix_path, report)
            state["status"] = "completed"
            snapshot(force=True)
            base._atomic_json(root / "curriculum_results.json", dict(
                schema="v12-pure-ppo-curriculum-results-v1",
                records=[reports[x["id"]] for x in runs if reports.get(x["id"])],
                external_parent=external_parent_report,
                contrasts=_compare(reports, runs),
            ))
            _results_csv(
                [reports.get(x["id"]) for x in runs],
                root / "curriculum_success_rates.csv",
            )
            print(
                f"[CURRICULUM] Stage={item['stage']} arm={item['arm']} "
                f"seed={item['seed']} own_success={report['own_task_success_rate']:.1%} "
                f"gate={'PASS' if report['gate_met'] else 'FAIL'}",
                flush=True,
            )
        active_id = None
        return snapshot("completed", force=True)
    except KeyboardInterrupt:
        snapshot("interrupted", force=True)
        raise
    except Exception:
        snapshot("failed", force=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    run()
