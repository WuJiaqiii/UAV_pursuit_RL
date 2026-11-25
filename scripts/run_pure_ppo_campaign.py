"""Sequential four-task V12 pure-PPO campaign with persistent, weighted progress.

The campaign itself never computes PPO gradients. Every stage invokes the
existing standalone pure-PPO runner, preserving its checkpoint and RNG contract.
An interrupted stage resumes its exact latest checkpoint, never from a
partially completed PPO minibatch. The campaign weights progress by planned
physics steps, not by completed run count.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from pursuit_rl.ctbr.checkpoint import source_fingerprint
from pursuit_rl.ctbr.config import fingerprint
from scripts.run_pure_ppo import PRESETS, TASKS, make_config

def _handle_termination(_signum, _frame):
    raise KeyboardInterrupt


def _pid_alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


SCHEMA = "v12-pure-ppo-campaign-v1"
DEFAULT_PLAN = "configs/experiments/10_v12_pure_ppo_campaign.json"
DEFAULTS_KEYS = frozenset((
    "preset", "num_envs", "rollout_steps", "batch_size",
    "epochs", "history_steps", "legacy_terminal_records",
))
STAGE_KEYS = frozenset(("name", "task", "seeds", "preset", "physics_steps"))
PROGRESS_SCHEMA = "v12-pure-ppo-campaign-progress-v1"


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(obj, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_plan(plan, device):
    """Validate and freeze every stage into a standalone pure-PPO config."""
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"Expected campaign schema {SCHEMA}")
    if set(plan) - {"schema", "description", "defaults", "stages"}:
        raise ValueError("Unknown campaign plan field")
    defaults = plan.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - DEFAULTS_KEYS:
        raise ValueError("Invalid campaign defaults")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Campaign must declare a nonempty stages list")
    names = set()
    runs = []
    for stage_index, stage in enumerate(stages, 1):
        if not isinstance(stage, dict) or set(stage) - STAGE_KEYS:
            raise ValueError(f"Invalid stage definition {stage_index}")
        task = stage.get("task")
        seeds = stage.get("seeds")
        preset = stage.get("preset", defaults.get("preset", "quick"))
        name = stage.get("name", task)
        if (task not in TASKS or preset not in PRESETS or
                not isinstance(name, str) or not name.strip() or name in names):
            raise ValueError(f"Invalid or repeated task/stage {stage_index}")
        names.add(name)
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or seed < 0 for seed in seeds) or
                len(seeds) != len(set(seeds))):
            raise ValueError(f"Invalid stage {name} seeds")
        budget = stage.get("physics_steps")
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ValueError(f"Invalid stage {name} physics_steps")
        for seed in seeds:
            args = SimpleNamespace(
                task=task, preset=preset, device=device, seed=seed,
                num_envs=defaults.get("num_envs"),
                rollout_steps=defaults.get("rollout_steps"),
                batch_size=defaults.get("batch_size"),
                epochs=defaults.get("epochs"),
                history_steps=defaults.get("history_steps"),
                physics_steps=budget,
                legacy_terminal_records=defaults.get("legacy_terminal_records", False),
            )
            config = make_config(args)
            index = len(runs) + 1
            runs.append(dict(
                id=f"{index:02d}_{task}_seed{seed}",
                stage=name,
                stage_index=stage_index,
                task=task, seed=seed, preset=preset,
                budget=int(config["training"]["physics_steps"]),
                config_fingerprint=fingerprint(config),
                config=config,
            ))
    return runs


def default_state(run):
    return dict(
        id=run["id"], stage=run["stage"], task=run["task"],
        seed=run["seed"], budget=run["budget"], steps=0,
        status="pending", ppo_updates=0, completed_episodes=0,
        successful_episodes=0, training_success_rate=None,
    )


def compute_snapshot(runs, states, elapsed_s, status="running", active_id=None):
    """Pure deterministic aggregation, suitable for tests and monitor CLI."""
    total = sum(x["budget"] for x in runs)
    done = sum(
        min(max(int(states[x["id"]]["steps"]), 0), x["budget"])
        for x in runs
    )
    completed = sum(states[x["id"]]["status"] == "completed" for x in runs)
    rate = done / elapsed_s if elapsed_s > 0 and done > 0 else None
    eta = (total - done) / rate if rate else None
    current = states.get(active_id) if active_id else None
    stage_summaries = []
    names = list(dict.fromkeys(x["stage"] for x in runs))
    for name in names:
        group = [x for x in runs if x["stage"] == name]
        group_budget = sum(x["budget"] for x in group)
        stage_steps = sum(
            min(max(states[x["id"]]["steps"], 0), x["budget"])
            for x in group
        )
        stage_summaries.append(dict(
            name=name, runs=len(group),
            completed=sum(states[x["id"]]["status"] == "completed" for x in group),
            budget=group_budget, steps=stage_steps,
            percent=100 * stage_steps / group_budget,
        ))
    return dict(
        schema=PROGRESS_SCHEMA,
        updated_at=_now(),
        status=status,
        total_runs=len(runs),
        completed_runs=completed,
        total_budget_steps=total,
        completed_budget_steps=done,
        percent=100 * done / total,
        elapsed_wall_s=max(elapsed_s, 0),
        estimated_remaining_s=eta,
        effective_physics_steps_per_s=rate,
        active_run=dict(current) if current else None,
        stages=stage_summaries,
        runs=[dict(states[x["id"]]) for x in runs],
    )


def _hms(value):
    if value is None:
        return "--:--"
    seconds = max(int(value), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _progress_bar(percentage, width=22):
    fill = round(width * min(max(percentage, 0), 100) / 100)
    return "[" + "█" * fill + "░" * (width - fill) + "]"


def progress_lines(data):
    """Two consistent lines: whole experiment and current task/seed."""
    current = data.get("active_run")
    overall = (
        f"Overall {_progress_bar(data['percent'])} {data['percent']:5.1f}% | "
        f"{data['completed_runs']}/{data['total_runs']} runs | "
        f"{data['completed_budget_steps']:,}/{data['total_budget_steps']:,} steps | "
        f"Elapsed {_hms(data['elapsed_wall_s'])} | "
        f"ETA {_hms(data['estimated_remaining_s'])} | {data['status']}"
    )
    if not current:
        return overall, "Current none"
    fraction = 100 * min(current["steps"], current["budget"]) / current["budget"]
    train_rate = current["training_success_rate"]
    sr = f"{100 * train_rate:.1f}%" if train_rate is not None else "n/a"
    detail = (
        f"Current {_progress_bar(fraction)} {fraction:5.1f}% | "
        f"{current['stage']} / {current['task']} seed={current['seed']} | "
        f"{current['steps']:,}/{current['budget']:,} steps | "
        f"PPO {current['ppo_updates']} | Train SR {sr} | {current['status']}"
    )
    return overall, detail


class CampaignDisplay:
    """TTY two-line display; non-TTY periodic append-only status lines."""

    def __init__(self, stream=None, interval=10.0, clock=None):
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.interval = max(float(interval), 0.0)
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.last = None
        self.shown = False

    def render(self, data, force=False):
        now = self.clock()
        if not force and self.last is not None and now - self.last < self.interval:
            return
        overall, detail = progress_lines(data)
        if self.tty:
            if self.shown:
                self.stream.write("\x1b[2A")
            self.stream.write("\r\x1b[2K" + overall + "\n")
            self.stream.write("\r\x1b[2K" + detail + "\n")
            self.shown = True
        else:
            self.stream.write("[CAMPAIGN] " + overall + "\n")
            self.stream.write("[CURRENT]  " + detail + "\n")
        self.stream.flush()
        self.last = now


class MetricsTail:
    """Read only fresh complete JSONL lines; ignore stale pre-resume records."""

    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if resume and self.path.exists() else 0

    def poll(self):
        if not self.path.exists():
            return None
        if self.path.stat().st_size < self.offset:
            self.offset = 0
        latest = None
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                raw = handle.readline()
                if not raw or not raw.endswith(b"\n"):
                    break
                self.offset = handle.tell()
                try:
                    item = json.loads(raw)
                except (UnicodeError, ValueError):
                    continue
                if item.get("event") == "ppo_update":
                    latest = item
        return latest


def update_from_metrics(state, row):
    if row is None:
        return
    state["steps"] = min(int(row["physics_steps"]), state["budget"])
    state["ppo_updates"] = int(row.get("ppo_updates", 0))
    state["completed_episodes"] = int(row.get("completed_episodes_total", 0))
    state["successful_episodes"] = int(row.get("successful_episodes_total", 0))
    total = state["completed_episodes"]
    state["training_success_rate"] = (
        state["successful_episodes"] / total if total else None
    )


def run_command(item, directory, device, resume_checkpoint=None):
    """Use one existing standalone PPO subprocess for each campaign slot."""
    command = [
        sys.executable, "-u", "-m", "scripts.run_pure_ppo",
        "--out", str(directory), "--device", device,
        "--task", item["task"], "--preset", item["preset"],
        "--physics-steps", str(item["budget"]), "--no-progress",
    ]
    if resume_checkpoint:
        return command + ["--resume", str(resume_checkpoint)]
    config = item["config"]
    command += [
        "--seed", str(item["seed"]),
        "--num-envs", str(config["num_envs"]),
        "--rollout-steps", str(config["training"]["rollout_steps"]),
        "--batch-size", str(config["training"]["batch_size"]),
        "--epochs", str(config["training"]["epochs"]),
        "--history-steps", str(config["policy"]["history_steps"]),
    ]
    if config["training"]["fast_terminal_records"] is False:
        command.append("--legacy-terminal-records")
    return command


def load_checkpoint_steps(checkpoint, expected):
    """Inspect a trusted local checkpoint without instantiating a GPU model."""
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["source_fingerprint"] != source_fingerprint():
        raise ValueError("Checkpoint source fingerprint differs; cannot resume")
    if fingerprint(saved["config"]) != expected["config_fingerprint"]:
        raise ValueError("Checkpoint config differs from campaign plan")
    return saved["progress"]


def inspect_run(item, root):
    """Rebuild progress from summaries/checkpoints, not an optimistic UI cache."""
    directory = root / item["id"]
    state = default_state(item)
    if not directory.exists():
        return state, None
    config_file = directory / "config.json"
    if not config_file.exists():
        if any(directory.iterdir()):
            raise RuntimeError(
                f"{directory} has partial files but no config; archive it manually"
            )
        return state, None
    if fingerprint(_read_json(config_file)) != item["config_fingerprint"]:
        raise ValueError(f"Existing run config does not match plan: {directory}")
    summary = directory / "training_summary.json"
    if summary.exists():
        result = _read_json(summary)
        if (result.get("source_fingerprint") != source_fingerprint()
                or result.get("physics_steps", 0) < item["budget"]):
            raise ValueError(f"Existing summary incompatible or incomplete: {summary}")
        state["status"] = "completed"
        state["steps"] = item["budget"]
        state["ppo_updates"] = int(result.get("ppo_updates", 0))
        state["completed_episodes"] = int(result.get("completed_episodes", 0))
        state["successful_episodes"] = int(result.get("successful_episodes", 0))
        if state["completed_episodes"]:
            state["training_success_rate"] = (
                state["successful_episodes"] / state["completed_episodes"]
            )
        return state, None
    for filename in ("latest.pt", "initial.pt", "bc_initial.pt"):
        checkpoint = directory / "checkpoints" / filename
        if checkpoint.exists():
            saved = load_checkpoint_steps(checkpoint, item)
            state["steps"] = min(int(saved.get("physics_steps", 0)), item["budget"])
            state["ppo_updates"] = int(saved.get("ppo_updates", 0))
            state["completed_episodes"] = int(saved.get("completed_episodes", 0))
            state["successful_episodes"] = int(saved.get("successful_episodes", 0))
            if state["completed_episodes"]:
                state["training_success_rate"] = (
                    state["successful_episodes"] / state["completed_episodes"]
                )
            return state, checkpoint
    raise RuntimeError(
        f"Run {directory} was interrupted before its first checkpoint; "
        "archive the incomplete run directory manually before retrying"
    )


def create_manifest(plan_path, device, runs):
    return dict(
        schema=SCHEMA,
        plan_sha256=_hash_file(plan_path),
        runner_sha256=_hash_file(__file__),
        source_fingerprint=source_fingerprint(),
        device=device,
        runs=[dict(
            id=x["id"], stage=x["stage"], task=x["task"], seed=x["seed"],
            preset=x["preset"], budget=x["budget"],
            config_fingerprint=x["config_fingerprint"],
        ) for x in runs],
    )


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true",
                        help="Read last campaign_progress.json; no GPU or training")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--log-interval", type=float, default=10.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    root = Path(args.out).resolve()
    state_path = root / "campaign_progress.json"
    if args.status:
        if not state_path.exists():
            parser.error(f"No progress file at {state_path}")
        current = _read_json(state_path)
        print("\n".join(progress_lines(current)))
        print(f"Last updated: {current['updated_at']}")
        return current
    if args.poll_seconds <= 0 or args.log_interval <= 0 or args.cpu_threads < 1:
        parser.error("Poll/log intervals and CPU thread count must be positive")
    plan_path = Path(args.plan).resolve()
    plan = _read_json(plan_path)
    runs = expand_plan(plan, args.device)
    manifest = create_manifest(plan_path, args.device, runs)
    if args.dry_run:
        print(json.dumps(dict(
            manifest=manifest, total_runs=len(runs),
            total_budget_steps=sum(item["budget"] for item in runs),
            stages=[dict(stage=item["stage"], task=item["task"], seed=item["seed"],
                         budget=item["budget"]) for item in runs],
        ), ensure_ascii=False, indent=2))
        return None
    if args.resume_existing:
        if not (root / "campaign_manifest.json").exists():
            parser.error("Cannot resume: campaign_manifest.json is missing")
        if _read_json(root / "campaign_manifest.json") != manifest:
            parser.error(
                "Campaign source/plan/device changed. Start a new --out directory."
            )
    else:
        if root.exists() and any(root.iterdir()):
            parser.error("Existing campaign detected; use --resume-existing")
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "campaign_manifest.json", manifest)

    # Linux flock prevents two campaign launchers from editing the same
    # state/checkpoints concurrently; --status stays lock-free.
    lock = (root / ".campaign.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another campaign process owns this directory")
    prior = _read_json(state_path) if args.resume_existing and state_path.exists() else {}
    previous_elapsed = float(prior.get("elapsed_wall_s", 0))
    started = time.monotonic()
    current_id = None
    states = {}
    resume_files = {}
    display = CampaignDisplay(interval=args.log_interval)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    previous_termination_handler = signal.signal(
        signal.SIGTERM, _handle_termination
    )

    def snapshot(status="running", force=False):
        elapsed = previous_elapsed + (time.monotonic() - started)
        data = compute_snapshot(runs, states, elapsed, status, current_id)
        _atomic_json(state_path, data)
        display.render(data, force=force)
        return data

    try:
        previous_child = prior.get("active_run") or {}
        if (args.resume_existing
                and previous_child.get("status") == "running"
                and _pid_alive(previous_child.get("child_pid"))):
            raise RuntimeError(
                "Previous training subprocess is still running; stop it "
                "before resuming the same checkpoint directory."
            )
        for item in runs:
            state, checkpoint = inspect_run(item, root)
            states[item["id"]] = state
            resume_files[item["id"]] = checkpoint
        snapshot("running", force=True)
        for item in runs:
            state = states[item["id"]]
            if state["status"] == "completed":
                continue
            current_id = item["id"]
            state["status"] = "running"
            snapshot(force=True)
            directory = root / item["id"]
            command = run_command(
                item, directory, args.device, resume_files[item["id"]]
            )
            logfile = logs / f"{item['id']}.log"
            with logfile.open("a", encoding="utf-8") as output:
                output.write(f"\n# {_now()} cmd={command!r}\n")
                output.flush()
                cursor = MetricsTail(
                    directory / "metrics.jsonl",
                    resume=resume_files[item["id"]] is not None,
                )
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(args.cpu_threads)
                child = subprocess.Popen(
                    command, stdout=output, stderr=subprocess.STDOUT,
                    env=env,
                )
                state["child_pid"] = getattr(child, "pid", None)
                snapshot(force=True)
                try:
                    while True:
                        row = cursor.poll()
                        update_from_metrics(state, row)
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
            # Final flush after process exit.
            update_from_metrics(state, cursor.poll())
            if code != 0:
                state["child_pid"] = None
                state["status"] = "failed"
                snapshot("failed", force=True)
                raise RuntimeError(
                    f"{item['id']} exited {code}; inspect {logfile}. "
                    "Use --resume-existing once the cause is corrected "
                    "without changing code or campaign plan."
                )
            result_path = directory / "training_summary.json"
            if not result_path.exists():
                raise RuntimeError(f"Missing completed training summary: {result_path}")
            result = _read_json(result_path)
            if result.get("physics_steps", 0) < item["budget"]:
                raise RuntimeError(f"Stage did not exhaust training budget: {item['id']}")
            state["status"] = "completed"
            state.pop("child_pid", None)
            state["steps"] = item["budget"]
            state["ppo_updates"] = int(result["ppo_updates"])
            state["completed_episodes"] = int(result["completed_episodes"])
            state["successful_episodes"] = int(result["successful_episodes"])
            if state["completed_episodes"]:
                state["training_success_rate"] = (
                    state["successful_episodes"] / state["completed_episodes"]
                )
            snapshot(force=True)
            _atomic_json(root / "campaign_results.json", dict(
                schema="v12-pure-ppo-campaign-results-v1",
                runs=[dict(states[x["id"]]) for x in runs],
            ))
        current_id = None
        return snapshot("completed", force=True)
    except KeyboardInterrupt:
        snapshot("interrupted", force=True)
        raise
    except Exception:
        snapshot("failed", force=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_termination_handler)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    run()
