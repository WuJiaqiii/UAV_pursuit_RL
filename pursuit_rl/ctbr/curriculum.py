"""Capability screening, small difficulty increments, and retained-skill gates."""

import copy
import hashlib
import json
from pathlib import Path
from .config import load as config_load, fingerprint, validate
from .checkpoint import load, write_json, source_fingerprint
from .training import train
from .evaluation import evaluate
from .errors import ExperimentGateError
from .selection import summary as selection_summary

SCHEMA = "pursuit-direct-ctbr-curriculum-v4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def gate(
    model,
    c,
    sentinels,
    directory,
    episodes,
    seed,
    minimum_success,
    plan,
    label,
    fail_fast=False,
):
    directory = Path(directory) / label
    current, retained = [], []

    def finish(ok, complete=True, reason=None):
        result = dict(
            passed=ok,
            current=current,
            retention=retained,
            complete=complete,
            reason=reason,
        )
        write_json(directory / "summary.json", result)
        return result

    for offset in plan.get("gate_seed_offsets", [79000, 179000]):
        report, _ = evaluate(model, c, episodes, offset + seed)
        write_json(directory / f"current_{offset + seed}.json", report)
        item = selection_summary(report, c, minimum_success)
        current.append(item)
        if fail_fast and not item["passed"]:
            return finish(False, False, "current_task_failed")

    relative = plan.get("retention_rmse_relative_budget", 0.2)
    slack = plan.get("retention_rmse_absolute_slack_m", 0.05)
    # A failed current task can never be promoted. Full final audit can still
    # request retention reports; periodic/prescreen checks return immediately.
    for anchor in sentinels:
        for reference in anchor["reference"]:
            report, _ = evaluate(model, anchor["config"], episodes, reference["seed"])
            write_json(
                directory / "retention" / f"{anchor['name']}_{reference['seed']}.json",
                report,
            )
            rmse = report["mean_tail_position_rmse"]
            bound = reference["tail_rmse"] * (1 + relative) + slack
            item = dict(
                task=anchor["name"],
                seed=reference["seed"],
                success_rate=report["success_rate"],
                tail_rmse=rmse,
                rmse_budget=bound,
                passed=report["success_rate"] >= minimum_success
                and rmse is not None
                and rmse <= bound,
            )
            retained.append(item)
            if fail_fast and not item["passed"]:
                return finish(False, False, "retained_skill_failed")
    return finish(
        all(r["passed"] for r in current) and all(r["passed"] for r in retained)
    )


def run(
    plan_path,
    out,
    device="cpu",
    seed=42,
    steps=None,
    episodes=32,
    minimum_success=0.95,
    phases=None,
    init_from=None,
):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("Wrong curriculum schema")
    if not 0 <= minimum_success <= 1 or episodes < 1:
        raise ValueError("Invalid validation gate")
    for key, default in (
        ("gate_seed_offsets", [79000, 179000]),
        ("retention_seed_offsets", [89000, 189000]),
    ):
        offsets = plan.get(key, default)
        if (
            len(offsets) < 2
            or len(set(offsets)) != len(offsets)
            or any(type(v) is not int or v < 0 for v in offsets)
        ):
            raise ValueError(
                "Each curriculum gate requires at least two distinct validation seeds"
            )
    if (
        plan.get("retention_rmse_relative_budget", 0.2) < 0
        or plan.get("retention_rmse_absolute_slack_m", 0.05) < 0
    ):
        raise ValueError("Invalid retention precision budget")
    names = [p["name"] for p in plan["phases"]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(n in (".", "..") or Path(n).name != n for n in names)
    ):
        raise ValueError("Curriculum phase names must be unique directory names")
    if phases and (set(phases) - set(names)):
        raise ValueError("Unknown curriculum phases")
    if phases and phases != names[: len(phases)]:
        raise ValueError("Run a prefix of the ordered plan, or the full plan")
    selected = plan["phases"][: len(phases)] if phases else plan["phases"]
    profiles = {
        phase["name"]: config_load(path.parent / phase["profile"], device, seed)
        for phase in plan["phases"]
    }
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        plan=plan,
        seed=seed,
        episodes=episodes,
        minimum_success=minimum_success,
        source_fingerprint=source_fingerprint(),
        initial_checkpoint_sha256=digest(init_from),
        profile_fingerprints={name: fingerprint(c) for name, c in profiles.items()},
    )
    manifest_path = root / "curriculum_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Curriculum plan, source or gate configuration changed; use a new output root"
        )
    write_json(manifest_path, manifest)
    parent, recent = Path(init_from).resolve() if init_from else None, None
    rows, sentinels = [], []
    for index, phase in enumerate(selected):
        c = copy.deepcopy(profiles[phase["name"]])
        if c["task"]["name"] != phase["name"]:
            raise ValueError("Phase name and task profile disagree")
        if steps is not None:
            c["training"]["physics_steps"] = steps
        validate(c)
        minimum_steps = (
            2 * c["num_envs"] * round(c["task"]["duration"] / c["physics"]["dt"])
        )
        if c["training"]["physics_steps"] < minimum_steps:
            raise ValueError(
                f"{phase['name']}: budget below two full deadlines per environment; use train for explicit smoke tests"
            )
        directory = root / phase["name"] / f"seed_{seed}"
        latest = directory / "checkpoints/latest.pt"
        skip_file = directory / "phase_skip.json"
        screening = None
        if skip_file.exists():
            state = json.loads(skip_file.read_text())
            comparison = copy.deepcopy(c)
            comparison["training"]["physics_steps"] = 0
            if state["parent_sha256"] != digest(parent) or state[
                "config_fingerprint"
            ] != fingerprint(comparison):
                raise ValueError(
                    "Skipped phase configuration/parent changed; use a new output root"
                )
            best = Path(parent)
            status = "skipped_already_mastered"
        elif latest.exists():
            _, _, saved = load(latest, device)
            actual = (
                saved["progress"]["parent"]["sha256"]
                if saved["progress"]["parent"]
                else None
            )
            if actual != digest(parent):
                raise ValueError(
                    "Parent checkpoint changed: use a new curriculum output root"
                )
            best = directory / "checkpoints/best_current.pt"
            status = "trained"
        else:
            if parent is not None and plan.get("prescreen", True):
                model, _, _ = load(parent, device)
                screening = gate(
                    model,
                    c,
                    sentinels,
                    root / "screening",
                    episodes,
                    seed,
                    minimum_success,
                    plan,
                    phase["name"],
                    fail_fast=True,
                )
            if screening is not None and screening["passed"]:
                comparison = copy.deepcopy(c)
                comparison["training"]["physics_steps"] = 0
                write_json(directory / "config.json", c)
                write_json(
                    skip_file,
                    dict(
                        parent_sha256=digest(parent),
                        config_fingerprint=fingerprint(comparison),
                        selected_checkpoint=str(parent),
                    ),
                )
                best = Path(parent)
                status = "skipped_already_mastered"
            else:
                best = directory / "checkpoints/best_current.pt"
                status = "trained"
        if status != "skipped_already_mastered":
            anchor = (
                sentinels[(seed + index) % len(sentinels)]["config"]
                if sentinels
                else None
            )
            replay = dict(recent=recent, anchor=anchor) if recent or anchor else None

            def promotion_check(model, current_config, progress):
                if not plan.get("early_promotion", True) or (
                    progress["ppo_updates"] > 0
                    and progress["physics_steps"] < minimum_steps
                ):
                    return False
                report = gate(
                    model,
                    current_config,
                    sentinels,
                    directory,
                    episodes,
                    seed,
                    minimum_success,
                    plan,
                    f"promotion_{progress['ppo_updates']:05d}",
                    fail_fast=True,
                )
                return report["passed"]

            train(
                c,
                directory,
                resume=latest if latest.exists() else None,
                init_from=parent if not latest.exists() else None,
                replay=replay,
                promotion_check=promotion_check,
            )
        feasible_path = directory / "checkpoints/best_feasible.pt"
        if status != "skipped_already_mastered" and feasible_path.exists():
            best = feasible_path
        model, _, saved = load(best, device)
        result = (
            screening
            if status == "skipped_already_mastered" and screening
            else gate(
                model,
                c,
                sentinels,
                directory,
                episodes,
                seed,
                minimum_success,
                plan,
                "gate",
            )
        )
        origin = saved["origin"]
        if status == "trained":
            status = (
                "improved_by_training"
                if origin == "ppo"
                and saved["progress"]["best_feasible_improved_over_initial"]
                else (
                    "passed_after_ppo"
                    if origin == "ppo"
                    else "passed_with_initialization"
                )
            )
        local_updates = (
            saved["progress"]["ppo_updates"] if best.parent.parent == directory else 0
        )
        row = dict(
            phase=phase["name"],
            profile=str(path.parent / phase["profile"]),
            passed=result["passed"],
            status=status if result["passed"] else "failed_gate",
            best_origin=origin,
            selected_local_ppo_updates=local_updates,
            selected_checkpoint=str(best),
            selected_checkpoint_sha256=digest(best),
            current_success_rate=min(r["success_rate"] for r in result["current"]),
            current=result["current"],
            retention=result["retention"],
        )
        rows.append(row)
        write_json(
            root / "curriculum_summary.json",
            dict(
                schema=SCHEMA,
                seed=seed,
                phases=rows,
                note="Skipped phases reuse an unchanged checkpoint. Gate/MC hold-out seeds are validation data; reserve fresh final-test seeds.",
            ),
        )
        if not result["passed"]:
            details = dict(
                event="curriculum_gate_failed",
                phase=phase["name"],
                success_rate=row["current_success_rate"],
                minimum_success=minimum_success,
                retention=result["retention"],
                report=str(directory / "gate/summary.json"),
                checkpoint=str(best),
                message=f"{phase['name']} failed current-task or retained-skill/precision gates; results preserved",
            )
            write_json(directory / "gate_failure.json", details)
            raise ExperimentGateError(details)
        if phase.get("retention_anchor", False):
            reference = []
            for offset in plan.get("retention_seed_offsets", [89000, 189000]):
                report, _ = evaluate(model, c, episodes, offset + seed)
                if report["mean_tail_position_rmse"] is None:
                    raise RuntimeError(
                        "Retained-skill anchor requires complete tail observations"
                    )
                write_json(directory / f"anchor_reference_{offset + seed}.json", report)
                reference.append(
                    dict(
                        seed=offset + seed, tail_rmse=report["mean_tail_position_rmse"]
                    )
                )
            sentinels.append(dict(name=phase["name"], config=c, reference=reference))
        recent = c
        parent = best
    return rows
