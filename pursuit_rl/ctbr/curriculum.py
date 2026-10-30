"""Sequential direct-CTBR curriculum with source lineage and skill retention."""

import copy
import hashlib
import json
from pathlib import Path
from .config import load as config_load
from .checkpoint import load, write_json
from .training import train
from .evaluation import evaluate


def run(
    plan_path,
    out,
    device="cpu",
    seed=42,
    steps=None,
    episodes=32,
    minimum_success=0.9,
    phases=None,
):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    if plan.get("schema") != "pursuit-direct-ctbr-curriculum-v2":
        raise ValueError("Wrong curriculum schema")
    if not 0 <= minimum_success <= 1 or episodes < 1:
        raise ValueError("Invalid validation gate")
    names = [p["name"] for p in plan["phases"]]
    if phases and (set(phases) - set(names)):
        raise ValueError("Unknown curriculum phases")
    # A prefix only: an arbitrary subset could silently skip required skills.
    selected = plan["phases"]
    if phases:
        if phases != names[: len(phases)]:
            raise ValueError("Run a prefix of the ordered plan, or the full plan")
        selected = selected[: len(phases)]
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent = None
    rows = []
    sentinels = []
    for phase in selected:
        c = config_load(path.parent / phase["profile"], device, seed)
        if steps is not None:
            c["training"]["physics_steps"] = steps
        # 2 full deadlines of continuous PPO sampling minimum for a formal run.
        minimum_steps = (
            2 * c["num_envs"] * round(c["task"]["duration"] / c["physics"]["dt"])
        )
        if c["training"]["physics_steps"] < minimum_steps:
            raise ValueError(
                f"{phase['name']}: budget below two full deadlines per environment; use train for explicit smoke tests"
            )
        directory = root / phase["name"] / f"seed_{seed}"
        latest = directory / "checkpoints/latest.pt"
        if latest.exists():
            _, _, saved = load(latest, device)
            expected = (
                hashlib.sha256(Path(parent).read_bytes()).hexdigest()
                if parent
                else None
            )
            actual = (
                saved["progress"]["parent"]["sha256"]
                if saved["progress"]["parent"]
                else None
            )
            if actual != expected:
                raise ValueError(
                    "Parent checkpoint changed: use a new curriculum output root"
                )
            train(c, directory, resume=latest)
        else:
            train(c, directory, init_from=parent)
        best = directory / "checkpoints/best.pt"
        model, _, saved = load(best, device)
        current, _ = evaluate(model, c, episodes, 79000 + seed)
        write_json(directory / "gate_current.json", current)
        retained = []
        for previous in sentinels:
            old = config_load(path.parent / previous["profile"], device, seed)
            report, _ = evaluate(model, old, episodes, 89000 + seed)
            write_json(directory / "regression" / f"{previous['name']}.json", report)
            retained.append(
                dict(task=previous["name"], success_rate=report["success_rate"])
            )
        status = (
            "improved_by_training"
            if saved["origin"] == "ppo"
            else "passed_with_initialization"
        )
        passed = current["success_rate"] >= minimum_success and all(
            r["success_rate"] >= minimum_success for r in retained
        )
        row = dict(
            phase=phase["name"],
            passed=passed,
            status=status,
            best_origin=saved["origin"],
            selected_local_ppo_updates=saved["progress"]["ppo_updates"],
            selected_checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
            current_success_rate=current["success_rate"],
            retention=retained,
        )
        rows.append(row)
        write_json(
            root / "curriculum_summary.json",
            dict(
                schema=plan["schema"],
                seed=seed,
                phases=rows,
                note="Gate seeds are validation data; use fresh seeds for final tests. Passing with initialization is not PPO improvement.",
            ),
        )
        if not passed:
            raise RuntimeError(
                f"{phase['name']} failed current-task or retained-skill gate; results preserved"
            )
        if phase.get("retention_anchor", False):
            sentinels.append(phase)
        parent = best
    return rows
