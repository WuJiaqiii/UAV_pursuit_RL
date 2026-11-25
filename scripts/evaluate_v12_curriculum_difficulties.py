"""Evaluate the final pure-PPO curriculum and scratch control on all 13 levels.

Read-only with respect to training checkpoints/campaign; produces an independent
JSON + CSV matrix. Uses matched heldout episode seeds per task and checkpoint.
Does NOT claim that these extreme-speed tasks were in PPO training.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint
from pursuit_rl.ctbr.checkpoint import source_fingerprint
from pursuit_rl.ctbr.config import load as load_config
from pursuit_rl.ctbr.evaluation import evaluate
from scripts.run_pure_ppo import TASKS
from scripts import run_pure_ppo_campaign as base


def tier(name):
    return (
        "static" if name == "static" else
        "low" if name in ("tail", "head_on", "random") else
        name.split("_", 1)[0]
    )


def evaluate_checkpoints(root, profiles, episodes=32, seed=970001,
                         device="cuda", include_all_stages=False):
    manifest = base._read_json(root / "campaign_manifest.json")
    if manifest.get("schema") != "v12-pure-ppo-curriculum-v1":
        raise ValueError("Directory is not a V12 staged curriculum")
    if manifest.get("source_fingerprint") != source_fingerprint():
        raise ValueError("Source changed since curriculum: evaluate with matching branch")
    ordered = [entry for entry in manifest["runs"]
               if include_all_stages or entry["stage"] ==
               manifest["runs"][-1]["stage"]]
    rows = []
    for item in ordered:
        checkpoint = root / item["id"] / "checkpoints/best_current.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, trained, saved = load_checkpoint(checkpoint, device=device)
        if (trained["prediction"]["enabled"] or
                trained["safety"]["constraint_enabled"] or
                trained["training"]["bc_updates"] != 0):
            raise ValueError("Checkpoint is not from the requested pure PPO pipeline")
        for idx, profile in enumerate(profiles):
            cfg = copy.deepcopy(trained)
            cfg["task"] = copy.deepcopy(load_config(TASKS[profile])["task"])
            report, _ = evaluate(
                model, cfg, episodes=episodes, seed=seed + idx * 10007
            )
            failures = report["failures"]
            rows.append(dict(
                arm=item["arm"],
                stage=item["stage"],
                stage_index=item.get("stage_index", None),
                seed=item["seed"],
                trained_task=item["task"],
                task=profile,
                difficulty=tier(profile),
                episodes=episodes,
                successes=report["successes"],
                success_rate=report["success_rate"],
                tilt_failure_rate=report["tilt_failure_rate"],
                timeout_rate=failures.get("timeout", 0) / episodes,
                unsafe_entry_rate=report["unsafe_entry_rate"],
                maximum_tilt_p95_deg=report["maximum_tilt_p95_deg"],
                mean_minimum_distance_m=report["mean_minimum_distance_m"],
                checkpoint=str(checkpoint.resolve()),
                eval_seed=seed + idx * 10007,
            ))
    # Unweighted macro-average across evaluation tasks within tier;
    # independent task episodes are fixed-size, so pooling is also possible.
    tiers = []
    for arm in sorted(set(x["arm"] for x in rows)):
        for stage in sorted(set(x["stage"] for x in rows if x["arm"] == arm)):
            for group in ("static", "low", "medium", "high", "extreme"):
                selected = [x for x in rows
                            if x["arm"] == arm and x["stage"] == stage
                            and x["difficulty"] == group]
                if not selected:
                    continue
                tiers.append(dict(
                    arm=arm, stage=stage, difficulty=group,
                    scenarios=len(selected),
                    episodes=sum(x["episodes"] for x in selected),
                    successes=sum(x["successes"] for x in selected),
                    success_rate=sum(x["successes"] for x in selected) /
                        sum(x["episodes"] for x in selected),
                ))
    return dict(
        schema="v12-curriculum-13-profile-matrix-v1",
        source_campaign=str(root),
        eval_seed=seed,
        episodes_per_profile=episodes,
        profiles=profiles,
        rows=rows,
        by_difficulty=tiers,
        note="Independent profile tests; higher-speed tasks are OOD unless explicitly trained.",
    )


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=970001)
    parser.add_argument("--profiles", nargs="+", choices=list(TASKS),
                        default=list(TASKS))
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        parser.error("Episodes must be positive; seed must be nonnegative")
    root = Path(args.campaign).resolve()
    out = Path(args.out).resolve() if args.out else root / "extended_difficulty_matrix.json"
    if out.exists():
        parser.error("Output exists; choose another report path")
    result = evaluate_checkpoints(
        root, args.profiles, args.episodes, args.seed,
        args.device, args.all_stages
    )
    base._atomic_json(out, result)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = list(result["rows"][0]) if result["rows"] else []
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["rows"])
    print(json.dumps(dict(
        json=str(out), csv=str(csv_path),
        by_difficulty=result["by_difficulty"],
    ), ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
