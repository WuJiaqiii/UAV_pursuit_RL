"""Independent medium/high/extreme first-entry stress evaluations for v10.

Never trains or promotes checkpoints. Uses the 05-13 legacy task geometries
with the selected V10 Actor and observation-matched pure-PD baselines.
"""
import argparse
import copy
import csv
import hashlib
import json
import traceback
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, source_fingerprint, write_json
from pursuit_rl.ctbr.config import load as load_profile, validate

if __package__:
    from .run_generalization_campaign import aggregate, evaluate_methods
else:
    from run_generalization_campaign import aggregate, evaluate_methods


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stress_configs(parent, plan, groups, seed, device):
    """Use historical task definitions with the exact trained physics contract.

    Only task parameters change. Training mixture, Actor dimensions, thrust,
    control frequency and first-entry safety thresholds remain unchanged.
    """
    task_paths = plan["stress_profiles"]
    specs = plan["stress_groups"]
    configs = {}
    seen = set()
    for group in groups:
        configs[group] = []
        for task_name in specs[group]:
            if task_name in seen:
                raise ValueError(f"Duplicate stress task: {task_name}")
            seen.add(task_name)
            path = task_paths[task_name]
            source = load_profile(path, device=device, seed=seed)
            if source["task"]["name"] != task_name:
                raise ValueError(f"Stress profile name/path mismatch: {task_name}")
            config = copy.deepcopy(parent)
            config["task"] = copy.deepcopy(source["task"])
            config["device"] = device
            config["seed"] = seed
            config["num_envs"] = plan["num_envs"]
            config["training"]["teacher_method"] = plan["teacher_method"]
            config = validate(config)
            for key in ("radius", "entry_relative_speed_max_mps"):
                if config["task"][key] != parent["task"][key]:
                    raise ValueError(f"Stress test changes first-entry contract: {key}")
            configs[group].append(config)
    return configs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-results", type=Path, required=True,
                   help="Completed V10 campaign_results.json")
    p.add_argument("--plan", type=Path,
                   default=Path("configs/experiments/06_v10_pd_assisted_ctbr.json"))
    p.add_argument("--parent-checkpoint", type=Path,
                   help="Overrides the parent path in campaign_manifest.json")
    p.add_argument("--stage", choices=("pilot", "full", "confirmation"),
                   default="confirmation")
    p.add_argument("--groups", nargs="+",
                   help="One or more of: medium, high, extreme")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--methods", nargs="+",
                   help="Subset of ppo_v9, ppo_v8, pn_ctbr, pd_ctbr, pd_observed, pd_observed_smoothed, pn_pd_ctbr")
    p.add_argument("--episodes", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--out", type=Path,
                   help="Separate output directory; defaults beside campaign_results")
    p.add_argument("--resume-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def to_csv(results, path):
    path = Path(path)
    fields = [
        "stage", "variant", "group", "task", "method",
        "successes", "episodes", "success_rate",
        "wilson_95_lower", "wilson_95_upper",
        "delta_success_vs_pd_observed",
        "delta_success_vs_pd_observed_smoothed",
        "delta_success_vs_pd_ctbr", "unsafe_entry_rate",
        "mean_entry_time_s", "mean_entry_relative_speed_mps",
        "mean_minimum_distance_m", "mean_action_saturation_fraction",
        "per_seed_success_rates", "failures",
    ]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, groups in results.get("summary_by_variant", {}).items():
            for group, tasks in groups.items():
                for task, methods in tasks.items():
                    observed = methods.get("pd_observed", {}).get("success_rate")
                    smoothed = methods.get("pd_observed_smoothed", {}).get("success_rate")
                    privileged = methods.get("pd_ctbr", {}).get("success_rate")
                    for method, row in methods.items():
                        success = row["success_rate"]
                        writer.writerow({
                            "stage": results["stage"], "variant": variant,
                            "group": group, "task": task, "method": method,
                            "successes": row["successes"], "episodes": row["episodes"],
                            "success_rate": success,
                            "wilson_95_lower": row["wilson_95"][0],
                            "wilson_95_upper": row["wilson_95"][1],
                            "delta_success_vs_pd_observed": (
                                success-observed if observed is not None else None
                            ),
                            "delta_success_vs_pd_observed_smoothed": (
                                success-smoothed if smoothed is not None else None
                            ),
                            "delta_success_vs_pd_ctbr": (
                                success-privileged if privileged is not None else None
                            ),
                            "unsafe_entry_rate": row.get("unsafe_entry_rate"),
                            "mean_entry_time_s": row.get("mean_entry_time_s"),
                            "mean_entry_relative_speed_mps": row.get("mean_entry_relative_speed_mps"),
                            "mean_minimum_distance_m": row.get("mean_minimum_distance_m"),
                            "mean_action_saturation_fraction": row.get("mean_action_saturation_fraction"),
                            "per_seed_success_rates": json.dumps(row["per_seed_success_rates"]),
                            "failures": json.dumps(row["failures"]),
                        })
    return path


def run(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if "stress_profiles" not in plan or "stress_groups" not in plan:
        raise ValueError("Selected experiment plan has no high-speed stress tests")
    groups = args.groups or list(plan["stress_groups"])
    if not groups or len(groups) != len(set(groups)) or (
        set(groups) - set(plan["stress_groups"])
    ):
        raise ValueError("Unknown or duplicated stress group")
    methods = args.methods or plan["stress_evaluation"]["default_methods"]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Controller methods must be nonempty and distinct")
    valid_methods = {
        "ppo_v9", "ppo_v8", "pn_ctbr", "pd_ctbr",
        "pd_observed", "pd_observed_smoothed", "pn_pd_ctbr",
    }
    if set(methods) - valid_methods:
        raise ValueError(f"Unknown controller: {sorted(set(methods)-valid_methods)}")
    episodes = args.episodes or plan["stress_evaluation"]["default_episodes"]
    if episodes < 1:
        raise ValueError("Nonpositive stress evaluation episode count")

    campaign_path = args.campaign_results.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf8"))
    stage = campaign["stages"].get(args.stage)
    if stage is None:
        raise ValueError(f"Campaign stage not found: {args.stage}")
    variant_rows = stage["variants"]
    if args.variants is not None and set(args.variants)-set(variant_rows):
        raise ValueError("Unknown campaign variant selection")
    names = args.variants or list(variant_rows)
    selected = []
    for name in names:
        records = variant_rows[name]
        missing = set(args.seeds or []) - {
            row["seed"] for row in records if row.get("status") == "completed"
        }
        if missing:
            raise ValueError(f"Missing completed seeds for {name}: {sorted(missing)}")
        for row in records:
            if row.get("status") != "completed":
                continue
            if args.seeds is not None and row["seed"] not in args.seeds:
                continue
            checkpoint = Path(row["checkpoint"]).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
            selected.append(dict(variant=name, seed=row["seed"],
                                 checkpoint=str(checkpoint),
                                 sha256=sha256(checkpoint)))
    if not selected:
        raise ValueError("No completed policy checkpoints match the selection")
    parent_path = args.parent_checkpoint
    if parent_path is None:
        manifest_path = campaign_path.with_name("campaign_manifest.json")
        original = json.loads(manifest_path.read_text(encoding="utf8"))
        parent_path = Path(original["parent_checkpoint"])
    parent_path = parent_path.resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint missing: {parent_path}")
    config = dict(
        stage=args.stage, groups=groups, methods=methods, episodes=episodes,
        selected=selected, parent_sha256=sha256(parent_path),
        plan_sha256=sha256(plan_path),
        script_sha256=sha256(__file__),
        source_fingerprint=source_fingerprint(),
    )
    if args.dry_run:
        print(json.dumps({"planned": config}, ensure_ascii=False, indent=2))
        return {"planned": config}

    out = (args.out or campaign_path.parent / "highspeed_stress").resolve()
    manifest_path = out / "stress_manifest.json"
    if manifest_path.exists():
        if not args.resume_existing:
            raise FileExistsError("Stress test exists; supply --resume-existing")
        existing = json.loads(manifest_path.read_text(encoding="utf8"))
        if existing != config:
            raise ValueError("Stress manifest changed; use another output directory")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError("Stress output directory is nonempty without manifest")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, config)

    progress = out / "stress_results.json"
    results = json.loads(progress.read_text(encoding="utf8")) if progress.exists() else {
        "schema": "pursuit-v10-highspeed-stress-v1",
        "stage": args.stage, "runs": {}, "errors": [],
    }
    parent_model, parent_config, _ = load_checkpoint(parent_path, args.device)
    for item in selected:
        variant, seed = item["variant"], item["seed"]
        run = results["runs"].setdefault(variant, {}).setdefault(str(seed), {})
        pending_groups = [group for group in groups
                          if run.get(group, {}).get("status") != "completed"]
        if not pending_groups:
            continue
        model, _, _ = load_checkpoint(item["checkpoint"], args.device)
        configs = stress_configs(parent_config, plan, pending_groups, seed, args.device)
        for group in pending_groups:
            destination = out / args.stage / variant / f"seed_{seed}" / group
            try:
                group_offset = list(plan["stress_groups"]).index(group) * 100000
                reports = evaluate_methods(
                    model, parent_model, configs[group], episodes,
                    seed + plan["evaluation_seed_offset"] + 900000 + group_offset,
                    destination, methods=methods,
                )
                run[group] = {
                    "status": "completed", "checkpoint": item["checkpoint"],
                    "checkpoint_sha256": item["sha256"],
                    "episodes_per_method_task": episodes,
                    "evaluations": reports,
                }
            except Exception as exc:
                run[group] = {
                    "status": "failed", "error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc(),
                }
                results["errors"].append({
                    "variant": variant, "seed": seed, "group": group,
                    "error": str(exc),
                })
                write_json(progress, results)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            write_json(progress, results)
    results["summary_by_variant"] = {
        variant: {
            group: aggregate([
                seed_groups[group]["evaluations"]
                for seed_groups in results["runs"].get(variant, {}).values()
                if seed_groups.get(group, {}).get("status") == "completed"
            ])
            for group in groups
        }
        for variant in names
    }
    write_json(progress, results)
    to_csv(results, out / "stress_summary.csv")
    return results


if __name__ == "__main__":
    run()
