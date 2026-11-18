"""One-command, resumable four-task first-entry generalization campaign.

Use the v8 low-tail best checkpoint for initialization, never for on-policy
behavior mixing. Every learned policy still directly outputs four CTBR controls.
"""
import argparse
import copy
import hashlib
import json
import math
import traceback
from collections import Counter
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import (
    load as load_checkpoint, source_fingerprint, write_json,
)
from pursuit_rl.ctbr.config import load as load_profile, validate, fingerprint
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import train


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="configs/experiments/05_v9_generalization.json")
    ap.add_argument("--parent-checkpoint", required=True)
    ap.add_argument("--static-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("pilot", "full", "confirmation", "all"),
                    default="all")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume-existing", action="store_true")
    return ap.parse_args(argv)


def model_configs(parent_config, plan, variant, stage, seed, device):
    """Only task dynamics and training hyperparameters vary; fixed architecture."""
    configurations = {}
    current = plan["primary"]
    for name, path in plan["profiles"].items():
        task = load_profile(path, device=device, seed=seed)
        config = copy.deepcopy(parent_config)
        config["task"] = copy.deepcopy(task["task"])
        config["device"] = device
        config["seed"] = seed
        config["num_envs"] = plan["num_envs"]
        config["training"].update(copy.deepcopy(plan["training_defaults"]))
        config["training"].update(copy.deepcopy(plan["stages"][stage]["training"]))
        config["training"]["bc_updates"] = int(variant.get("bc_updates", 0))
        config["training"]["bc_rounds"] = int(variant.get("bc_rounds", 1))
        config["training"]["initialization_success_floor"] = 0.0
        config["training"]["prediction_updates_enabled"] = False
        config["training"]["replay_recent_fraction"] = 0.0
        config["training"]["replay_anchor_fraction"] = 0.0
        config["training"]["teacher_method"] = plan["teacher_method"]
        for key, value in variant.get("training", {}).items():
            if key not in ("actor_lr", "critic_lr", "advantage_time_s"):
                raise ValueError(f"Disallowed generalization training override: {key}")
            config["training"][key] = value
        for key, value in variant.get("reward", {}).items():
            if key not in ("progress", "braking", "entry_velocity"):
                raise ValueError(f"Disallowed generalization reward override: {key}")
            config["reward"][key] = value
        configurations[name] = validate(config)
    fractions = variant["mix"]
    if set(fractions) != set(configurations):
        raise ValueError("Mixture must allocate every named task explicitly")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Mixture fractions must sum to one")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("Every task needs a positive allocation")
    primary = configurations[current]
    others = [
        dict(config=configurations[name], fraction=fractions[name])
        for name in configurations if name != current
    ]
    return primary, others, list(configurations.values())


def heldout_configs(parent_config, plan, seed, device):
    """Never used for training or promotion; independent distribution-shift probes."""
    profiles = []
    for name, path in plan.get("heldout_profiles", {}).items():
        source = load_profile(path, device=device, seed=seed)
        config = copy.deepcopy(parent_config)
        config["task"] = copy.deepcopy(source["task"])
        config["task"]["name"] = name
        config["device"] = device
        config["seed"] = seed
        config["num_envs"] = plan["num_envs"]
        profiles.append(validate(config))
    return profiles


def evaluate_methods(model, parent_model, task_profiles, episodes, seed, dest):
    """Matched initial states and 4-CTBR physics across learned/geometric methods."""
    dest = Path(dest)
    reports = {}
    methods = {
        "ppo_v9": (model, None),
        "ppo_v8": (parent_model, None),
        "pn_ctbr": (None, "pn"),
        "pd_ctbr": (None, "pd"),
        "pn_pd_ctbr": (None, "pn_hybrid"),
    }
    for task_index, config in enumerate(task_profiles):
        task = config["task"]["name"]
        task_seed = seed + task_index * 10007
        reports[task] = {}
        digest = None
        for name, (policy, reference_method) in methods.items():
            if name == "ppo_v9" and model is None:
                continue
            case = copy.deepcopy(config)
            if reference_method:
                case["training"]["teacher_method"] = reference_method
            result, _ = evaluate(
                policy, case, episodes, task_seed,
                reference=reference_method is not None,
            )
            if digest is None:
                digest = result["initial_state_digest"]
            elif digest != result["initial_state_digest"]:
                raise RuntimeError(
                    f"Unmatched initial states in {task} at seed {task_seed}"
                )
            write_json(dest / f"{task}_{name}_{task_seed}.json", result)
            reports[task][name] = dict(
                seed=task_seed,
                episodes=result["episodes"],
                successes=result["successes"],
                success_rate=result["success_rate"],
                unsafe_entry_rate=result["unsafe_entry_rate"],
                failures=result["failures"],
                mean_entry_time_s=result["mean_entry_time_s"],
                mean_entry_relative_speed_mps=result["mean_entry_relative_speed_mps"],
                mean_minimum_distance_m=result["mean_minimum_distance_m"],
                mean_action_saturation_fraction=result["mean_action_saturation_fraction"],
                initial_state_digest=digest,
            )
    return reports


def aggregate(per_seed):
    """No mixed-training success counts; only independent, complete final tests."""
    names = set()
    for sample in per_seed:
        names.update(sample)
    result = {}
    for task in sorted(names):
        methods = sorted(set().union(*(r.get(task, {}) for r in per_seed)))
        result[task] = {}
        for method in methods:
            records = [r[task][method] for r in per_seed if method in r.get(task, {})]
            n = sum(x["episodes"] for x in records)
            k = sum(x["successes"] for x in records)
            p = k / n
            z = 1.959963984540054
            denominator = 1 + z*z/n
            center = (p + z*z/(2*n))/denominator
            half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
            failures = Counter()
            for item in records:
                failures.update(item["failures"])
            def average(key, count_key="episodes"):
                parts = [
                    (item.get(key), item[count_key]) for item in records
                    if item.get(key) is not None and item[count_key] > 0
                ]
                total_weight = sum(weight for _, weight in parts)
                return (
                    sum(value * weight for value, weight in parts) / total_weight
                    if total_weight else None
                )
            result[task][method] = dict(
                successes=k, episodes=n, success_rate=p,
                unsafe_entry_rate=average("unsafe_entry_rate"),
                mean_entry_time_s=average("mean_entry_time_s", "successes"),
                mean_entry_relative_speed_mps=average(
                    "mean_entry_relative_speed_mps", "successes"
                ),
                mean_minimum_distance_m=average("mean_minimum_distance_m"),
                mean_action_saturation_fraction=average(
                    "mean_action_saturation_fraction"
                ),
                failures=dict(failures),
                wilson_95=[
                    0.0 if k == 0 else max(0.0, center-half),
                    1.0 if k == n else min(1.0, center+half)
                ],
                per_seed_success_rates=[x["success_rate"] for x in records],
                paired_initial_conditions=True,
            )
    return result


def stage_candidates(stage, plan, reports):
    if stage == "pilot":
        return plan["stages"]["pilot"]["variants"]
    previous = "pilot" if stage == "full" else "full"
    if previous not in reports:
        raise ValueError(
            f"{stage} requires a completed {previous} phase in the same campaign"
        )
    results = reports[previous]["variants"]
    scores = []
    expected = len(plan["stages"][previous]["seeds"])
    for name, records in results.items():
        complete = [r for r in records if r.get("status") == "completed"]
        if len(complete) != expected:
            continue
        task_rates = {}
        for r in complete:
            for task, controls in r["evaluations"].items():
                task_rates.setdefault(task, []).append(controls["ppo_v9"]["success_rate"])
        worst = min(sum(rates)/len(rates) for rates in task_rates.values())
        mean = sum(sum(rates)/len(rates) for rates in task_rates.values())/len(task_rates)
        scores.append((worst, mean, name))
    scores.sort(reverse=True)
    minimum = plan["stages"][stage]["minimum_previous_worst"]
    eligible = [s for s in scores if s[0] >= minimum]
    return [name for _, _, name in eligible[:plan["stages"][stage]["top_k"]]]


def run(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    plan_file = Path(args.plan).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf8"))
    if plan["schema"] != "pursuit-v9-generalization-campaign-v1":
        raise ValueError("Invalid v9 campaign plan")
    parent_file = Path(args.parent_checkpoint).resolve()
    static_file = Path(args.static_checkpoint).resolve()
    parent, parent_config, _ = load_checkpoint(parent_file, args.device)
    # The best static teacher can have a different reward profile, but must
    # use the same dynamics, Actor observation, CTBR and model architecture.
    from pursuit_rl.ctbr.config import transfer_interface
    static_teacher, static_config, _ = load_checkpoint(static_file, args.device)
    if transfer_interface(static_config) != transfer_interface(parent_config):
        raise ValueError("Static teacher and tail parent have different interfaces")
    del static_teacher
    root = Path(args.out).resolve()
    manifest = dict(
        schema=plan["schema"],
        plan_sha256=checksum(plan_file),
        parent_sha256=checksum(parent_file),
        static_teacher_sha256=checksum(static_file),
        source_fingerprint=source_fingerprint(),
        plan=str(plan_file), parent_checkpoint=str(parent_file),
        static_checkpoint=str(static_file),
    )
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.exists():
        if not args.resume_existing:
            raise FileExistsError(
                "Existing campaign detected; add --resume-existing or choose new --out"
            )
        old = json.loads(manifest_path.read_text(encoding="utf8"))
        for key in ("schema", "plan_sha256", "parent_sha256",
                    "static_teacher_sha256", "source_fingerprint"):
            if old[key] != manifest[key]:
                raise ValueError(f"Campaign changed {key}; resume refused")
    elif root.exists() and any(
        item.name not in ("campaign.log", "campaign.pid") for item in root.iterdir()
    ):
        # The nohup launcher creates only its log and PID in the campaign
        # directory before invoking Python. Accept exactly these two benign
        # bookkeeping files, but refuse arbitrary previous experiment data.
        raise FileExistsError("Campaign root is nonempty without manifest")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
    if args.stage == "all" and (args.variants or args.seeds):
        raise ValueError(
            "--stage all uses pre-registered, matched seeds and automatic "
            "variant promotion; specify a single --stage for manual subsets"
        )
    stages = ["pilot", "full", "confirmation"] if args.stage == "all" else [args.stage]
    progress_path = root / "campaign_results.json"
    results = (
        json.loads(progress_path.read_text(encoding="utf8"))
        if progress_path.exists()
        else dict(schema=plan["schema"], stages={}, errors=[])
    )
    variants = {v["name"]: v for v in plan["variants"]}
    for stage in stages:
        candidates = args.variants or (
            stage_candidates(stage, plan, results["stages"]) if args.stage == "all"
            else plan["stages"][stage]["variants"]
        )
        if not candidates:
            results["stages"][stage] = dict(
                status="skipped_low_generalization",
                variants={}, reason="No candidate met previous-phase threshold",
            )
            write_json(progress_path, results)
            break
        unknown = set(candidates) - set(variants)
        if unknown:
            raise ValueError(f"Unknown campaign variants: {sorted(unknown)}")
        stage_result = results["stages"].setdefault(stage, dict(
            status="running", variants={},
        ))
        seeds = args.seeds if args.seeds else plan["stages"][stage]["seeds"]
        for name in candidates:
            v = variants[name]
            rows = stage_result["variants"].setdefault(name, [])
            for seed in seeds:
                existing = next((x for x in rows if x["seed"] == seed), None)
                if existing and existing.get("status") == "completed":
                    continue
                config, others, eval_profiles = model_configs(
                    parent_config, plan, v, stage, seed, args.device
                )
                run_dir = root / stage / name / f"seed_{seed}"
                mix = {config["task"]["name"]: v["mix"][plan["primary"]]}
                mix.update({
                    x["config"]["task"]["name"]: x["fraction"] for x in others
                })
                row = dict(
                    variant=name, seed=seed, mix=mix,
                    ppo_physics_steps_budget=config["training"]["physics_steps"],
                    status="planned",
                )
                if args.dry_run:
                    row["status"] = "dry_run"
                else:
                    if run_dir.exists() and any(run_dir.iterdir()):
                        if not args.resume_existing:
                            raise FileExistsError(
                                f"Run already exists: {run_dir}"
                            )
                        latest = run_dir / "checkpoints" / "latest.pt"
                        if (run_dir / "first_anomaly.pt").exists():
                            raise RuntimeError(
                                f"Numerically invalid run cannot be resumed: {run_dir}"
                            )
                        if not latest.exists():
                            if (run_dir / "training_summary.json").exists():
                                raise FileNotFoundError(
                                    f"Completed training has no latest.pt: {run_dir}"
                                )
                            # Interrupted before the first PPO checkpoint (for
                            # example in BC or Critic warmup): archive the entire
                            # attempt intact, then restart from the v8 parent.
                            # Never silently overwrite or delete experiment data.
                            count = 1
                            while True:
                                archive = run_dir.with_name(
                                    f"{run_dir.name}__precheckpoint_attempt_{count:02d}"
                                )
                                if not archive.exists():
                                    break
                                count += 1
                            run_dir.rename(archive)
                            print(
                                f"Archived pre-checkpoint run {archive}; restarting {run_dir}",
                                flush=True,
                            )
                            resume = None
                        else:
                            resume = str(latest)
                    else:
                        resume = None
                    distill_coef = float(v.get("distill_coef", 0.0))
                    teachers = {
                        "01_static": str(static_file),
                        "02_low_tail": str(parent_file),
                    } if distill_coef > 0 else {}
                    distillation = dict(
                        coefficient=distill_coef, teachers=teachers,
                    )
                    bc_profiles = (
                        eval_profiles if v.get("multitask_bc", False) else None
                    )
                    run_manifest = dict(
                        schema="v9-generalization-run-v1",
                        campaign_sha256=manifest["plan_sha256"],
                        parent_sha256=manifest["parent_sha256"],
                        static_teacher_sha256=manifest["static_teacher_sha256"],
                        config_fingerprint=fingerprint(config),
                        variant=v, mix=mix, seed=seed, stage=stage,
                    )
                    try:
                        summary = train(
                            config, run_dir,
                            init_from=None if resume else str(parent_file),
                            resume=resume,
                            replay={"tasks": others},
                            distillation=distillation,
                            bc_profiles=bc_profiles,
                            validation_profiles=eval_profiles,
                            run_manifest=None if resume else run_manifest,
                        )
                        row["physics_steps"] = summary["physics_steps"]
                        row["task_environment_counts"] = summary["task_environment_counts"]
                        if summary["physics_steps"] < config["training"]["physics_steps"]:
                            row["status"] = "incomplete"
                        else:
                            checkpoint = run_dir / "checkpoints" / "best_current.pt"
                            trained, _, _ = load_checkpoint(checkpoint, args.device)
                            reports = evaluate_methods(
                                trained, parent, eval_profiles,
                                plan["stages"][stage]["final_eval_episodes"],
                                seed + plan["evaluation_seed_offset"],
                                run_dir / "comparison",
                            )
                            if stage in ("full", "confirmation"):
                                holdouts = heldout_configs(
                                    config, plan, seed, args.device
                                )
                                ood = evaluate_methods(
                                    trained, parent, holdouts,
                                    plan["stages"][stage].get(
                                        "ood_eval_episodes",
                                        plan["stages"][stage]["final_eval_episodes"] // 2,
                                    ),
                                    seed + plan["evaluation_seed_offset"] + 500000,
                                    run_dir / "ood_comparison",
                                ) if holdouts else {}
                            else:
                                ood = {}
                            row.update(
                                status="completed",
                                checkpoint=str(checkpoint),
                                evaluations=reports,
                                heldout_evaluations=ood,
                                worst_task_success_rate=min(
                                    r["ppo_v9"]["success_rate"]
                                    for r in reports.values()
                                ),
                                macro_success_rate=sum(
                                    r["ppo_v9"]["success_rate"]
                                    for r in reports.values()
                                ) / len(reports),
                            )
                    except Exception as exc:
                        row.update(
                            status="failed",
                            error_type=type(exc).__name__, error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                        run_dir.mkdir(parents=True, exist_ok=True)
                        write_json(run_dir / "failure.json", row)
                        results["errors"].append({
                            "stage": stage, "variant": name,
                            "seed": seed, "error": str(exc),
                        })
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            write_json(progress_path, results)
                            raise
                rows[:] = [x for x in rows if x["seed"] != seed] + [row]
                write_json(progress_path, results)
        stage_result["status"] = (
            "completed" if all(
                r.get("status") in ("completed", "dry_run")
                for rows in stage_result["variants"].values() for r in rows
            ) else "completed_with_failures"
        )
        # Never pool episodes across different candidate policies; statistical
        # summaries are per variant and keep independent seeds separate.
        stage_result["summary_by_variant"] = {
            name: dict(
                in_distribution=aggregate([
                    row["evaluations"] for row in rows
                    if row.get("status") == "completed"
                ]),
                heldout=aggregate([
                    row["heldout_evaluations"] for row in rows
                    if row.get("status") == "completed"
                    and row.get("heldout_evaluations")
                ]),
            )
            for name, rows in stage_result["variants"].items()
        }
        write_json(progress_path, results)
        from scripts.summarize_v9_campaign import to_csv
        to_csv(results, root / "comparison_summary.csv")
        if args.dry_run:
            # Dry runs should not accidentally promote candidates on empty
            # performance estimates, nor modify a real training campaign.
            break
    return results


if __name__ == "__main__":
    run()
