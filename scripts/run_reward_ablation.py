"""Run matched first-entry reward ablations with auditable resume support.

The PPO physics-step budget excludes BC/DAgger, MC critic fitting and evaluation
interactions. The default experiment mode records BC quality but does not gate
PPO on the BC success rate; use --enforce-initialization-gate for strict runs.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from pursuit_rl.ctbr.checkpoint import (
    load as load_checkpoint,
    source_fingerprint,
    write_json,
)
from pursuit_rl.ctbr.config import fingerprint, load, validate
from pursuit_rl.ctbr.errors import ExperimentGateError
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import train


SUPPORTED_SCHEMAS = {
    "pursuit-first-entry-reward-ablation-v1",
    "pursuit-first-entry-reward-factorial-v2",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan",
        default="configs/experiments/02_low_tail_reward_ablation.json",
    )
    p.add_argument("--parent-checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=("screening", "full"), default="screening")
    p.add_argument("--variants", nargs="*")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--enforce-initialization-gate",
        action="store_true",
        help="Abort PPO if BC initialization misses the configured success floor.",
    )
    p.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume incomplete run directories from checkpoints/latest.pt.",
    )
    return p


def git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass
    return os.environ.get("GIT_COMMIT", "unknown")


def build_configs(
    plan, variant, stage_name, seed, device, enforce_initialization_gate=False
):
    base = load(plan["base_profile"], device=device, seed=seed)
    base["reward"].update(copy.deepcopy(plan["base_reward"]))
    base["reward"].update(copy.deepcopy(variant.get("reward", {})))
    base["training"].update(copy.deepcopy(plan["training_defaults"]))
    base["training"].update(copy.deepcopy(plan["stages"][stage_name]["training"]))
    base["training"]["bc_updates"] = int(
        variant.get("bc_updates", plan["training_defaults"]["bc_updates"])
    )
    base["training"]["bc_rounds"] = int(
        variant.get(
            "bc_rounds",
            plan["training_defaults"].get("bc_rounds", 1),
        )
    )
    if base["training"]["bc_updates"] == 0:
        base["training"]["bc_rounds"] = 1
    base["training"]["teacher_method"] = plan["teacher_method"]
    base["training"]["initialization_success_floor"] = (
        plan["initialization_success_floor"]
        if enforce_initialization_gate and base["training"]["bc_updates"] > 0
        else 0.0
    )
    base["training"]["replay_recent_fraction"] = plan["replay"]["recent_fraction"]
    base["training"]["replay_anchor_fraction"] = 0.0
    base["training"]["prediction_updates_enabled"] = False
    validate(base)

    replay = load(plan["replay_profile"], device=device, seed=seed)
    replay["num_envs"] = base["num_envs"]
    replay["reward"] = copy.deepcopy(base["reward"])
    for key in ("discount_time_s", "advantage_time_s"):
        replay["training"][key] = base["training"][key]
    validate(replay)
    return base, replay


def _upsert_result(matrix, row):
    matrix["results"] = [
        item
        for item in matrix["results"]
        if not (
            item.get("variant") == row.get("variant")
            and item.get("seed") == row.get("seed")
        )
    ]
    matrix["results"].append(row)


def _task_mix(config, replay):
    replay_envs = round(
        config["num_envs"] * config["training"]["replay_recent_fraction"]
    )
    return {
        "current_fraction": 1.0 - config["training"]["replay_recent_fraction"],
        "replay_fraction": config["training"]["replay_recent_fraction"],
        "current_envs": config["num_envs"] - replay_envs,
        "replay_envs": replay_envs,
        "current_task": config["task"]["name"],
        "replay_task": replay["task"]["name"],
    }


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if plan.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported experiment plan schema: {plan.get('schema')}")
    parent = Path(args.parent_checkpoint).resolve()
    if not parent.is_file():
        raise FileNotFoundError(f"Parent checkpoint not found: {parent}")
    parent_sha256 = hashlib.sha256(parent.read_bytes()).hexdigest()
    commit_sha = git_revision()
    variants = {item["name"]: item for item in plan["variants"]}
    default_key = f"default_{args.stage}_variants"
    selected = args.variants or plan.get(default_key, list(variants))
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError(f"Unknown reward-ablation variants: {unknown}")
    stage = plan["stages"][args.stage]
    seeds = args.seeds or stage["seeds"]
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    matrix_path = root / "ablation_matrix.json"
    if matrix_path.exists():
        if not args.resume_existing:
            raise FileExistsError(
                f"Experiment matrix already exists; use --resume-existing: {matrix_path}"
            )
        matrix = json.loads(matrix_path.read_text(encoding="utf8"))
        if matrix.get("stage") != args.stage or matrix.get("parent_sha256") != parent_sha256:
            raise ValueError("Existing matrix stage or parent checkpoint does not match")
    else:
        matrix = {
            "schema": plan["schema"],
            "stage": args.stage,
            "plan": str(plan_path),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "parent_checkpoint": str(parent),
            "parent_sha256": parent_sha256,
            "commit_sha": commit_sha,
            "source_fingerprint": source_fingerprint(),
            "results": [],
            "budget_definition": (
                "physics_steps counts PPO rollout physics interactions only; "
                "BC/DAgger, MC critic fitting, validation and final evaluation "
                "interactions are reported separately."
            ),
            "note": (
                "Screening ranks hypotheses only. Promotion requires independent "
                "current-task evaluations and retained-static-task checks."
            ),
        }
        write_json(matrix_path, matrix)

    for name in selected:
        variant = variants[name]
        for seed in seeds:
            config, replay = build_configs(
                plan,
                variant,
                args.stage,
                seed,
                args.device,
                enforce_initialization_gate=args.enforce_initialization_gate,
            )
            run_dir = root / name / f"seed_{seed}"
            mix = _task_mix(config, replay)
            config_hash = fingerprint(config)
            manifest = {
                "schema": "pursuit-reward-ablation-run-manifest-v2",
                "variant": name,
                "seed": seed,
                "stage": args.stage,
                "commit_sha": commit_sha,
                "source_fingerprint": source_fingerprint(),
                "config_fingerprint": config_hash,
                "parent_checkpoint": str(parent),
                "parent_sha256": parent_sha256,
                "ppo_physics_steps_budget": config["training"]["physics_steps"],
                "bc_updates_total": config["training"]["bc_updates"],
                "bc_rounds": config["training"]["bc_rounds"],
                "initialization_gate_enforced": args.enforce_initialization_gate,
                "initialization_success_floor": config["training"]["initialization_success_floor"],
                "task_mix": mix,
                "reward": copy.deepcopy(config["reward"]),
            }
            print(
                json.dumps(
                    {
                        "event": "ablation_start",
                        "variant": name,
                        "seed": seed,
                        "ppo_physics_steps_budget": config["training"]["physics_steps"],
                        "bc_updates_total": config["training"]["bc_updates"],
                        "bc_rounds": config["training"]["bc_rounds"],
                        "reward": config["reward"],
                        "task_mix": mix,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.dry_run:
                _upsert_result(
                    matrix,
                    {
                        "variant": name,
                        "seed": seed,
                        "status": "dry_run",
                        "config_fingerprint": config_hash,
                        "ppo_physics_steps_budget": config["training"]["physics_steps"],
                        "bc_updates_total": config["training"]["bc_updates"],
                        "bc_rounds": config["training"]["bc_rounds"],
                        "task_mix": mix,
                    },
                )
                write_json(matrix_path, matrix)
                continue

            resume_path = None
            if run_dir.exists() and any(run_dir.iterdir()):
                if not args.resume_existing:
                    raise FileExistsError(
                        f"Refusing to overwrite existing run: {run_dir}"
                    )
                latest = run_dir / "checkpoints" / "latest.pt"
                manifest_path = run_dir / "run_manifest.json"
                if not latest.is_file() or not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Cannot resume without latest checkpoint and manifest: {run_dir}"
                    )
                old_manifest = json.loads(manifest_path.read_text(encoding="utf8"))
                for key in ("config_fingerprint", "parent_sha256", "source_fingerprint"):
                    if old_manifest.get(key) != manifest.get(key):
                        raise ValueError(
                            f"Resume manifest mismatch for {key}: {run_dir}"
                        )
                resume_path = latest
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "run_manifest.json", manifest)

            try:
                training_summary = train(
                    config,
                    run_dir,
                    resume=str(resume_path) if resume_path else None,
                    init_from=None if resume_path else str(parent),
                    replay={"recent": replay, "anchor": None},
                )
                ppo_complete = (
                    training_summary["physics_steps"]
                    >= config["training"]["physics_steps"]
                )
                row = {
                    "variant": name,
                    "seed": seed,
                    "status": "trained" if ppo_complete else "incomplete",
                    "ppo_complete": ppo_complete,
                    "config_fingerprint": config_hash,
                    "ppo_physics_steps_budget": config["training"]["physics_steps"],
                    "physics_steps": training_summary["physics_steps"],
                    "decision_transitions": training_summary["decision_transitions"],
                    "ppo_updates": training_summary["ppo_updates"],
                    "bc_updates": training_summary["bc_updates"],
                    "bc_rounds_completed": training_summary["bc_rounds_completed"],
                    "imitation_physics_steps": training_summary["imitation_physics_steps"],
                    "mc_physics_steps": training_summary["mc_physics_steps"],
                    "validation_physics_steps": training_summary["validation_physics_steps"],
                    "total_local_interactions": training_summary["total_local_interactions"],
                    "task_environment_counts": training_summary["task_environment_counts"],
                    "stop_reason": training_summary["stop_reason"],
                    "checkpoint": str(run_dir / "checkpoints" / "best_current.pt"),
                    "initialization_report": str(run_dir / "initialization_bc.json"),
                }
                if ppo_complete:
                    checkpoint = run_dir / "checkpoints" / "best_current.pt"
                    model, saved_config, saved = load_checkpoint(checkpoint, args.device)
                    final_reports = []
                    final_passed = True
                    for offset in plan["final_gate_seed_offsets"]:
                        eval_seed = seed + offset
                        report, _ = evaluate(
                            model,
                            saved_config,
                            stage["final_eval_episodes"],
                            eval_seed,
                        )
                        write_json(
                            run_dir / "final" / f"evaluation_{eval_seed}.json",
                            report,
                        )
                        passed = (
                            report["success_rate"] >= plan["minimum_success"]
                            and report["unsafe_entry_rate"] == 0.0
                        )
                        final_passed &= passed
                        final_reports.append(
                            {
                                "seed": eval_seed,
                                "success_rate": report["success_rate"],
                                "unsafe_entry_rate": report["unsafe_entry_rate"],
                                "mean_minimum_distance_m": report["mean_minimum_distance_m"],
                                "mean_entry_relative_speed_mps": report["mean_entry_relative_speed_mps"],
                                "failures": report["failures"],
                                "passed": passed,
                            }
                        )
                    retention_reports = []
                    retention_passed = True
                    for offset in plan["retention_seed_offsets"]:
                        eval_seed = seed + offset
                        report, _ = evaluate(
                            model,
                            replay,
                            stage["final_eval_episodes"],
                            eval_seed,
                        )
                        write_json(
                            run_dir / "final" / f"retention_{eval_seed}.json",
                            report,
                        )
                        passed = (
                            report["success_rate"] >= plan["minimum_success"]
                            and report["unsafe_entry_rate"] == 0.0
                        )
                        retention_passed &= passed
                        retention_reports.append(
                            {
                                "seed": eval_seed,
                                "task": replay["task"]["name"],
                                "success_rate": report["success_rate"],
                                "unsafe_entry_rate": report["unsafe_entry_rate"],
                                "mean_minimum_distance_m": report["mean_minimum_distance_m"],
                                "mean_entry_relative_speed_mps": report["mean_entry_relative_speed_mps"],
                                "failures": report["failures"],
                                "passed": passed,
                            }
                        )
                    row.update(
                        final_evaluations=final_reports,
                        retention_evaluations=retention_reports,
                        retention_passed=retention_passed,
                        final_gate_passed=final_passed and retention_passed,
                    )
            except ExperimentGateError as exc:
                failure_path = run_dir / "gate_failure.json"
                failure = (
                    json.loads(failure_path.read_text(encoding="utf8"))
                    if failure_path.exists()
                    else exc.details
                )
                row = {
                    "variant": name,
                    "seed": seed,
                    "status": "bc_initialization_gate_failed",
                    "config_fingerprint": config_hash,
                    "gate_failure": failure,
                }
            except Exception as exc:
                failure = {
                    "event": "experiment_failed",
                    "variant": name,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "ppo_physics_steps_budget": config["training"]["physics_steps"],
                }
                write_json(run_dir / "failure.json", failure)
                row = {
                    "variant": name,
                    "seed": seed,
                    "status": "failed",
                    "config_fingerprint": config_hash,
                    "failure": failure,
                }
                _upsert_result(matrix, row)
                write_json(matrix_path, matrix)
                raise

            _upsert_result(matrix, row)
            write_json(matrix_path, matrix)
            print(
                json.dumps(
                    {
                        "event": "ablation_complete",
                        "variant": name,
                        "seed": seed,
                        "status": row["status"],
                        "ppo_complete": row.get("ppo_complete"),
                        "final_gate_passed": row.get("final_gate_passed"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_json(matrix_path, matrix)
    return matrix


if __name__ == "__main__":
    main()
