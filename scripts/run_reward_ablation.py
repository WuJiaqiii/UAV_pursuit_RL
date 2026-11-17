"""Run matched reward-shaping ablations for the 02_low_tail first-entry task.

Example:
  python scripts/run_reward_ablation.py \
    --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
    --out runs/low_tail_reward_ablation --stage screening --device cuda

Each variant warm-starts from the same parent Actor, optionally performs
current-task PD DAgger/BC, and trains with the same 80/20 current/replay mix.
Hard first-entry and physical termination rules are never ablated.
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, write_json
from pursuit_rl.ctbr.config import load, validate
from pursuit_rl.ctbr.errors import ExperimentGateError
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import train


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
    return p


def build_configs(plan, variant, stage_name, seed, device):
    base = load(plan["base_profile"], device=device, seed=seed)
    base["reward"].update(copy.deepcopy(plan["base_reward"]))
    base["reward"].update(copy.deepcopy(variant.get("reward", {})))
    base["training"].update(copy.deepcopy(plan["training_defaults"]))
    base["training"].update(copy.deepcopy(plan["stages"][stage_name]["training"]))
    base["training"]["bc_updates"] = int(
        variant.get("bc_updates", plan["training_defaults"]["bc_updates"])
    )
    base["training"]["teacher_method"] = plan["teacher_method"]
    base["training"]["initialization_success_floor"] = (
        plan["initialization_success_floor"]
        if base["training"]["bc_updates"] > 0
        else 0.0
    )
    base["training"]["replay_recent_fraction"] = plan["replay"]["recent_fraction"]
    base["training"]["replay_anchor_fraction"] = 0.0
    base["training"]["prediction_updates_enabled"] = False
    validate(base)

    replay = load(plan["replay_profile"], device=device, seed=seed)
    replay["num_envs"] = base["num_envs"]
    replay["reward"] = copy.deepcopy(base["reward"])
    validate(replay)
    return base, replay


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
    if plan.get("schema") != "pursuit-first-entry-reward-ablation-v1":
        raise ValueError("Unsupported reward-ablation plan schema")
    parent = Path(args.parent_checkpoint).resolve()
    if not parent.is_file():
        raise FileNotFoundError(f"Parent checkpoint not found: {parent}")
    variants = {item["name"]: item for item in plan["variants"]}
    selected = args.variants or list(variants)
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError(f"Unknown reward-ablation variants: {unknown}")
    stage = plan["stages"][args.stage]
    seeds = args.seeds or stage["seeds"]
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    matrix = {
        "schema": plan["schema"],
        "stage": args.stage,
        "parent_checkpoint": str(parent),
        "parent_sha256": __import__("hashlib").sha256(parent.read_bytes()).hexdigest(),
        "results": [],
        "note": (
            "Screening results rank hypotheses only. Promotion requires the "
            "independent final evaluations and retained-skill checks."
        ),
    }

    for name in selected:
        variant = variants[name]
        for seed in seeds:
            config, replay = build_configs(plan, variant, args.stage, seed, args.device)
            run_dir = root / name / f"seed_{seed}"
            print(json.dumps({
                "event": "ablation_start",
                "variant": name,
                "seed": seed,
                "physics_steps": config["training"]["physics_steps"],
                "bc_updates": config["training"]["bc_updates"],
                "reward": config["reward"],
                "task_mix": {
                    config["task"]["name"]: 1.0 - config["training"]["replay_recent_fraction"],
                    replay["task"]["name"]: config["training"]["replay_recent_fraction"],
                },
            }, ensure_ascii=False), flush=True)
            if args.dry_run:
                matrix["results"].append({
                    "variant": name, "seed": seed, "status": "dry_run",
                    "config": config,
                })
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite existing run: {run_dir}"
                )
            try:
                training_summary = train(
                    config,
                    run_dir,
                    init_from=str(parent),
                    replay={"recent": replay, "anchor": None},
                )
                checkpoint = run_dir / "checkpoints/best_current.pt"
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
                    final_reports.append({
                        "seed": eval_seed,
                        "success_rate": report["success_rate"],
                        "unsafe_entry_rate": report["unsafe_entry_rate"],
                        "mean_minimum_distance_m": report["mean_minimum_distance_m"],
                        "mean_entry_relative_speed_mps": report[
                            "mean_entry_relative_speed_mps"
                        ],
                        "failures": report["failures"],
                        "passed": passed,
                    })
                row = {
                    "variant": name,
                    "seed": seed,
                    "status": "trained",
                    "physics_steps": training_summary["physics_steps"],
                    "ppo_updates": training_summary["ppo_updates"],
                    "bc_updates": training_summary["bc_updates"],
                    "imitation_physics_steps": training_summary[
                        "imitation_physics_steps"
                    ],
                    "task_environment_counts": training_summary[
                        "task_environment_counts"
                    ],
                    "final_evaluations": final_reports,
                    "final_gate_passed": final_passed,
                    "checkpoint": str(checkpoint),
                }
            except ExperimentGateError as exc:
                failure_path = run_dir / "gate_failure.json"
                failure = (
                    json.loads(failure_path.read_text())
                    if failure_path.exists()
                    else exc.details
                )
                row = {
                    "variant": name,
                    "seed": seed,
                    "status": "bc_initialization_gate_failed",
                    "gate_failure": failure,
                }
            matrix["results"].append(row)
            write_json(root / "ablation_matrix.json", matrix)
            print(json.dumps({
                "event": "ablation_complete",
                "variant": name,
                "seed": seed,
                "status": row["status"],
                "final_gate_passed": row.get("final_gate_passed"),
            }, ensure_ascii=False), flush=True)

    write_json(root / "ablation_matrix.json", matrix)
    return matrix


if __name__ == "__main__":
    main()
