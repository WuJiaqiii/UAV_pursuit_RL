"""Minimal, reproducible direct-CTBR PPO: no LSTM, teacher, DAgger or safety critic.

Run independent static/low-speed experiments without the 28-run V11 campaign.
An --init-from checkpoint must itself be a compatible pure-PPO checkpoint;
it only transfers network weights, never supplies demonstration actions.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, write_json
from pursuit_rl.ctbr.config import load as load_config, validate
from pursuit_rl.ctbr.training import train


TASKS = {
    "near_static": "configs/first_entry_highspeed/14_v13_near_static.json",
    "static": "configs/first_entry_highspeed/01_static.json",
    "tail": "configs/first_entry_highspeed/02_low_tail.json",
    "head_on": "configs/first_entry_highspeed/03_low_head_on.json",
    "random": "configs/first_entry_highspeed/04_low_random.json",
    "medium_tail": "configs/first_entry_highspeed/05_medium_tail.json",
    "medium_head_on": "configs/first_entry_highspeed/06_medium_head_on.json",
    "medium_random": "configs/first_entry_highspeed/07_medium_random.json",
    "high_tail": "configs/first_entry_highspeed/08_high_tail.json",
    "high_head_on": "configs/first_entry_highspeed/09_high_head_on.json",
    "high_random": "configs/first_entry_highspeed/10_high_random.json",
    "extreme_tail": "configs/first_entry_highspeed/11_extreme_tail.json",
    "extreme_head_on": "configs/first_entry_highspeed/12_extreme_head_on.json",
    "extreme_random": "configs/first_entry_highspeed/13_extreme_random.json",
}
PRESETS = {
    "smoke": {"physics_steps": 100_000, "validation_every": 100_000, "validation_episodes": 8},
    "quick": {"physics_steps": 500_000, "validation_every": 250_000, "validation_episodes": 12},
    "pilot": {"physics_steps": 2_000_000, "validation_every": 500_000, "validation_episodes": 24},
    "standard": {"physics_steps": 5_000_000, "validation_every": 1_000_000, "validation_episodes": 32},
}


def make_config(args):
    if args.task not in TASKS:
        raise ValueError("Unknown first-entry task profile")
    c = load_config(TASKS[args.task], device=args.device, seed=args.seed)
    preset = PRESETS[args.preset]
    c["num_envs"] = args.num_envs if args.num_envs is not None else 64
    if getattr(args, "pursuit_v13", False):
        # V13 is explicit opt-in so the old V12 experimental baselines remain
        # numerically and semantically unchanged.
        c["policy"]["relative_position_scale_m"] = 5.0
        c["policy"]["relative_velocity_scale_mps"] = 10.0
        c["policy"]["relative_position_encoding"] = "asinh"
        c["policy"]["relative_frame"] = "body"
        c["reward"]["progress_mode"] = "initial_normalized"
        c["training"]["selection_mode"] = "distance_first"
    c["policy"]["history_steps"] = (
        args.history_steps if args.history_steps is not None else 1
    )
    c["policy"]["initial_std"] = 0.15
    c["policy"]["min_std"] = 0.02
    c["policy"]["max_std"] = 0.3
    # No temporal predictor, feature network or supervised predictor fit.
    c["prediction"]["enabled"] = False
    c["safety"]["predictive_enabled"] = False
    c["safety"]["constraint_enabled"] = False
    c["safety"]["critical_reset_fraction"] = 0.0
    c["reward"]["actuator_risk"] = 0.0
    c["reward"]["tilt_guard"] = 0.0
    c["training"].update({
        "bc_updates": 0, "bc_rounds": 1, "aux_pd_coef": 0.0,
        "prediction_updates_enabled": False,
        "critic_updates": 0,
        "mc_calibration_every_updates": 0,
        "mc_episodes": 4,
        "mc_calibration_updates": 1,
        "replay_recent_fraction": 0.0,
        "replay_anchor_fraction": 0.0,
        "zero_initial_actor_mean": True,
        "actor_lr": 1e-4,
        "critic_lr": 3e-4,
        "batch_size": args.batch_size if args.batch_size is not None else 512,
        "epochs": args.epochs if args.epochs is not None else 2,
        "rollout_steps": (
            args.rollout_steps if args.rollout_steps is not None else 128
        ),
        "save_every_updates": 100,
        "checkpoint_every_updates": 10,
        "fast_terminal_records": not args.legacy_terminal_records,
        "skip_mc_diagnostics": True,
        "physics_steps": (
            args.physics_steps
            if args.physics_steps is not None
            else preset["physics_steps"]
        ),
        "validation_every": preset["validation_every"],
        "validation_episodes": preset["validation_episodes"],
        "initialization_success_floor": 0.0,
    })
    c["training"]["validation_every"] = min(
        c["training"]["validation_every"],
        c["training"]["physics_steps"],
    )
    return validate(c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=tuple(TASKS))
    ap.add_argument("--preset", choices=tuple(PRESETS), default="quick")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int)
    ap.add_argument("--rollout-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--history-steps", type=int)
    ap.add_argument("--physics-steps", type=int)
    ap.add_argument("--init-from")
    ap.add_argument("--resume")
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--legacy-terminal-records", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--pursuit-v13", action="store_true",
        help="Use direction-sensitive state, distance-normalized progress, "
             "and pursuit-first checkpoint ranking.",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable live progress display")
    args = ap.parse_args(argv)
    if args.init_from and args.resume:
        ap.error("--init-from and --resume are mutually exclusive")
    torch.set_num_threads(args.cpu_threads)
    if args.resume:
        if any(v is not None for v in (
            args.num_envs, args.rollout_steps,
            args.batch_size, args.epochs, args.history_steps,
            args.pursuit_v13,
        )) or args.legacy_terminal_records:
            ap.error("Resume locks task, parallelism and algorithm hyperparameters")
        _, c, _ = load_checkpoint(args.resume, device=args.device)
        if args.task is not None:
            expected = load_config(TASKS[args.task], device=args.device)["task"]["name"]
            if c["task"]["name"] != expected:
                raise ValueError("Resumed checkpoint task does not match --task")
        if args.physics_steps is not None:
            c["training"]["physics_steps"] = args.physics_steps
        c = validate(c)
        for path, target in [
            ("prediction.enabled", c["prediction"]["enabled"]),
            ("training.bc_updates", c["training"]["bc_updates"] != 0),
            ("training.aux_pd_coef", c["training"]["aux_pd_coef"] != 0),
            ("safety.predictive_enabled", c["safety"]["predictive_enabled"]),
            ("safety.constraint_enabled", c["safety"]["constraint_enabled"]),
        ]:
            if target:
                raise ValueError("Not a pure PPO checkpoint: " + path)
    else:
        c = make_config(args)
    if args.init_from:
        _, pc, _ = load_checkpoint(args.init_from, device=args.device)
        if (pc["prediction"]["enabled"] or pc["safety"]["predictive_enabled"]
                or pc["safety"]["constraint_enabled"]
                or pc["training"]["bc_updates"] > 0
                or pc["training"]["aux_pd_coef"] > 0):
            raise ValueError("--init-from requires a pure-PPO, no-predictor checkpoint")
    if args.dry_run:
        print(json.dumps(dict(
            task=c["task"]["name"], device=c["device"], seed=c["seed"],
            num_envs=c["num_envs"], policy=c["policy"],
            prediction_enabled=c["prediction"]["enabled"],
            safety_enabled=c["safety"]["predictive_enabled"],
            bc_updates=c["training"]["bc_updates"],
            steps=c["training"]["physics_steps"],
            rollout_steps=c["training"]["rollout_steps"],
            minibatch=c["training"]["batch_size"],
            epochs=c["training"]["epochs"],
            fast_terminal_records=c["training"]["fast_terminal_records"],
            selection_mode=c["training"]["selection_mode"],
            relative_position_encoding=c["policy"]["relative_position_encoding"],
            progress_mode=c["reward"]["progress_mode"],
        ), indent=2, ensure_ascii=False))
        return None
    root = Path(args.out)
    start = time.perf_counter()
    result = train(
        c, root, resume=args.resume, init_from=args.init_from,
        replay=None, promotion_check=None, run_manifest=None,
        bc_profiles=None, distillation=None, validation_profiles=None,
        show_progress=not args.no_progress,
    )
    elapsed = time.perf_counter() - start
    best = root / "checkpoints/best_current.pt"
    latest = root / "checkpoints/latest.pt"
    output = dict(
        schema="pure-ppo-fast-baseline-v1",
        task=c["task"]["name"],
        preset="resume" if args.resume else args.preset,
        seed=c["seed"],
        wall_time_seconds=elapsed,
        ppo_physics_steps=result["physics_steps"],
        total_interactions=result["total_local_interactions"],
        ppo_physics_steps_per_wall_second=(
            result["physics_steps"] / max(elapsed, 1e-9)
        ),
        bc_updates=0, predictor_enabled=False,
        safety_critic_enabled=False, pd_aux_coef=0,
        num_envs=c["num_envs"],
        rollout_steps=c["training"]["rollout_steps"],
        batch_size=c["training"]["batch_size"],
        epochs=c["training"]["epochs"],
        checkpoint=str(best if best.exists() else latest),
        notes="No MC critic prefit or validation; Critic is optimized only in PPO. No teacher, prediction network or Safety Critic.",
    )
    write_json(root / "pure_ppo_summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    main()
