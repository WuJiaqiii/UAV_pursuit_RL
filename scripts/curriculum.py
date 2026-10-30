#!/usr/bin/env python3
"""Sequential task transfer with checkpoint lineage and baseline regression gates."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pursuit_rl.config.rescue import load_rescue_profile
from pursuit_rl.training.rescue import write_json


def execute(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf8") as sink:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)


def select_phases(plan, names):
    phases = plan["phases"]
    known = {phase["name"] for phase in phases}
    if names:
        unknown = set(names) - known
        if unknown:
            raise ValueError("Unknown curriculum phases: " + str(sorted(unknown)))
        # A named subset is an explicit user selection; preserve plan order.
        phases = [phase for phase in phases if phase["name"] in names]
    if not phases:
        raise ValueError("Empty curriculum")
    return phases


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", default=str(project / "configs/curriculum_v681/plan.json")
    )
    parser.add_argument(
        "--baseline-root", default=str(project / "baseline_models/velocity_50hz_2m")
    )
    parser.add_argument("--out", default=str(project / "runs/v681_curriculum"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--phases", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physics-steps", type=int, default=2000000)
    parser.add_argument("--regression-episodes", type=int, default=64)
    parser.add_argument("--regression-seed", type=int, default=79000)
    parser.add_argument("--minimum-success", type=float, default=0.95)
    parser.add_argument("--max-regression-time-ratio", type=float, default=1.25)
    args = parser.parse_args(argv)
    if (
        args.physics_steps < 0
        or not 0 <= args.minimum_success <= 1
        or args.regression_episodes < 1
        or args.max_regression_time_ratio < 1
    ):
        parser.error("Invalid budget or success gate")
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "pursuit-curriculum-plan-v1":
        parser.error("Unknown curriculum plan")
    phases = select_phases(plan, args.phases)
    all_summaries = []
    for seed in args.seeds:
        parent = (
            Path(args.baseline_root).resolve() / f"seed_{seed}" / "checkpoints/best.pt"
        )
        if not parent.is_file():
            raise FileNotFoundError(f"Missing matching-seed baseline: {parent}")
        anchor_dir = Path(args.out).resolve() / f"baseline_anchor_seed_{seed}"
        execute(
            [
                sys.executable,
                str(project / "scripts/challenge.py"),
                "--checkpoint",
                str(parent),
                "--device",
                args.device,
                "--out",
                str(anchor_dir),
                "--scenarios",
                "baseline",
                "--episodes",
                str(args.regression_episodes),
                "--seeds",
                str(args.regression_seed),
            ],
            anchor_dir / "console.log",
        )
        anchor = json.loads((anchor_dir / "summary.json").read_text())["cases"][0][
            "learned"
        ]
        if anchor["success_rate"] < args.minimum_success:
            raise RuntimeError(
                "Baseline anchor missed validation gate on this evaluation set"
            )
        for phase in phases:
            profile_path = (plan_path.parent / phase["profile"]).resolve()
            directory = Path(args.out).resolve() / phase["name"] / f"seed_{seed}"
            latest = directory / "checkpoints/latest.pt"
            common = [
                sys.executable,
                str(project / "scripts/rescue.py"),
                "train",
                "--profile",
                str(profile_path),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--out",
                str(directory),
            ]
            if not latest.is_file():
                execute(
                    common + ["--physics-steps", "0", "--init-from", str(parent)],
                    directory / "initialization_console.log",
                )
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            expected = hashlib.sha256(parent.read_bytes()).hexdigest()
            lineage = saved["progress"].get("parent_checkpoint", {})
            if lineage.get("sha256") != expected:
                raise ValueError(
                    "Existing phase uses a different parent; use a new output root to change curriculum lineage"
                )
            if saved["progress"]["physics_steps"] < args.physics_steps:
                execute(
                    common
                    + [
                        "--physics-steps",
                        str(args.physics_steps),
                        "--resume",
                        str(latest),
                    ],
                    directory / f"ppo_console_{args.physics_steps}.log",
                )
            summary = json.loads((directory / "training_summary.json").read_text())
            validations = list(directory.glob("validation_*.json"))
            validation = (
                max(validations, key=lambda p: int(p.stem.split("_")[-1]))
                if validations
                else directory / "initial_validation.json"
            )
            score = json.loads(validation.read_text())
            if score["success_rate"] < args.minimum_success:
                raise RuntimeError(
                    f"Current-task validation missed gate at {directory}; results retained"
                )
            best = directory / "checkpoints/best.pt"
            execute(
                [
                    sys.executable,
                    str(project / "scripts/challenge.py"),
                    "--checkpoint",
                    str(best),
                    "--device",
                    args.device,
                    "--out",
                    str(directory / "baseline_regression"),
                    "--scenarios",
                    "baseline",
                    "--episodes",
                    str(args.regression_episodes),
                    "--seeds",
                    str(args.regression_seed),
                ],
                directory / "regression_console.log",
            )
            regression = json.loads(
                (directory / "baseline_regression/summary.json").read_text()
            )["cases"][0]["learned"]
            if regression["success_rate"] < args.minimum_success:
                raise RuntimeError(
                    f"Original-task regression missed gate at {directory}; do not continue increasing difficulty"
                )
            ratio = regression["mean_capture_time_timeout_penalty"] / max(
                anchor["mean_capture_time_timeout_penalty"], 1e-8
            )
            if ratio > args.max_regression_time_ratio:
                raise RuntimeError(
                    f"Original-task capture time regressed by factor {ratio:.3f}; stop curriculum at {directory}"
                )
            all_summaries.append(
                {
                    "phase": phase["name"],
                    "training_seed": seed,
                    "summary": summary,
                    "validation_success": score["success_rate"],
                    "baseline_regression": regression,
                    "baseline_time_ratio": ratio,
                    "next_parent": str(best),
                }
            )
            Path(args.out).resolve().mkdir(parents=True, exist_ok=True)
            write_json(
                Path(args.out).resolve() / "curriculum_summary.json",
                {"schema": "pursuit-curriculum-result-v1", "phases": all_summaries},
            )
            parent = best
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
