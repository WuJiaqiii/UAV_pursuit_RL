"""Run matched PPO-only hover cold-start experiments with the CTBR curriculum."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    "random-original": (False, False),
    "zero-original": (True, False),
    "random-local": (False, True),
    "zero-local": (True, True),
}
PHASES = (
    ("00_hover_equilibrium", 3, [0, 0], 0, 0.0, 0.15, 2_000_000),
    ("01_hover_tilt", 5, [0, 0], 1, 0.0, 0.20, 4_000_000),
    ("02_hover_drift", 8, [0, 0.15], 2, 0.05, 0.25, 6_000_000),
    ("03_hover_small", 20, [0, 0.3], 5, 0.1, 0.10, 20_000_000),
)


def write_config(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Existing configuration differs: {path}. Use a new --out.")
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare(out, cell, seed, device, through):
    out.mkdir(parents=True, exist_ok=True)
    names = [phase[0] for phase in PHASES]
    zero, local = CELLS[cell]
    write_config(out / "experiment.json", dict(cell=cell, seed=seed, device=device))
    profiles = out / "profiles"
    profiles.mkdir(exist_ok=True)
    plan = profiles / "plan.json"
    write_config(plan, dict(
        schema="pursuit-direct-ctbr-curriculum-v4",
        prescreen=True,
        early_promotion=True,
        gate_seed_offsets=[97000, 197000],
        phases=[dict(name=n, profile=f"{n}.json", retention_anchor=n == names[0]) for n in names],
    ))
    shared = json.loads((ROOT / "configs/ctbr/01_hover_small.json").read_text())
    reward = dict(progress=0.125, completion_bonus=6.0)
    if local:
        # Tune the existing dense reward around the reference point.
        reward.update(position_length=0.5, velocity_length=0.5, acquisition_bonus=0.0)
    for name, duration, dist, tilt, noise, rmse, steps in PHASES:
        task = dict(
            shared["task"], name=name, objective="sustained", duration=duration,
            distance=dist, initial_tilt_deg=tilt, initial_velocity_noise=noise,
            maximum_tail_rmse=rmse, tail_seconds=duration / 2,
        )
        write_config(profiles / f"{name}.json", dict(
            task=task, reward=reward, num_envs=32,
            training=dict(
                physics_steps=steps, bc_updates=0, dagger_rounds=0,
                transfer_dagger_updates=0, zero_initial_actor_mean=zero,
                prediction_updates_enabled=True, minimum_success=0.90,
                validation_seed=78000 + seed, validation_every=500_000,
                validation_episodes=64, save_every_updates=8,
            ),
        ))
    return [
        sys.executable, "-u", "main.py", "curriculum", "--plan", str(plan),
        "--out", str(out / "training"), "--seed", str(seed), "--device", device,
        "--episodes", "32", "--minimum-success", "0.90", "--phases",
        *names[:through],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=CELLS, default="zero-local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--through", type=int, choices=range(1, len(PHASES) + 1), default=len(PHASES))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not (out / "experiment.json").is_file():
        parser.error("--out contains unrelated files")
    command = prepare(out, args.cell, args.seed, args.device, args.through)
    print(f"Experiment: {out}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.prepare_only:
        raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
