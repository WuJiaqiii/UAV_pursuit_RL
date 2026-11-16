"""Adapt the single-aircraft survey controller with mixed route demonstrations."""

import argparse
from pathlib import Path

import torch

from run_waypoint_curriculum import (
    GATES, HELD_OUT, PHASES, assessments, config, passed, prepare, read, write_json,
)
from pursuit_rl.ctbr.checkpoint import load, save
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import fit_actor, imitation_data


def inspect_gates(profile, out, model, device, seed):
    rows = []
    for name in PHASES:
        c = config(profile, name, device, seed)
        geometry = assessments(c, None, out / name / "geometry", GATES)
        expected = {row["seed"]: row["initial_state_digest"] for row in geometry}
        learned = assessments(c, model, out / name / "mixed_bc", GATES, expected)
        rows.append(dict(phase=name, passed=passed(geometry) and passed(learned),
                         geometry=geometry, learned=learned))
        write_json(out / "curriculum_summary.json", rows)
        if not rows[-1]["passed"]:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out = args.out.resolve()
    profile = prepare(args.out, args.device, 42)
    checkpoint = args.out / "mixed_bc.pt"
    if not checkpoint.is_file():
        torch.manual_seed(42)
        model, _, saved = load(args.base, args.device)
        data = []
        receipt = []
        for index, (name, samples) in enumerate(
            (("01_waypoint", 20000), ("02_survey_arc", 32000),
             ("04_offset_start", 24000))
        ):
            c = config(profile, name, args.device, 42)
            c["training"].update(bc_samples=samples, bc_segment_seconds=2)
            x, y, physics = imitation_data(c, seed=4010042 + index * 1000)
            data.append((x, y))
            receipt.append(dict(task=name, samples=samples, physics_steps=physics))
        loss = fit_actor(model, tuple(torch.cat(parts) for parts in zip(*data)), 2400, 2048)
        c = config(profile, "02_survey_arc", args.device, 42)
        progress = dict(saved["progress"], bc_updates=saved["progress"]["bc_updates"] + 2400,
                        accepted_bc_updates=saved["progress"]["accepted_bc_updates"] + 2400,
                        imitation_physics_steps=saved["progress"]["imitation_physics_steps"]
                        + sum(row["physics_steps"] for row in receipt),
                        critic_needs_calibration=True, route_demonstrations=receipt)
        save(checkpoint, model, c, progress, "mixed_route_bc")
        write_json(args.out / "demonstrations.json", dict(loss=loss, sources=receipt))
        print(f"Mixed demonstration loss: {loss:.6f}", flush=True)

    rows = inspect_gates(profile, args.out, checkpoint, args.device, 42)
    if len(rows) != len(PHASES) or not all(row["passed"] for row in rows):
        return 2
    independent = []
    for name in PHASES:
        c = config(profile, name, args.device, 42)
        dest = args.out / "heldout" / name
        geometry = assessments(c, None, dest / "geometry", HELD_OUT, traces=True)
        expected = {row["seed"]: row["initial_state_digest"] for row in geometry}
        learned = assessments(c, checkpoint, dest / "mixed_bc", HELD_OUT, expected, traces=True)
        independent.append(dict(phase=name, passed=passed(learned), geometry=geometry, learned=learned))
        write_json(args.out / "heldout" / "summary.json", independent)
    return 0 if all(row["passed"] for row in independent) else 2


if __name__ == "__main__":
    raise SystemExit(main())
