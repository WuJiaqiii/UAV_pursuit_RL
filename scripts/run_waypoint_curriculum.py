"""Run the CTBR curriculum against stationary waypoints and preplanned survey arcs."""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.survey_route_profiles import PHASES, route_task

GATES = (3290042, 3390042)
HELD_OUT = (3490042, 3590042)
EPISODES = 32
MINIMUM_PASS = 29


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    from pursuit_rl.ctbr.checkpoint import write_json as save_json

    return save_json(path, value)


def config(profile, name, device, seed):
    from pursuit_rl.ctbr.config import load as load_config

    return load_config(profile / f"{name}.json", device, seed)


def prepare(out, device, seed):
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(device=device, seed=seed, phases=PHASES, gates=GATES, held_out=HELD_OUT)
    manifest = out / "manifest.json"
    if manifest.exists() and read(manifest) != json.loads(json.dumps(spec)):
        raise ValueError("Experiment configuration changed. Use a new output directory.")
    write_json(manifest, spec)
    profile = out / "profiles"
    for name in PHASES:
        write_json(profile / f"{name}.json", dict(
            task=route_task(name),
            training=dict(bc_updates=0, dagger_rounds=0, transfer_dagger_updates=0,
                          validation_seed=seed + 3100000, validation_episodes=32,
                          validation_every=250000, mc_episodes=8, critic_updates=96,
                          prediction_updates_enabled=False, save_every_updates=4),
        ))
    return profile


def evaluate_saved(model, c, seed, path, reference, traces):
    if path.exists():
        return read(path)
    from pursuit_rl.ctbr.evaluation import evaluate

    report, _ = evaluate(model, c, EPISODES, seed, reference=reference,
                         trace_dir=path.parent / str(seed) if traces else None,
                         trace_count=8)
    write_json(path, report)
    return report


def summarize(report):
    episodes = report["episodes_detail"]
    complete = report["full_tail_episodes"]
    joint = round(report["joint_precision_success_rate"] * report["episodes"])
    return dict(
        seed=report["seed"], episodes=report["episodes"],
        full_flights=complete, successes=report["successes"],
        precision_successes=joint, passed=joint >= MINIMUM_PASS,
        mean_position_rmse_m=report["mean_position_rmse"],
        mean_tail_position_rmse_m=report["mean_tail_position_rmse"],
        mean_tail_height_rmse_m=(report["mean_tail_axis_rmse_m"][2] if complete else None),
        mean_action_saturation_fraction=statistics.mean(
            row["action_saturation_fraction"] for row in episodes
        ),
        failures=report["failures"],
        initial_state_digest=report["initial_state_digest"],
    )


def assessments(c, checkpoint, root, seeds, expected=None, traces=False):
    from pursuit_rl.ctbr.checkpoint import load as load_checkpoint

    model = load_checkpoint(checkpoint, c["device"])[0] if checkpoint else None
    results = []
    for seed in seeds:
        report = evaluate_saved(model, c, seed, root / f"{seed}.json", checkpoint is None, traces)
        brief = summarize(report)
        if expected and brief["initial_state_digest"] != expected[seed]:
            raise RuntimeError(f"Initial states differ for {root}, seed {seed}")
        results.append(brief)
    write_json(root / "summary.json", results)
    print(root, [(r["precision_successes"], r["mean_position_rmse_m"]) for r in results],
          flush=True)
    return results


def passed(reports):
    return all(row["passed"] for row in reports)


def score(reports):
    return (sum(r["precision_successes"] for r in reports),
            -statistics.mean(r["mean_position_rmse_m"] for r in reports))


def fit(c, directory, steps, parent=None, bc=False):
    summary = directory / "training_summary.json"
    if not summary.exists():
        from pursuit_rl.ctbr.training import train

        c["training"]["physics_steps"] = steps
        if bc:
            c["training"].update(bc_samples=48000, bc_updates=2000, bc_segment_seconds=2,
                                 initialization_success_floor=0)
        latest = directory / "checkpoints/latest.pt"
        if latest.exists():
            train(c, directory, resume=latest)
        else:
            train(c, directory, init_from=parent)
    return directory / "checkpoints"


def evaluate_heldout(profile, out, names, candidates, device, seed):
    for name in names:
        c = config(profile, name, device, seed)
        ref = assessments(c, None, out / "heldout" / name / "geometry", HELD_OUT, traces=True)
        expected = {row["seed"]: row["initial_state_digest"] for row in ref}
        for label, checkpoint in candidates.items():
            assessments(c, checkpoint, out / "heldout" / name / label, HELD_OUT,
                        expected=expected, traces=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "ppo-compare":
        from scripts.route_ppo_compare import main as compare_main

        return compare_main(argv[1:])

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use ppo-compare --help for the paired PPO experiment.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    args.out = args.out.resolve()
    profile = prepare(args.out, args.device, args.seed)
    selected = None
    rows = []
    first_candidates = {}
    for index, name in enumerate(PHASES):
        phase = args.out / name
        c = config(profile, name, args.device, args.seed)
        ref = assessments(c, None, phase / "geometry", GATES)
        expected = {r["seed"]: r["initial_state_digest"] for r in ref}
        if not passed(ref):
            rows.append(dict(phase=name, status="geometry_failed", geometry=ref))
            break
        candidates = {}
        if selected:
            candidates["transfer"] = selected
        else:
            candidates["bc"] = fit(config(profile, name, args.device, args.seed),
                                   phase / "bc_train", 0, bc=True) / "initial.pt"
        label, parent = next(iter(candidates.items()))
        reports = {label: assessments(c, parent, phase / label, GATES, expected)}
        if not index or not passed(reports[label]):
            paths = fit(config(profile, name, args.device, args.seed),
                        phase / "ppo_train", 500000, parent=parent)
            candidates["ppo_final"] = paths / "latest.pt"
            reports["ppo_final"] = assessments(c, candidates["ppo_final"],
                                                phase / "ppo_final", GATES, expected)
        if not index:
            first_candidates = dict(candidates)
        good = [label for label in candidates if passed(reports[label])]
        retained = {}
        for label in sorted(good, key=lambda value: score(reports[value]), reverse=True):
            if index and label != "transfer":
                anchor_name = next(iter(PHASES))
                ret = assessments(config(profile, anchor_name, args.device, args.seed),
                                  candidates[label], phase / f"{label}_retention", GATES)
                retained[label] = ret
                if not passed(ret):
                    continue
            selected = candidates[label]
            rows.append(dict(phase=name, status="passed", selected=str(selected),
                             selected_label=label, geometry=ref, reports=reports,
                             retention=retained))
            break
        else:
            rows.append(dict(phase=name, status="learned_gate_failed", geometry=ref,
                             reports=reports, retention=retained))
            break
        write_json(args.out / "curriculum_summary.json", rows)
    write_json(args.out / "curriculum_summary.json", rows)

    names = [row["phase"] for row in rows if row["status"] == "passed"]
    if first_candidates:
        evaluate_heldout(profile, args.out, [next(iter(PHASES))], first_candidates,
                         args.device, args.seed)
    if names:
        evaluate_heldout(profile, args.out, names, {"promoted": selected},
                         args.device, args.seed)
    return 0 if len(names) == len(PHASES) else 2


if __name__ == "__main__":
    sys.exit(main())
