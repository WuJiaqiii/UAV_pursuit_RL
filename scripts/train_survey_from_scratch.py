"""Train a new single-aircraft survey Actor with demonstration-based progression."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from adapt_random_survey import demonstrations
from evaluate_random_survey import Leg, configure, draw_pair, flight
from pursuit_rl.ctbr.checkpoint import load, new_model, save, source_fingerprint, write_json
from pursuit_rl.ctbr.config import load as load_config, validate
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import fit_actor, imitation_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "survey" / "scratch_bc_plan.json"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(plan, base):
    if plan.get("schema") != "single-aircraft-survey-bc-v1":
        raise ValueError("Expected the single-aircraft survey BC plan")
    phases = {}
    for phase in plan["phases"]:
        name = phase["name"]
        if name in phases or name in ("", ".", "..") or Path(name).name != name:
            raise ValueError(f"Invalid or duplicate phase name: {name}")
        if phase["samples"] <= 0 or phase["updates"] <= 0:
            raise ValueError(f"Each phase requires positive training budgets: {name}")
        if ("leg" in phase) == ("task" in phase):
            raise ValueError(f"Expected exactly one of task or leg: {name}")
        if "leg" in phase:
            leg = Leg(**phase["leg"])
            config = configure(base, name, leg)
        else:
            task = phase["task"]
            if task["kind"] not in ("hover", "goto"):
                raise ValueError(f"Expected a stationary navigation reference: {name}")
            leg = None
            config = dict(base, task={**base["task"], **task, "name": name,
                                      "objective": "sustained"})
        phases[name] = dict(phase, config=validate(config), leg=leg)
    episodes, required = plan["validation_episodes"], plan["minimum_qualified"]
    if not phases or not 1 <= required <= episodes:
        raise ValueError("Invalid phase list or qualification threshold")
    offsets = [*plan["gate_seed_offsets"], plan["retention_seed_offset"],
               *plan["test_seed_offsets"]]
    if len(plan["gate_seed_offsets"]) != 2 or len(plan["test_seed_offsets"]) != 2 \
            or len(set(offsets)) != len(offsets):
        raise ValueError("Use distinct gate, retention and test seeds")
    return phases


def collect(phase, seed):
    c = phase["config"]
    limit = phase["samples"]
    if phase["leg"] is None:
        profile = dict(c, training={**c["training"], "bc_samples": limit,
                                    "bc_segment_seconds": 2.0})
        x, y, steps = imitation_data(profile, seed=seed)
        return x[:, :21], y, steps
    states, actions, steps, samples = [], [], 0, 0
    while samples < limit:
        x, y, used = demonstrations(c, phase["leg"], seed + len(states))
        states.append(x)
        actions.append(y)
        samples += len(x)
        steps += used
    return torch.cat(states)[:limit], torch.cat(actions)[:limit], steps


def demo_data(path, phase, seed):
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data["states"], data["actions"], int(data["physics_steps"])
    x, y, steps = collect(phase, seed)
    x, y = x.cpu().numpy(), y.cpu().numpy()
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, states=x, actions=y, physics_steps=steps)
    temporary.replace(path)
    return x, y, steps


def assess(model, phase, seed, plan, out, render=False):
    c, leg = phase["config"], phase["leg"]
    n = plan["validation_episodes"]
    if leg is None:
        report, _ = evaluate(model, c, n, seed)
        count = round(report["joint_precision_success_rate"] * n)
        rmse = report["mean_tail_position_rmse"]
    else:
        report, time, track = flight(c, leg, seed, n, model)
        count = report["qualified"]
        values = [r["tail_position_rmse_m"] for r in report["episodes"]
                  if r["tail_position_rmse_m"] is not None]
        rmse = sum(values) / len(values) if values else None
        if render:
            geometry, gt, gp = flight(c, leg, seed, n)
            if (geometry["route_digest"], geometry["initial_state_digest"]) != (
                report["route_digest"], report["initial_state_digest"]
            ):
                raise RuntimeError("Unpaired survey replay")
            common, gi, ai = np.intersect1d(gt, time, return_indices=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out.with_suffix(".npz"), geometry_time=gt, geometry=gp,
                                actor_time=time, actor=track)
            report["animated_scene"] = draw_pair(
                out.with_suffix(""), common, gp[gi], track[ai], report["episodes"]
            )
    write_json(out, report)
    return dict(seed=seed, qualified=count, episodes=n, mean_tail_position_rmse_m=rmse,
                passed=count >= plan["minimum_qualified"])


def fit_phase(model, phase, prepared, directory, anchors, index, seed, plan, progress):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", phase["config"])
    data_path = directory / "demonstrations.npz"
    x, y, steps = demo_data(data_path, phase, seed + 100000 + index * 1000)

    torch.manual_seed(seed + index)
    rng = np.random.default_rng(seed + index)
    xs, ys, replay = [x], [y], {}
    for name in anchors:
        with np.load(prepared[name], allow_pickle=False) as data:
            ids = rng.choice(len(data["states"]), min(
                plan["replay_samples_per_task"], len(data["states"])), replace=False)
            xs.append(data["states"][ids])
            ys.append(data["actions"][ids])
            replay[name] = len(ids)

    train_data = tuple(torch.from_numpy(np.concatenate(parts)).to(model.device)
                       for parts in (xs, ys))
    loss = fit_actor(model, train_data, phase["updates"], plan["batch_size"])
    progress = dict(
        progress, bc_updates=progress["bc_updates"] + phase["updates"],
        imitation_physics_steps=progress["imitation_physics_steps"] + steps,
        critic_needs_calibration=True, survey_phase=phase["config"]["task"]["name"],
    )
    candidate = directory / "actor.pt"
    save(candidate, model, phase["config"], progress, "survey_curriculum_bc")
    return candidate, data_path, progress, dict(
        samples=len(x), replay=replay, updates=phase["updates"], loss=loss, physics_steps=steps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--through", help="Stop after this named phase")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = read(args.plan)
    base = load_config(device=args.device, seed=args.seed)
    phases = prepare(plan, base)
    names = list(phases)
    if args.through:
        if args.through not in phases:
            parser.error(f"Unknown phase: {args.through}")
        names = names[:names.index(args.through) + 1]
    out = args.out.resolve()
    manifest = dict(
        plan=plan, seed=args.seed, device=args.device, initialization="random",
        source_fingerprint=source_fingerprint(),
        script_sha256={name: digest(Path(__file__).with_name(name)) for name in
                       ("train_survey_from_scratch.py", "evaluate_random_survey.py",
                        "adapt_random_survey.py")},
    )
    if out.exists() and any(out.iterdir()) and not (out / "manifest.json").is_file():
        parser.error("--out contains unrelated files")
    if (out / "manifest.json").is_file() and read(out / "manifest.json") != manifest:
        parser.error("Run configuration changed; use a different --out")
    write_json(out / "manifest.json", manifest)
    torch.manual_seed(args.seed)
    model = new_model(base, optimizers=False)
    progress = dict(parent=None, ppo_updates=0, bc_updates=0, accepted_bc_updates=0,
                    imitation_physics_steps=0, cumulative_ppo_physics_steps=0,
                    critic_needs_calibration=True)
    initial = out / "random_initial_actor.pt"
    if not initial.exists():
        save(initial, model, base, progress, "random_initial")
    else:
        model, _, saved = load(initial, args.device)
        progress = saved["progress"]
    anchors, prepared, results = [], {}, []
    for index, name in enumerate(names):
        phase = phases[name]
        directory = out / name
        result_path = directory / "gate.json"
        if result_path.exists():
            result = read(result_path)
            checkpoint = directory / "actor.pt"
            if digest(checkpoint) != result["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint changed: {checkpoint}")
            model, _, saved = load(checkpoint, args.device)
            progress = saved["progress"]
            prepared[name] = directory / "demonstrations.npz"
        else:
            replay = list(dict.fromkeys([*anchors, *names[max(0, index - 1):index]]))
            checkpoint, prepared[name], progress, fit = fit_phase(
                model, phase, prepared, directory, replay, index, args.seed, plan, progress
            )
            gates = [assess(model, phase, args.seed + offset, plan,
                            directory / "gate" / f"{args.seed + offset}.json")
                     for offset in plan["gate_seed_offsets"]]
            retention = {
                prior: assess(model, phases[prior], args.seed + plan["retention_seed_offset"],
                              plan, directory / "retention" / f"{prior}.json")
                for prior in replay
            }
            qualified = all(r["passed"] for r in [*gates, *retention.values()])
            if qualified:
                progress["accepted_bc_updates"] += phase["updates"]
                save(checkpoint, model, phase["config"], progress, "survey_curriculum_bc")
            result = dict(
                phase=name, passed=qualified,
                fit=fit, current=gates, retention=retention,
                checkpoint_sha256=digest(checkpoint),
            )
            write_json(result_path, result)
        results.append(result)
        print(name, "passed" if result["passed"] else "failed",
              [r["qualified"] for r in result["current"]], flush=True)
        write_json(out / "curriculum_summary.json", results)
        if not result["passed"]:
            return 2
        if phase.get("anchor"):
            anchors.append(name)

    test_root = out / "heldout" / names[-1]
    tests = {}
    for name in names:
        tests[name] = []
        for index, offset in enumerate(plan["test_seed_offsets"]):
            seed = args.seed + offset
            tests[name].append(assess(
                model, phases[name], seed, plan, test_root / name / f"{seed}.json",
                render=index == 0 and not args.no_gif,
            ))
        write_json(test_root / "summary.json", dict(
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), tasks=tests,
        ))
    return 0 if all(r["passed"] for group in tests.values() for r in group) else 2


if __name__ == "__main__":
    raise SystemExit(main())
