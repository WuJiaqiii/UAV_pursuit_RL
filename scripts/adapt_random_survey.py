"""Adapt the low-speed survey Actor with randomized navigation demonstrations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluate_random_survey import LEGS, SurveyBatch, configure
from pursuit_rl.ctbr.checkpoint import load, save, write_json
from pursuit_rl.ctbr.control import teacher
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import fit_actor, imitation_data

ROUTE_SEEDS = (5220042, 5320042)
RETENTION_SEED = 5420042
REPLAY_TASKS = ("01_waypoint", "02_survey_arc")


@torch.no_grad()
def demonstrations(c, leg, seed):
    env = SurveyBatch(c, leg, seed, 32)
    obs, _ = env.observe()
    states, actions = [], []
    for tick in range(env.max_decision_steps):
        command = teacher(env)
        if tick % 4 == 0:
            active = ~env.finished
            states.append(obs[active, :21].clone())
            actions.append(command[active].clone())
        (obs, _), _, _, _, _ = env.step(command)
        if bool(env.finished.all()):
            break
    return torch.cat(states), torch.cat(actions), int(env.steps.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, c, saved = load(args.checkpoint, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "random_survey_bc.pt"
    source = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if not target.is_file():
        data, sources = [], []
        for name in REPLAY_TASKS:
            base = json.loads((args.profiles / f"{name}.json").read_text())
            replay = dict(c, task={**c["task"], **base["task"]},
                          training={**c["training"], "bc_samples": 12000,
                                    "bc_segment_seconds": 2.0})
            x, y, steps = imitation_data(replay, seed=5520042 + len(data))
            data.append((x[:, :21], y))
            sources.append(dict(task=name, samples=len(x), physics_steps=steps))
        for name, leg in LEGS.items():
            for seed in ROUTE_SEEDS:
                x, y, steps = demonstrations(configure(c, name, leg), leg, seed)
                data.append((x, y))
                sources.append(dict(task=name, seed=seed, samples=len(x), physics_steps=steps))
        loss = fit_actor(model, tuple(torch.cat(parts) for parts in zip(*data)), 2200, 2048)
        progress = dict(
            saved["progress"], survey_parent_sha256=source, critic_needs_calibration=True,
            bc_updates=saved["progress"]["bc_updates"] + 2200,
            accepted_bc_updates=saved["progress"]["accepted_bc_updates"] + 2200,
            imitation_physics_steps=saved["progress"]["imitation_physics_steps"]
            + sum(r["physics_steps"] for r in sources),
        )
        save(target, model, c, progress, "random_survey_bc")
        write_json(args.out / "demonstrations.json", dict(parent_sha256=source, loss=loss,
                                                           sources=sources))
    else:
        model, _, candidate = load(target, args.device)
        if candidate["progress"].get("survey_parent_sha256") != source:
            raise RuntimeError("Existing checkpoint has a different parent")

    retention = []
    base_model, _, _ = load(args.checkpoint, args.device)
    for name in REPLAY_TASKS:
        task = json.loads((args.profiles / f"{name}.json").read_text())["task"]
        profile = dict(c, task={**c["task"], **task})
        row = {"task": name}
        reports = []
        for label, policy in (("original", base_model), ("adapted", model)):
            path = args.out / "retention" / name / f"{label}.json"
            if path.exists():
                report = json.loads(path.read_text())
            else:
                report, _ = evaluate(policy, profile, 16, RETENTION_SEED)
                write_json(path, report)
            reports.append(report)
            row[label] = dict(
                qualified=round(16 * report["joint_precision_success_rate"]), episodes=16,
                mean_tail_position_rmse_m=report["mean_tail_position_rmse"],
                failures=report["failures"],
            )
        if reports[0]["initial_state_digest"] != reports[1]["initial_state_digest"]:
            raise RuntimeError("Unpaired retention initial states")
        retention.append(row)
    write_json(args.out / "retention" / "summary.json", retention)


if __name__ == "__main__":
    main()
