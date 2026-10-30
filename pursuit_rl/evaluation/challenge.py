"""Paired, frozen-network challenge evaluation with auditable task shifts."""

import copy
import hashlib
import json
import math
from pathlib import Path
from pursuit_rl.config.challenge import (
    apply_scenario,
    action_envelope,
    assert_transfer_compatible,
)
from pursuit_rl.config.rescue import rescue_contract
from pursuit_rl.learning.rescue_checkpoint import load_rescue_checkpoint
from pursuit_rl.evaluation.rescue import evaluate_rescue
from pursuit_rl.simulation.rescue_environment import RescueBatch


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    temporary.replace(path)


def initial_state_digest(report):
    rows = [
        {
            "episode": r["episode"],
            "position": r["start_rel_pos"],
            "target_velocity": r["start_target_vel"],
        }
        for r in report["episodes_detail"]
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def run_challenge(
    checkpoint,
    scenarios,
    episodes,
    seeds,
    out,
    device="cpu",
    num_envs=None,
    stochastic=False,
    include_reference=True,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    model, training_cfg, training_profile, saved = load_rescue_checkpoint(
        checkpoint, device=device
    )
    if num_envs is not None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        training_profile["num_envs"] = num_envs
    cases = []
    for scenario in scenarios:
        cfg, profile, multipliers = apply_scenario(
            training_cfg, training_profile, scenario
        )
        assert_transfer_compatible(training_cfg, training_profile, cfg, profile)

        def factory(c, p, **kwargs):
            env = RescueBatch(c, p, **kwargs)
            env.record_task_metrics = True
            # Change the actual plant, keeping the controller's nominal model fixed.
            for key, factor in multipliers.items():
                setattr(env.env.pursuer, key, getattr(env.env.pursuer, key) * factor)
            return env

        for seed in seeds:
            reports = {}
            for controller in (
                ["learned", "reference"] if include_reference else ["learned"]
            ):
                report, _ = evaluate_rescue(
                    model if controller == "learned" else None,
                    cfg,
                    profile,
                    episodes,
                    seed,
                    reference=controller == "reference",
                    stochastic=stochastic and controller == "learned",
                    environment_factory=factory,
                )
                report.update(
                    challenge_schema="pursuit-challenge-v1",
                    scenario=scenario,
                    checkpoint=str(Path(checkpoint).resolve()),
                    checkpoint_origin=saved["origin"],
                    checkpoint_sha256=checkpoint_sha256,
                    source_ppo_updates=saved["progress"]["ppo_updates"],
                    source_physics_steps=saved["progress"]["physics_steps"],
                    training_contract=saved["contract"],
                    evaluation_contract=rescue_contract(cfg, profile),
                    actual_plant_multipliers=multipliers,
                    control_interface_preserved=True,
                    critic_calibration_matches_sampling_policy=stochastic
                    and controller == "learned",
                    action_envelope=action_envelope(cfg, profile),
                )
                report["initial_state_digest"] = initial_state_digest(report)
                filename = (
                    f"{scenario}__seed_{seed}__{controller}__"
                    + (
                        "stochastic"
                        if stochastic and controller == "learned"
                        else "deterministic"
                    )
                    + ".json"
                )
                _write(out / filename, report)
                reports[controller] = report
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "controller": controller,
                            "seed": seed,
                            "successes": report["successes"],
                            "episodes": episodes,
                            "mean_capture_time_success": report[
                                "mean_capture_time_success"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if (
                include_reference
                and reports["learned"]["initial_state_digest"]
                != reports["reference"]["initial_state_digest"]
            ):
                raise RuntimeError("Learned and reference initial states differ")
            case = {
                "scenario": scenario,
                "seed": seed,
                "episodes": episodes,
                "paired_with_reference": include_reference,
                "task": copy.deepcopy(cfg["curriculum"][profile["stage"]]),
                "plant_multipliers": multipliers,
                "envelope": action_envelope(cfg, profile),
            }
            for controller, r in reports.items():
                # Penalized duration includes failures instead of dropping them from the mean.
                times = sorted(e["time"] for e in r["episodes_detail"] if e["success"])
                case[controller] = {
                    "successes": r["successes"],
                    "success_rate": r["success_rate"],
                    "mean_capture_time_success": r["mean_capture_time_success"],
                    "mean_capture_time_timeout_penalty": sum(
                        (
                            e["time"]
                            if e["success"]
                            else cfg["curriculum"][profile["stage"]]["time"]
                        )
                        for e in r["episodes_detail"]
                    )
                    / episodes,
                    "mean_episode_end_time": sum(
                        e["time"] for e in r["episodes_detail"]
                    )
                    / episodes,
                    "failures": r["failures"],
                    "initial_state_digest": r["initial_state_digest"],
                }
            cases.append(case)
            result = {
                "schema": "pursuit-challenge-suite-v1",
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_origin": saved["origin"],
                "checkpoint_sha256": checkpoint_sha256,
                "source_ppo_updates": saved["progress"]["ppo_updates"],
                "episodes_per_case": episodes,
                "evaluation_seeds": seeds,
                "training_seed": training_cfg["seed"],
                "device": device,
                "num_envs": training_profile["num_envs"],
                "stochastic": stochastic,
                "scope": "Frozen-policy OOD evaluation, not new policy training or multi-training-seed evidence.",
                "cases": cases,
            }
            _write(out / "summary.json", result)
    return result
