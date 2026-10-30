"""Regression coverage for reward, protocol, configuration and recovery fixes."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.evaluation.benchmark import run_benchmark
from pursuit_rl.learning.checkpoints import ARCHIVED_SCHEMA, load_model
from pursuit_rl.evaluation.comparison import compare, exact_mcnemar_p
from pursuit_rl.config import load_config, migrate_checkpoint_config, validate
from pursuit_rl.learning.contracts import require_compatible_policy
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.evaluation.rollout import evaluate, task_hash
from pursuit_rl.evaluation.feasibility import ideal_intercept_time
from pursuit_rl.artifacts.io import write_json
from pursuit_rl.simulation.observations import ACTOR_SLICES, observation_dimensions
from pursuit_rl.visualization.animation import prepare_trace, render_animation
from pursuit_rl.visualization.trajectory import plot_trace
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.simulation.rewards import potential_reward
from pursuit_rl.evaluation.scenarios import SCENARIOS, configuration, evaluation_stage
from pursuit_rl.evaluation.metrics.statistics import wilson
from pursuit_rl.training.trainer import Trainer


def tiny_config():
    return load_config("configs/smoke.yaml")


@pytest.mark.parametrize("event", ["ordinary", "success", "failure", "timeout"])
def test_environment_cpa_potential_is_counted_once(event):
    cfg = load_config()
    for key in cfg["reward"]:
        if key.startswith("w_"):
            cfg["reward"][key] = 0.0
    cfg["reward"].update(
        cpa_mode="potential", w_cpa=0.25, success_bonus=0.0, failure_penalty=0.0
    )
    cfg["curriculum"][0]["success_radius"] = 1.0
    env = PursuitBatch(cfg, n=1, stage=0, randomize=False)
    env.target.pos[:] = env.pursuer.pos + [20.0, 0.0, 0.0]
    env.target.vel[:] = 0.0
    env.last_cpa_potential[:] = -0.3
    if event == "success":
        env.target.pos[:] = env.pursuer.pos + [0.5, 0.0, 0.0]
    elif event == "failure":
        env.pursuer.pos[:, 0] = cfg["physics"]["workspace_xy"] + 1.0
    elif event == "timeout":
        cfg["curriculum"][0]["time"] = cfg["physics"]["dt"]
    _, reward, term, trunc, _ = env.step(np.zeros((1, 4)))
    relative = env.target.pos - env.pursuer.pos
    from pursuit_rl.simulation.geometry import cpa

    distance, _ = cpa(
        relative, env.target.vel - env.pursuer.vel, cfg["reward"]["cpa_horizon"]
    )
    phi = -distance / cfg["reward"]["distance_scale"]
    expected = 0.25 * (cfg["training"]["gamma"] * (0 if term[0] else phi[0]) + 0.3)
    assert reward[0] == pytest.approx(expected, abs=1e-7)
    assert bool(term[0]) == (event in ("success", "failure"))
    assert bool(trunc[0]) == (event == "timeout")


def test_precision_potential_terminal_vs_timeout():
    assert potential_reward(0.2, 0.5, True, 0.9, 0.3) == pytest.approx(-0.06)
    assert potential_reward(0.2, 0.5, False, 0.9, 0.3) == pytest.approx(0.075)


def test_short_animation_trail_between_samples():
    from pursuit_rl.visualization.animation import _path_up_to

    time = np.array([0.0, 1.0])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    path, current = _path_up_to(time, positions, 0.5, trail_seconds=0.1)
    np.testing.assert_allclose(path[:, 0], [0.4, 0.5])
    np.testing.assert_allclose(current, [0.5, 0.0, 0.0])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_named_scenarios_are_independent_of_training_distribution(name):
    standard = load_config()
    altered = load_config("configs/experiments/hard_hybrid_2m.yaml")
    altered["physics"]["tau_rate"] = 0.31
    a, b = configuration(standard, name), configuration(altered, name)
    assert task_hash(a, evaluation_stage(a)) == task_hash(b, evaluation_stage(b))
    require_compatible_policy(standard, a)
    env = PursuitBatch(a, n=2, stage=evaluation_stage(a), seed=22)
    assert np.isfinite(env.observe()[0]).all()


def test_config_overrides_inheritance_and_validation(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("training:\n  max_stage: 3\n", encoding="utf8")
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: parent.yaml\nstages:\n  stage0:\n    time: 7.0\n", encoding="utf8"
    )
    config = load_config(child)
    assert (
        config["training"]["max_stage"] == 3 and config["curriculum"][0]["time"] == 7.0
    )
    parent.write_text("extends: child.yaml\n", encoding="utf8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(child)
    for section, key, value in [
        ("physics", "substeps", 1.5),
        ("observation", "use_cpa", 1),
        ("target", "singer_alpha", 0.0),
        ("target", "singer_sigma", float("nan")),
        ("training", "actor_lr", -1.0),
        ("observation", "target_accel_scale", 0.0),
    ]:
        changed = load_config()
        changed[section][key] = value
        with pytest.raises(ValueError):
            validate(changed)
    bad = load_config()
    bad["curriculum"][0]["modes"] = []
    with pytest.raises(ValueError, match="modes"):
        validate(bad)
    bad = load_config()
    bad["curriculum"][1]["name"] = "stage0"
    with pytest.raises(ValueError, match="unique"):
        validate(bad)


def test_cpa_boundary_features_and_acceleration_encoding():
    cfg = load_config()
    cfg["observation"].update(use_cpa=False, boundary_features=True)
    env = PursuitBatch(cfg, n=3, seed=18, stage=0)
    actor, critic = env.observe()
    assert observation_dimensions(cfg) == (34, 46)
    assert actor.shape == (3, 34) and critic.shape == (3, 46)
    assert np.all(actor[:, ACTOR_SLICES["cpa"]] == 0)
    changed = configuration(cfg, "high_accel_rapid")
    assert changed["target"]["max_accel"] == 10.0
    assert changed["observation"]["target_accel_scale"] == 7.0
    require_compatible_policy(cfg, changed)
    changed["observation"]["target_accel_scale"] = 10.0
    with pytest.raises(ValueError, match="contract"):
        require_compatible_policy(cfg, changed)


def test_strict_target_ceiling_and_initial_geometry():
    cfg = configuration(load_config(), "strict_target_speed")
    env = PursuitBatch(cfg, n=30, seed=7, stage=evaluation_stage(cfg))
    for _ in range(100):
        env.target.step(0.05)
    assert env.target.max_observed_speed.max() <= 25.0 + 1e-7
    for name, projection in [
        ("overtaking_edge", 1),
        ("head_on_edge", -1),
        ("crossing_edge", 0),
    ]:
        scenario = configuration(load_config(), name)
        env = PursuitBatch(scenario, n=30, seed=8, stage=evaluation_stage(scenario))
        rel, vel = env.initial_rel_pos, env.initial_target_vel
        cosine = np.sum(rel * vel, axis=1) / (
            np.linalg.norm(rel, axis=1) * np.linalg.norm(vel, axis=1)
        )
        assert np.max(np.abs(cosine - projection)) < 0.1


def test_precision_shaping_preserves_initial_state():
    cfg = tiny_config()
    plain = copy.deepcopy(cfg)
    plain["reward"]["w_precision_potential"] = 0.0
    a = PursuitBatch(cfg, n=3, stage=0, seed=18)
    b = PursuitBatch(plain, n=3, stage=0, seed=18)
    assert np.array_equal(a.pursuer.pos, b.pursuer.pos)
    assert np.array_equal(a.target.pos, b.target.pos)
    for env in (a, b):
        env.target.pos[:] = env.pursuer.pos + [8.0, 0.0, 0.0]
        env.last_precision_potential[:] = np.exp(
            -8.0 / env.cfg["reward"]["precision_scale_m"]
        )
    ra = a.step(np.zeros((3, 4)))[1]
    rb = b.step(np.zeros((3, 4)))[1]
    assert np.any(np.abs(ra - rb) > 1e-7)


def test_new_run_protection_and_current_resume(tmp_path):
    cfg = tiny_config()
    out = tmp_path / "run"
    trainer = Trainer(cfg, out)
    trainer.run()
    before = (out / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        Trainer(cfg, out)
    with pytest.raises(FileExistsError):
        Trainer(cfg, out, init_from=out / "checkpoints/best.pt")
    resumed = Trainer(cfg, out, resume=out / "checkpoints/latest.pt")
    assert resumed.steps == cfg["training"]["total_steps"]
    resumed.run()
    assert (out / "run.json").read_bytes() == before
    assert "resume" in (out / "events.jsonl").read_text(encoding="utf8")
    changed = copy.deepcopy(cfg)
    changed["reward"]["w_cpa"] = 0.0
    with pytest.raises(ValueError, match="Resume"):
        Trainer(changed, out, resume=out / "checkpoints/latest.pt")
    fresh = Trainer(
        changed, tmp_path / "fresh", init_from=out / "checkpoints/latest.pt"
    )
    assert fresh.steps == 0 and not fresh.model.actor_optimizer.state
    fresh.run()


def test_archived_weights_keep_encoding_but_cannot_resume(tmp_path):
    cfg = tiny_config()
    model = AsymmetricPPO(*observation_dimensions(cfg), cfg["training"])
    archived = copy.deepcopy(cfg)
    archived["observation"].pop("target_accel_scale")
    archived["target"]["min_speed"] = 0.0
    archived["reward"]["success_radius"] = 5.0
    archived["training"]["test_seed"] = 82000
    checkpoint = tmp_path / "archived.pt"
    torch.save(
        {
            "schema": ARCHIVED_SCHEMA,
            "config": archived,
            "steps": 12,
            "actor_critic": model.state_dict(),
        },
        checkpoint,
    )
    predictor, data = load_model(checkpoint)
    assert predictor.actor_optimizer is None
    obs = PursuitBatch(cfg, n=2, stage=0).observe()[0]
    assert np.array_equal(predictor.predict(obs), model.predict(obs))
    assert data["config"]["observation"]["target_accel_scale"] == 7.0
    assert "min_speed" not in data["config"]["target"]
    warm = Trainer(cfg, tmp_path / "warm", init_from=checkpoint)
    assert warm.parent_checkpoint["parent_steps"] == 12
    warm.run()
    with pytest.raises(ValueError, match="Resume"):
        Trainer(cfg, tmp_path / "bad", resume=checkpoint)


def test_paired_reports_and_protocol_mismatch(tmp_path):
    cfg = tiny_config()
    cfg["curriculum"][0]["time"] = 0.1
    paths = []
    for name in ["PN", "APN"]:
        report = evaluate(GuidanceController(name), cfg, stage=0, episodes=3, seed=222)
        path = tmp_path / name / "episodes.json"
        write_json(path, report)
        paths.append(path.parent)
    result = compare(*paths)
    assert result["episodes"] == 3 and result["both_fail"] == 3
    assert exact_mcnemar_p(0, 0) == 1 and exact_mcnemar_p(3, 0) == pytest.approx(0.25)
    report = json.loads((paths[1] / "episodes.json").read_text(encoding="utf8"))
    report["evaluation_protocol"]["task_hash"] = "other"
    write_json(paths[1] / "episodes.json", report)
    with pytest.raises(ValueError, match="task_hash"):
        compare(*paths)


@pytest.mark.parametrize("radius", [1.0, 2.0, 3.0, 5.0])
def test_plot_and_animation_use_recorded_capture_radius(radius, tmp_path):
    trace = {
        "t": np.array([0.0, 1.0]),
        "pursuer_pos": np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]),
        "target_pos": np.array([[radius + 1.0, 0.0, -10.0], [radius, 0.0, -10.0]]),
        "pursuer_vel": np.zeros((2, 3)),
        "target_vel": np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        "capture_radius": np.asarray(radius),
        "hit_time": np.asarray(1.0),
        "hit_pursuer_pos": np.array([0.0, 0.0, -10.0]),
        "hit_target_pos": np.array([radius, 0.0, -10.0]),
    }
    time, pursuer, target, hit = prepare_trace(trace)
    assert hit == 1.0 and np.linalg.norm(target[-1] - pursuer[-1]) == radius
    outputs = plot_trace(trace, tmp_path / f"plot_{radius}.png")
    assert len(outputs) == 3 and all(path.stat().st_size > 0 for path in outputs)
    if radius == 3.0:
        outputs = render_animation(
            trace, tmp_path / "animation", fps=2, duration=1, max_frames=2, dpi=50
        )
        assert outputs[0].stat().st_size > 0


def test_json_and_feasibility():
    from pursuit_rl.artifacts.io import json_default

    assert json_default(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError):
        json_default(object())
    assert ideal_intercept_time([500, 0, 0], [25, 0, 0], 30) == pytest.approx(100.0)
    low, high = wilson(16, 32)
    assert low < 0.5 < high


@pytest.mark.parametrize(
    "module",
    [
        "train",
        "evaluate",
        "benchmark",
        "trace-metrics",
        "visualize",
        "report",
        "compare",
        "export",
        "feasibility",
    ],
)
def test_cli_imports_and_help(module):
    importlib.import_module("main")
    completed = subprocess.run(
        [sys.executable, "main.py", module, "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
