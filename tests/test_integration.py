"""Regression tests for the shared training/evaluation/trace workflows."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from pursuit_rl.config import load_config, dump_config, validate
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.geometry import exp_so3
from pursuit_rl.control.guidance import acceleration_to_action, GuidanceController
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.artifacts.trace import load_trace, save_episode
from pursuit_rl.evaluation.metrics import evaluate_trace
from pursuit_rl.evaluation.metrics.mission import mission_metrics
from pursuit_rl.evaluation.metrics.maneuver import (
    lateral_acceleration_cost,
    control_variation,
)
from pursuit_rl.evaluation.metrics.prediction import cpa_distance
from pursuit_rl.evaluation.metrics.report import aggregate
from pursuit_rl.evaluation.benchmark import run_benchmark


@pytest.fixture
def tiny_cfg():
    cfg = load_config()
    cfg["training"].update(
        num_envs=2,
        n_steps=4,
        batch_size=8,
        epochs=1,
        total_steps=8,
        eval_every=8,
        eval_episodes=2,
        initial_stage=0,
        max_stage=0,
        hidden_actor=[16],
        hidden_critic=[16],
    )
    for stage in cfg["curriculum"]:
        stage["time"] = 0.15
    return cfg


def test_config_and_validation():
    c = load_config()
    assert c["training"]["max_stage"] == 6
    assert [s["success_radius"] for s in c["curriculum"][-3:]] == [3.0, 2.0, 1.0]
    assert c["reward"]["w_precision_potential"] > 0
    assert c["reward"]["w_speed_risk"] > 0
    wrong = copy.deepcopy(c)
    wrong["curriculum"] = {"stages": [{"radius": 2}]}
    with pytest.raises(ValueError, match="curriculum"):
        validate(wrong)
    wrong = copy.deepcopy(c)
    wrong["training"]["eval_every"] = 0
    with pytest.raises(ValueError):
        validate(wrong)


def test_controller_ned_signs_yaw_and_attitude_feedback():
    e = PursuitBatch(load_config(), n=1, stage=0, randomize=False)
    assert np.allclose(acceleration_to_action(e, [[0, 0, 0]]), 0)
    assert acceleration_to_action(e, [[0, 0, -2]])[0, 0] > 0
    assert acceleration_to_action(e, [[0, 2, 0]])[0, 1] > 0
    assert acceleration_to_action(e, [[2, 0, 0]])[0, 2] < 0
    e.pursuer.rot[:] = exp_so3(np.array([[0.0, 0.0, np.pi / 2]]))
    assert acceleration_to_action(e, [[2, 0, 0]])[0, 1] < 0
    e.pursuer.rot[:] = exp_so3(np.array([[0.2, 0.0, 0.0]]))
    assert acceleration_to_action(e, [[0, 0, 0]])[0, 1] < 0


@pytest.mark.parametrize("algorithm", ["PN", "APN"])
def test_baseline_can_accelerate_collinear_chase(algorithm):
    e = PursuitBatch(load_config(), n=1, stage=0, randomize=False)
    e.target.pos[:] = e.pursuer.pos + [20.0, 0.0, 0.0]
    e.target.vel[:] = [4.0, 0.0, 0.0]
    initial = e.pursuer.vel[0, 0]
    ctrl = GuidanceController(algorithm)
    for _ in range(20):
        action = ctrl.act(e, e.observe()[0])
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0
        e.pursuer.step(action, 0.05)
    assert e.pursuer.vel[0, 0] > initial + 0.5


def test_crossing_event_survives_offline_analysis(tmp_path):
    c = load_config()
    c["curriculum"][0]["success_radius"] = 1.0
    e = PursuitBatch(c, n=1, stage=0, randomize=False)
    e.target.pos[:] = e.pursuer.pos + [3.0, 0.0, 0.0]
    seq = [{k: v[0].copy() for k, v in e.snapshot().items()}]

    def crossing(dt):
        e.target.pos[:] = e.pursuer.pos + [-3.0, 0.0, 0.0]

    e.target.step = crossing
    _, _, term, trunc, info = e.step(np.zeros((1, 4)))
    assert term[0] and not trunc[0] and info[0]["success"]
    seq.append({k: v[0].copy() for k, v in e.snapshot().items()})
    rec = {"episode": 0, **info[0]}
    path = tmp_path / "episode.npz"
    save_episode(path, seq, rec, c, 0, 1234)
    z = load_trace(path)
    metrics = evaluate_trace(z)
    assert metrics["success"]
    assert metrics["capture_time"] == pytest.approx(0.05 / 3)
    assert metrics["min_sampled_distance"] == pytest.approx(3.0)
    assert metrics["min_distance"] == pytest.approx(0.0)
    assert np.linalg.norm(z["hit_target_pos"] - z["hit_pursuer_pos"]) == pytest.approx(
        1.0
    )
    with pytest.raises(ValueError, match="relabel"):
        mission_metrics(z, capture_radius=2.0)
    legacy = {k: z[k] for k in ("t", "target_pos", "pursuer_pos", "relative_distance")}
    assert mission_metrics(legacy, capture_radius=1.0)["capture_time"] == pytest.approx(
        0.05 / 3
    )
    with pytest.raises(ValueError, match="explicit"):
        mission_metrics(legacy)


def test_cost_uses_recorded_intervals_and_missing_is_not_zero():
    trace = {
        "t": np.array([0.0, 0.2, 0.5]),
        "vel": np.array([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.5, 0.0]]),
    }
    first = 0.2
    v = trace["vel"][1]
    a = np.array([0.0, 1.0, 0.0])
    vh = v / np.linalg.norm(v)
    expected = first + np.sum((a - np.dot(a, vh) * vh) ** 2) * 0.3
    assert lateral_acceleration_cost(trace) == pytest.approx(expected)
    assert control_variation(trace) is None
    with pytest.raises(ValueError, match="dt"):
        lateral_acceleration_cost(trace, dt=0.05)
    assert cpa_distance([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0])[0] == pytest.approx(8.0)


def test_seeded_batches_paired_and_no_post_terminal_trace(tiny_cfg, tmp_path):
    arrays = []
    for name in ("PN", "APN"):
        folder = tmp_path / name
        result = evaluate(
            GuidanceController(name),
            tiny_cfg,
            stage=0,
            episodes=65,
            seed=222,
            trace_dir=folder,
        )
        assert len(result["episodes_detail"]) == 65
        traces = [load_trace(p) for p in sorted(folder.glob("episode_*.npz"))]
        assert len(traces) == 65
        assert int(traces[0]["batch_seed"]) == 222
        assert int(traces[64]["batch_seed"]) == 286
        for z in traces:
            assert z["t"][0] == 0.0
            assert z["t"][-1] <= 0.15 + 1e-9
            assert evaluate_trace(z)["reason"] == "timeout"
        arrays.append(traces)
    for a, b in zip(*arrays):
        assert np.array_equal(a["target_pos"], b["target_pos"])
        assert np.array_equal(a["target_vel"], b["target_vel"])
        assert np.array_equal(a["pursuer_pos"][0], b["pursuer_pos"][0])


def test_training_cli_and_three_method_report(tiny_cfg, tmp_path):
    cfg_path = tmp_path / "smoke.yaml"
    dump_config(tiny_cfg, cfg_path)
    train_dir = tmp_path / "train"
    subprocess.run(
        [
            sys.executable,
            "main.py",
            "train",
            "--config",
            str(cfg_path),
            "--out",
            str(train_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checkpoint = train_dir / "checkpoints/best.pt"
    assert checkpoint.is_file()
    out = tmp_path / "comparison"
    subprocess.run(
        [
            sys.executable,
            "main.py",
            "benchmark",
            "--algorithms",
            "PPO",
            "PN",
            "APN",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(cfg_path),
            "--stage",
            "0",
            "--episodes",
            "3",
            "--seed",
            "333",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads((out / "summary.json").read_text())
    assert [r["algorithm"] for r in rows] == ["PPO", "PN", "APN"]
    assert all(r["episodes"] == 3 for r in rows)
    for file in (
        "protocol.json",
        "summary.csv",
        "summary_table.tex",
        "summary.png",
        "complete.json",
    ):
        assert (out / file).stat().st_size > 0
    for row in rows:
        folder = out / row["algorithm"] / "configured_stage"
        records = json.loads((folder / "metrics.json").read_text())
        assert row["success_rate"] == aggregate(records)["success_rate"]
        assert all(r["mission_source"] == "environment_event" for r in records)
    trace = out / "PPO/configured_stage/episode_0000.npz"
    offline = tmp_path / "offline.json"
    subprocess.run(
        [sys.executable, "main.py", "trace-metrics", str(trace), "--out", str(offline)],
        check=True,
        capture_output=True,
    )
    first_record = json.loads((out / "PPO/configured_stage/metrics.json").read_text())[
        0
    ]
    assert json.loads(offline.read_text())["success"] == first_record["success"]
    with pytest.raises(FileExistsError):
        run_benchmark(tiny_cfg, ("PN",), episodes=3, out=out, plots=False)


def test_promotion_and_warmstart_use_existing_trainer(tiny_cfg, tmp_path):
    from pursuit_rl.training.trainer import Trainer

    c = copy.deepcopy(tiny_cfg)
    c["training"].update(
        max_stage=1,
        total_steps=16,
        min_stage_steps=0,
        promotion_success=0.0,
        promotion_consecutive=1,
    )
    trainer = Trainer(c, tmp_path / "curriculum")
    trainer.run()
    assert trainer.stage == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "curriculum/events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "promotion" for e in events)
    c["reward"]["w_precision_potential"] = 0.2
    fresh = Trainer(
        c,
        tmp_path / "warmstart",
        init_from=str(tmp_path / "curriculum/checkpoints/latest.pt"),
    )
    assert fresh.steps == 0 and fresh.parent_checkpoint["parent_steps"] == 16
    with pytest.raises(ValueError, match="Resume"):
        Trainer(
            c,
            tmp_path / "bad_resume",
            resume=str(tmp_path / "curriculum/checkpoints/latest.pt"),
        )


def test_incompatible_ppo_observation_contract_rejected(tiny_cfg, tmp_path):
    from pursuit_rl.training.trainer import Trainer

    training = Trainer(tiny_cfg, tmp_path / "train")
    training.run()
    changed = copy.deepcopy(tiny_cfg)
    changed["observation"]["use_cpa"] = False
    with pytest.raises(ValueError, match="observation"):
        run_benchmark(
            changed,
            ("PPO",),
            tmp_path / "train/checkpoints/best.pt",
            out=tmp_path / "bad",
            plots=False,
        )
