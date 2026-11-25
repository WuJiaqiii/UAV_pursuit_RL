"""Stage-to-stage PPO-only transfer curriculum: plans, scores, resume, gates."""
import json
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import source_fingerprint
from scripts import run_pure_ppo_curriculum as curriculum


@pytest.fixture()
def plan_file(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(
        schema=curriculum.SCHEMA,
        defaults=dict(
            preset="quick", num_envs=2, rollout_steps=2,
            batch_size=4, epochs=1, history_steps=1,
        ),
        stages=[
            dict(name="Stage 0", task="static", seeds=[42], physics_steps=40),
            dict(name="Stage 1", task="tail", seeds=[42], physics_steps=40),
        ],
        evaluation=dict(
            profiles=["static", "tail"],
            episodes=8,
            seed=920001,
            gate_success_rate=0.80,
            gate_mode="record_only",
        ),
        compare_direct_baseline=True,
    )), encoding="utf-8")
    return path


def test_expand_true_transfer_and_matched_scratch_control(plan_file):
    plan = json.loads(plan_file.read_text())
    runs, evaluation = curriculum.validate_plan(plan, "cpu")
    assert [r["id"] for r in runs] == [
        "curr_01_static_seed42", "curr_02_tail_seed42",
        "direct_02_tail_seed42",
    ]
    assert runs[0]["parent_id"] is None
    assert runs[1]["parent_id"] == runs[0]["id"]
    assert runs[2]["parent_id"] is None
    assert len({r["config_fingerprint"] for r in runs[1:]}) == 1
    assert sum(r["budget"] for r in runs) == 120
    assert evaluation["episodes"] == 8
    assert evaluation["profiles"] == ["static", "tail"]
    assert all(not r["config"]["prediction"]["enabled"] for r in runs)
    assert all(r["config"]["training"]["bc_updates"] == 0 for r in runs)
    assert all(r["config"]["training"]["aux_pd_coef"] == 0 for r in runs)
    assert all(not r["config"]["safety"]["constraint_enabled"] for r in runs)


def test_pairing_requires_same_seeds_at_each_stage(plan_file):
    plan = json.loads(plan_file.read_text())
    plan["stages"][1]["seeds"] = [43]
    with pytest.raises(ValueError, match="seeds"):
        curriculum.validate_plan(plan, "cpu")
    plan["stages"][1]["seeds"] = [42]
    plan["evaluation"]["profiles"] = ["tail", "tail"]
    with pytest.raises(ValueError, match="repeated"):
        curriculum.validate_plan(plan, "cpu")


def test_heldout_matrix_is_seed_matched_and_gates_are_explicit(
    tmp_path, plan_file, monkeypatch
):
    item = curriculum.validate_plan(
        json.loads(plan_file.read_text()), "cpu"
    )[0][0]
    checkpoint = tmp_path / "best_current.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    eval_calls = []

    def fake_load(path, device=None):
        return object(), item["config"], dict(
            source_fingerprint=source_fingerprint()
        )

    def fake_eval(model, cfg, episodes, seed):
        task = cfg["task"]["name"]
        eval_calls.append((task, episodes, seed))
        wins = 7 if task == "01_static" else 2
        return dict(
            episodes=episodes,
            successes=wins,
            success_rate=wins / episodes,
            failures=dict(tilt=1, timeout=episodes - wins - 1),
            unsafe_entry_rate=0.0,
            mean_maximum_tilt_deg=12.0,
            maximum_tilt_p95_deg=20.0,
        ), None

    monkeypatch.setattr(curriculum, "load_checkpoint", fake_load)
    monkeypatch.setattr(curriculum, "evaluate", fake_eval)
    settings = curriculum.validate_plan(
        json.loads(plan_file.read_text()), "cpu"
    )[1]
    report = curriculum._matrix_report(
        checkpoint, item["config"], item, settings, "cpu"
    )
    assert eval_calls[0][1:] == (8, 920001)
    assert eval_calls[1][1:] == (8, 930008)
    assert report["own_task_success_rate"] == pytest.approx(7 / 8)
    assert report["gate_met"] is True
    assert report["results"][0]["tilt_failure_rate"] == pytest.approx(1 / 8)
    assert report["results"][1]["timeout_rate"] == pytest.approx(5 / 8)


def test_full_mocked_campaign_transfers_parent_and_reports_each_difficulty(
    tmp_path, plan_file, monkeypatch, capsys
):
    plan = json.loads(plan_file.read_text())
    runs, _ = curriculum.validate_plan(plan, "cpu")
    by_run = {i["id"]: i for i in runs}
    child_commands = []
    class FakePopen:
        def __init__(self, command, stdout, stderr, env):
            child_commands.append(command)
            directory = Path(command[command.index("--out") + 1])
            item = by_run[directory.name]
            directory.mkdir(parents=True)
            (directory / "checkpoints").mkdir()
            (directory / "config.json").write_text(json.dumps(item["config"]))
            (directory / "checkpoints" / "best_current.pt").write_bytes(
                f"dummy checkpoint {item['id']}".encode()
            )
            (directory / "training_summary.json").write_text(json.dumps(dict(
                physics_steps=item["budget"], ppo_updates=4,
                completed_episodes=10, successful_episodes=4,
                source_fingerprint=source_fingerprint(),
            )))
            (directory / "metrics.jsonl").write_text(
                json.dumps(dict(
                    event="ppo_update",
                    physics_steps=item["budget"],
                    ppo_updates=4,
                    completed_episodes_total=10,
                    successful_episodes_total=4,
                )) + "\n"
            )
        def poll(self):
            return 0
    def fake_matrix(path, config, item, settings, device):
        wins = 6 if item["arm"] == "curriculum" else 2
        records = []
        for task in settings["profiles"]:
            records.append(dict(
                evaluated_task=task, episodes=8, successes=wins,
                success_rate=wins / 8, tilt_failure_rate=1/8,
                timeout_rate=0.0, unsafe_entry_rate=0.0,
                mean_maximum_tilt_deg=10.0,
                maximum_tilt_p95_deg=20.0,
            ))
        return dict(
            schema="v12-curriculum-stage-evaluation-v1",
            id=item["id"], arm=item["arm"], stage=item["stage"],
            stage_index=item["stage_index"], seed=item["seed"],
            task=item["task"], checkpoint=str(path.resolve()),
            checkpoint_sha256=curriculum.base._hash_file(path),
            checkpoint_source_fingerprint=source_fingerprint(),
            gate_success_rate=settings["gate_success_rate"],
            own_task_success_rate=wins/8,
            gate_met=wins/8 >= settings["gate_success_rate"],
            results=records,
        )

    monkeypatch.setattr(curriculum.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(curriculum, "_matrix_report", fake_matrix)
    output = tmp_path / "campaign"
    args = [
        "--out", str(output), "--plan", str(plan_file), "--device", "cpu",
        "--poll-seconds", "0.001", "--log-interval", "0.001",
    ]
    state = curriculum.run(args)
    assert state["status"] == "completed"
    assert state["completed_runs"] == 3
    assert state["percent"] == pytest.approx(100)
    assert len(child_commands) == 3
    assert "--init-from" not in child_commands[0]
    assert "--init-from" in child_commands[1]
    assert child_commands[1][-2] == "--init-from"
    assert child_commands[1][-1].endswith(
        "curr_01_static_seed42/checkpoints/best_current.pt"
    )
    assert "--init-from" not in child_commands[2]
    rows = (output / "curriculum_success_rates.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 * 2
    contrasts = json.loads((output / "curriculum_results.json").read_text())
    assert len(contrasts["contrasts"]) == 1
    assert contrasts["contrasts"][0]["difference_pp"] == pytest.approx(50.0)
    assert all((output / i["id"] / "difficulty_matrix.json").exists() for i in runs)
    assert "[CURRICULUM]" in capsys.readouterr().out

    def disallow_any_more(*args, **kwargs):
        raise AssertionError("Completed courses must be skipped during resume")
    monkeypatch.setattr(curriculum.subprocess, "Popen", disallow_any_more)
    restored = curriculum.run(args + ["--resume-existing"])
    assert restored["completed_runs"] == 3
    assert restored["percent"] == pytest.approx(100)
    status = curriculum.run(["--out", str(output), "--status"])
    assert status["percent"] == pytest.approx(100)
    assert "Difficulty matrix" in capsys.readouterr().out

    # Altering plan invalidates manifest; the campaign must not rerun silently.
    plan["evaluation"]["seed"] += 100
    plan_file.write_text(json.dumps(plan))
    with pytest.raises(SystemExit):
        curriculum.run(args + ["--resume-existing"])


def test_strict_gate_stops_after_failed_first_stage(
    tmp_path, plan_file, monkeypatch
):
    plan = json.loads(plan_file.read_text())
    plan["evaluation"]["gate_mode"] = "stop"
    plan_file.write_text(json.dumps(plan))
    calls = []
    class FakePopen:
        def __init__(self, command, stdout, stderr, env):
            calls.append(command)
            item_id = Path(command[command.index("--out") + 1]).name
            directory = Path(command[command.index("--out") + 1])
            item = next(
                i for i in curriculum.validate_plan(plan, "cpu")[0]
                if i["id"] == item_id
            )
            directory.mkdir(parents=True)
            (directory / "checkpoints").mkdir()
            (directory / "config.json").write_text(json.dumps(item["config"]))
            (directory / "checkpoints" / "best_current.pt").write_bytes(b"x")
            (directory / "training_summary.json").write_text(json.dumps(dict(
                physics_steps=item["budget"], ppo_updates=1,
                completed_episodes=2, successful_episodes=0,
                source_fingerprint=source_fingerprint(),
            )))
        def poll(self):
            return 0
    def fake_eval(path, config, item, settings, device):
        return dict(
            schema="v12-curriculum-stage-evaluation-v1",
            id=item["id"], arm=item["arm"], stage=item["stage"],
            stage_index=item["stage_index"], seed=item["seed"],
            task=item["task"], checkpoint=str(path),
            checkpoint_sha256=curriculum.base._hash_file(path),
            checkpoint_source_fingerprint=source_fingerprint(),
            gate_success_rate=0.8, own_task_success_rate=0,
            gate_met=False, results=[],
        )
    monkeypatch.setattr(curriculum.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(curriculum, "_matrix_report", fake_eval)
    state = curriculum.run([
        "--out", str(tmp_path / "gated"), "--plan", str(plan_file),
        "--device", "cpu", "--poll-seconds", "0.001",
    ])
    assert state["status"] == "gate_failed"
    assert len(calls) == 1


def test_success_rate_wilson_interval_has_nonzero_width_at_edges():
    lo, hi = curriculum.wilson95(0, 64)
    assert lo == pytest.approx(0)
    assert 0.04 < hi < 0.08
    lo2, hi2 = curriculum.wilson95(64, 64)
    assert hi2 == pytest.approx(1)
    assert 0.92 < lo2 < 0.96
    lower, upper = curriculum.wilson95(52, 64)
    assert lower < 52/64 < upper


def test_all_historical_registered_difficulty_groups_available():
    from scripts.run_pure_ppo import TASKS
    from scripts.evaluate_v12_curriculum_difficulties import tier
    assert len(TASKS) == 13
    assert tier("static") == "static"
    assert tier("random") == "low"
    assert tier("medium_random") == "medium"
    assert tier("high_random") == "high"
    assert tier("extreme_random") == "extreme"
