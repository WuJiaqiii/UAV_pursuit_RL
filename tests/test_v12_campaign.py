"""V12 whole-campaign weighted progress, resume, manifest and UI tests."""
import io
import json
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import source_fingerprint
from scripts import run_pure_ppo_campaign as campaign


@pytest.fixture
def plan_file(tmp_path):
    path = tmp_path / "plan.json"
    plan = dict(
        schema=campaign.SCHEMA,
        defaults=dict(
            preset="quick", num_envs=2, rollout_steps=2,
            batch_size=4, epochs=1, history_steps=1,
        ),
        stages=[
            dict(name="Static", task="static", seeds=[42], physics_steps=40),
            dict(name="Tail", task="tail", seeds=[43], physics_steps=80),
        ],
    )
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_weighted_global_progress_is_not_run_count():
    specs = [
        dict(id="one", stage="Static", budget=100),
        dict(id="two", stage="Tail", budget=900),
    ]
    states = dict(
        one=dict(status="completed", steps=100),
        two=dict(status="running", steps=0, stage="Tail", task="tail",
                 seed=42, budget=900, ppo_updates=0,
                 completed_episodes=0, training_success_rate=None),
    )
    snap = campaign.compute_snapshot(
        specs, states, elapsed_s=20, active_id="two"
    )
    assert snap["percent"] == pytest.approx(10)
    assert snap["completed_runs"] == 1
    assert snap["estimated_remaining_s"] == pytest.approx(180)
    assert "10.0%" in campaign.progress_lines(snap)[0]
    assert "Tail" in campaign.progress_lines(snap)[1]


def test_plan_expansion_is_deterministic_and_strict(plan_file):
    plan = json.loads(plan_file.read_text())
    runs = campaign.expand_plan(plan, "cpu")
    assert len(runs) == 2
    assert [x["budget"] for x in runs] == [40, 80]
    assert all(x["config"]["prediction"]["enabled"] is False for x in runs)
    assert all(x["config"]["training"]["bc_updates"] == 0 for x in runs)
    assert all(x["config"]["training"]["aux_pd_coef"] == 0 for x in runs)
    assert all(x["config"]["safety"]["constraint_enabled"] is False for x in runs)
    assert runs[0]["id"] == "01_static_seed42"
    assert runs[1]["id"] == "02_tail_seed43"
    with pytest.raises(ValueError, match="repeated"):
        campaign.expand_plan({
            **plan,
            "stages": plan["stages"] + [plan["stages"][0]],
        }, "cpu")
    with pytest.raises(ValueError, match="seeds"):
        campaign.expand_plan({
            **plan,
            "stages": [{**plan["stages"][0], "seeds": [42, 42]}],
        }, "cpu")


def test_metrics_tail_ignores_pre_resume_updates(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(dict(event="ppo_update", physics_steps=90)) + "\n")
    tail = campaign.MetricsTail(path, resume=True)
    assert tail.poll() is None
    with path.open("a") as stream:
        stream.write(json.dumps(dict(
            event="ppo_update", physics_steps=40, ppo_updates=2,
            completed_episodes_total=10, successful_episodes_total=7,
        )) + "\n")
    row = tail.poll()
    assert row["physics_steps"] == 40
    state = dict(budget=100, steps=0, ppo_updates=0)
    campaign.update_from_metrics(state, row)
    assert state["steps"] == 40
    assert state["training_success_rate"] == pytest.approx(0.7)


def test_aggregate_display_is_append_only_in_non_tty():
    output = io.StringIO()
    renderer = campaign.CampaignDisplay(
        stream=output, interval=10, clock=lambda: 0,
    )
    specs = [dict(id="run", stage="Static", budget=100)]
    states = dict(run=dict(
        id="run", stage="Static", task="static", seed=42,
        status="running", budget=100, steps=50, ppo_updates=10,
        completed_episodes=4, successful_episodes=3,
        training_success_rate=0.75,
    ))
    snap = campaign.compute_snapshot(specs, states, elapsed_s=5, active_id="run")
    renderer.render(snap, force=True)
    assert output.getvalue().count("[CAMPAIGN]") == 1
    assert output.getvalue().count("[CURRENT]") == 1
    assert "50.0%" in output.getvalue()
    assert "\r" not in output.getvalue()


def test_checkpoint_progress_recovery_uses_checkpoint_not_old_ui(tmp_path, plan_file):
    spec = campaign.expand_plan(json.loads(plan_file.read_text()), "cpu")[0]
    folder = tmp_path / spec["id"]
    (folder / "checkpoints").mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(spec["config"]))
    checkpoint = folder / "checkpoints/latest.pt"
    torch.save(dict(
        source_fingerprint=source_fingerprint(),
        config=spec["config"],
        progress=dict(physics_steps=20, ppo_updates=1,
                      completed_episodes=2, successful_episodes=1),
    ), checkpoint)
    state, found = campaign.inspect_run(spec, tmp_path)
    assert found == checkpoint
    assert state["steps"] == 20
    assert state["status"] == "pending"
    assert state["training_success_rate"] == pytest.approx(0.5)


def test_campaign_runs_all_stages_and_resumes_without_duplication(
    tmp_path, plan_file, monkeypatch, capsys
):
    specs = campaign.expand_plan(json.loads(plan_file.read_text()), "cpu")
    spec_map = {x["id"]: x for x in specs}
    launched = []

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, env):
            launched.append(cmd)
            run_dir = Path(cmd[cmd.index("--out") + 1])
            spec = spec_map[run_dir.name]
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps(spec["config"]), encoding="utf-8"
            )
            (run_dir / "training_summary.json").write_text(
                json.dumps(dict(
                    source_fingerprint=source_fingerprint(),
                    physics_steps=spec["budget"], ppo_updates=2,
                    completed_episodes=10, successful_episodes=8,
                )), encoding="utf-8"
            )
            (run_dir / "metrics.jsonl").write_text(json.dumps(dict(
                event="ppo_update", physics_steps=spec["budget"],
                ppo_updates=2, completed_episodes_total=10,
                successful_episodes_total=8,
            )) + "\n", encoding="utf-8")

        def poll(self):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", FakePopen)
    out = tmp_path / "experiment"
    args = [
        "--plan", str(plan_file), "--out", str(out), "--device", "cpu",
        "--poll-seconds", "0.001", "--log-interval", "0.001",
    ]
    result = campaign.run(args)
    assert len(launched) == 2
    assert result["status"] == "completed"
    assert result["completed_runs"] == 2
    assert result["completed_budget_steps"] == 120
    assert result["percent"] == pytest.approx(100)
    assert [r["status"] for r in result["runs"]] == ["completed", "completed"]
    assert (out / "campaign_manifest.json").exists()
    assert (out / "campaign_progress.json").exists()
    assert (out / "campaign_results.json").exists()
    assert "[CAMPAIGN]" in capsys.readouterr().out

    def should_not_run(*args, **kwargs):
        raise AssertionError("A completed stage must never run again")

    monkeypatch.setattr(campaign.subprocess, "Popen", should_not_run)
    resumed = campaign.run(args + ["--resume-existing"])
    assert resumed["completed_runs"] == 2
    assert resumed["completed_budget_steps"] == 120
    status = campaign.run(["--out", str(out), "--status"])
    assert status["percent"] == pytest.approx(100)

    changed = json.loads(plan_file.read_text())
    changed["stages"][0]["physics_steps"] += 1
    plan_file.write_text(json.dumps(changed))
    with pytest.raises(SystemExit):
        campaign.run(args + ["--resume-existing"])
