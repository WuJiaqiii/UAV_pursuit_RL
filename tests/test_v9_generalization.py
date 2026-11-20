"""v9 unattended generalization: mixture, distillation, DAgger and campaign safety."""
import copy
import json
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import new_model, save, load as load_checkpoint
from pursuit_rl.ctbr.config import load, validate
from pursuit_rl.ctbr.mixture import TaskMixtureBatch
from pursuit_rl.ctbr.training import train
from scripts.run_generalization_campaign import (
    model_configs, aggregate, stage_candidates, run, evaluate_methods,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(17)
    torch.set_num_threads(1)


def quick(task="02_low_tail"):
    c = load("configs/first_entry_highspeed/01_static.json")
    c["num_envs"] = 8
    c["prediction"]["enabled"] = False
    c["task"].update(
        name=task, duration=0.02, distance=[0.65, 0.75],
        speed=[0.0, 0.0], kind="hover",
    )
    c["training"].update(
        physics_steps=80, rollout_steps=8, batch_size=16, epochs=1,
        bc_updates=0, bc_rounds=1, critic_updates=0,
        mc_episodes=2, validation_episodes=2,
        validation_every=80, mc_calibration_every_updates=0,
        prediction_updates_enabled=False, initialization_success_floor=0.0,
        replay_recent_fraction=0, replay_anchor_fraction=0,
    )
    return validate(c)


def four(c):
    tasks = []
    names = ["01_static", "03_low_head_on", "04_low_random"]
    weights = [0.2, 0.25, 0.3]
    for name, weight in zip(names, weights):
        cc = copy.deepcopy(c)
        cc["task"]["name"] = name
        tasks.append({"config": validate(cc), "fraction": weight})
    return tasks


def test_four_task_allocation_and_one_step():
    c = quick()
    c["num_envs"] = 64
    tasks = four(c)
    for item in tasks:
        item["config"]["num_envs"] = 64
    env = TaskMixtureBatch(c, None, None, seed=100, extra_tasks=tasks)
    assert env.task_counts == {
        "02_low_tail": 16, "01_static": 13,
        "03_low_head_on": 16, "04_low_random": 19,
    }
    assert set(env.task_slices) == set(env.task_counts)
    a, priv = env.observe()
    assert a.shape[0] == priv.shape[0] == 64
    action = torch.zeros(64, 4)
    (_, _), reward, done, _, _ = env.step(action, action)
    assert reward.shape == done.shape == (64,)
    assert torch.isfinite(reward).all()
    with pytest.raises(ValueError, match="Duplicate"):
        TaskMixtureBatch(c, None, None, seed=100,
                         extra_tasks=tasks + [tasks[0]])


def test_trainer_multitask_distillation_smoke_and_resume_contract(tmp_path):
    c = quick()
    tasks = four(c)
    parent_path = tmp_path / "parent.pt"
    static_path = tmp_path / "static.pt"
    model = new_model(c, optimizers=False)
    progress = {"physics_steps": 0, "ppo_updates": 0,
                "cumulative_ppo_physics_steps": 0}
    save(parent_path, model, c, progress, "test_parent")
    static = copy.deepcopy(c)
    static["task"]["name"] = "01_static"
    save(static_path, new_model(static, optimizers=False),
         static, progress, "test_static")
    distill = dict(
        coefficient=0.05,
        teachers={"01_static": str(static_path), "02_low_tail": str(parent_path)},
    )
    profiles = [c] + [item["config"] for item in tasks]
    root = tmp_path / "four_task"
    result = train(
        c, root, init_from=str(parent_path),
        replay={"tasks": tasks}, distillation=distill,
        validation_profiles=profiles,
    )
    assert result["ppo_updates"] >= 1
    assert result["task_environment_counts"] == {
        "02_low_tail": 2, "01_static": 2,
        "03_low_head_on": 2, "04_low_random": 2,
    }
    metrics = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text().splitlines()
    ]
    update = next(row for row in metrics if row["event"] == "ppo_update")
    assert update["metrics"]["distill_samples"] > 0
    assert update["metrics"]["distill_coef"] == pytest.approx(0.05)
    assert (root / "generalization_initial.json").exists()
    saved_model, _, saved = load_checkpoint(root / "checkpoints/latest.pt")
    assert saved["progress"]["generalization_contract"]["distill_coefficient"] == .05
    with pytest.raises(ValueError, match="generalization"):
        train(c, root, resume=str(root / "checkpoints/latest.pt"),
              replay={"tasks": tasks},
              validation_profiles=profiles, distillation=None)


def test_campaign_plan_and_dry_run(tmp_path):
    plan_path = Path("configs/experiments/05_v9_generalization.json")
    plan = json.loads(plan_path.read_text())
    c = quick()
    variants = {v["name"]: v for v in plan["variants"]}
    for name, variant in variants.items():
        primary, others, profiles = model_configs(
            c, plan, variant, "pilot", 42, "cpu",
        )
        assert len(others) == 3
        assert len(profiles) == 4
        assert primary["task"]["radius"] == 0.5
        assert primary["physics"]["max_load_factor_g"] == 2.5
        assert primary["task"]["entry_relative_speed_max_mps"] == 15
    fake = tmp_path / "p.pt"
    save(fake, new_model(c, optimizers=False), c,
         {"physics_steps": 0, "ppo_updates": 0,
          "cumulative_ppo_physics_steps": 0}, "test_parent")
    results = run([
        "--plan", str(plan_path),
        "--parent-checkpoint", str(fake),
        "--static-checkpoint", str(fake),
        "--out", str(tmp_path / "campaign"),
        "--stage", "pilot", "--variants", "joint_ppo",
        "--seeds", "42", "--device", "cpu", "--dry-run",
    ])
    entry = results["stages"]["pilot"]["variants"]["joint_ppo"][0]
    assert entry["status"] == "dry_run"
    assert entry["ppo_physics_steps_budget"] == 8_000_000
    assert not (tmp_path / "campaign/pilot/joint_ppo/seed_42").exists()
    assert (tmp_path / "campaign/campaign_manifest.json").exists()


def test_promotion_ranks_worst_task_and_rejects_incomplete():
    plan = json.loads(Path("configs/experiments/05_v9_generalization.json").read_text())
    reports = {"pilot": {"variants": {}}}
    for name, score in [("good", 0.8), ("bad", 0.0)]:
        reports["pilot"]["variants"][name] = [
            {"status": "completed", "evaluations": {
                task: {"ppo_v9": {"success_rate": score}}
                for task in plan["profiles"]
            }} for _ in (42, 43)
        ]
    assert stage_candidates("full", plan, reports) == ["good"]
    assert aggregate([]) == {}


def test_matched_reference_controllers_report_same_initial_digest(tmp_path):
    c = quick()
    model = new_model(c, optimizers=False)
    reports = evaluate_methods(
        model, model, [c], episodes=2, seed=73, dest=tmp_path
    )
    task = c["task"]["name"]
    assert set(reports[task]) == {
        "ppo_v9", "ppo_v8", "pn_ctbr", "pd_ctbr",
        "pd_observed", "pd_observed_smoothed", "pn_pd_ctbr"
    }
    assert len({r["initial_state_digest"] for r in reports[task].values()}) == 1
    assert all(r["episodes"] == 2 for r in reports[task].values())


def test_unattended_pipeline_full_smoke_and_resume(tmp_path):
    """A real, tiny PPO run must reach checkpoint, baselines and summary."""
    plan = json.loads(
        Path("configs/experiments/05_v9_generalization.json").read_text()
    )
    parent_config = quick()
    paths = {}
    for name in ("01_static", "02_low_tail", "03_low_head_on", "04_low_random"):
        profile = copy.deepcopy(parent_config)
        profile["task"]["name"] = name
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(profile))
        paths[name] = str(path)
    plan["profiles"] = paths
    plan["num_envs"] = 8
    plan["stages"]["pilot"]["seeds"] = [42]
    plan["stages"]["pilot"]["variants"] = ["joint_ppo"]
    plan["stages"]["pilot"]["final_eval_episodes"] = 2
    plan["stages"]["pilot"]["training"] = dict(
        physics_steps=80, rollout_steps=2, batch_size=16,
        epochs=1, validation_every=80, validation_episodes=2,
        critic_updates=0, mc_episodes=2,
        mc_calibration_every_updates=0,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    parent_path = tmp_path / "parent.pt"
    save(
        parent_path, new_model(parent_config, optimizers=False),
        parent_config,
        {"physics_steps": 0, "ppo_updates": 0,
         "cumulative_ppo_physics_steps": 0},
        "tiny_parent",
    )
    out = tmp_path / "real_campaign"
    out.mkdir()
    # Simulates launch_v9.sh, which starts nohup with its log and PID
    # under the output directory before the Python campaign checks it.
    (out / "campaign.log").write_text("")
    (out / "campaign.pid").write_text("999999")
    argv = [
        "--plan", str(plan_path),
        "--parent-checkpoint", str(parent_path),
        "--static-checkpoint", str(parent_path),
        "--out", str(out),
        "--stage", "pilot", "--device", "cpu",
    ]
    result = run(argv)
    row = result["stages"]["pilot"]["variants"]["joint_ppo"][0]
    assert row["status"] == "completed"
    assert row["physics_steps"] >= 80
    assert set(row["evaluations"]) == set(paths)
    assert all(len(x) == 7 for x in row["evaluations"].values())
    assert len(result["stages"]["pilot"]["summary_by_variant"]) == 1
    ckpt = out / "pilot/joint_ppo/seed_42/checkpoints/latest.pt"
    assert ckpt.exists()
    checkpoint_bytes = ckpt.read_bytes()
    next_result = run(argv + ["--resume-existing"])
    assert next_result["stages"]["pilot"]["variants"]["joint_ppo"][0]["status"] == "completed"
    assert ckpt.read_bytes() == checkpoint_bytes

 
def test_heldout_profiles_are_distinct_and_preserve_safety_contract():
    from scripts.run_generalization_campaign import heldout_configs
    plan = json.loads(Path("configs/experiments/05_v9_generalization.json").read_text())
    c = quick()
    profiles = heldout_configs(c, plan, 42, "cpu")
    assert len(profiles) == 3
    assert len({p["task"]["name"] for p in profiles}) == 3
    for p in profiles:
        assert p["task"]["entry_relative_speed_max_mps"] == 15
        assert p["task"]["radius"] == .5
        assert p["physics"]["max_load_factor_g"] == 2.5
    strong = next(p for p in profiles if p["task"]["name"] == "holdout_random_stronger")
    assert strong["task"]["max_accel"] == 2

 
def test_v9_summary_csv_separates_controllers_and_variants(tmp_path):
    from scripts.summarize_v9_campaign import to_csv
    sample = {
        "stages": {"pilot": {"summary_by_variant": {
            "test_A": {"in_distribution": {
                "02_low_tail": {
                    "ppo_v9": {
                        "successes": 8, "episodes": 10, "success_rate": .8,
                        "wilson_95": [.49, .94],
                        "per_seed_success_rates": [.8], "failures": {"timeout": 2}
                    },
                    "pn_ctbr": {
                        "successes": 10, "episodes": 10, "success_rate": 1.,
                        "wilson_95": [.72, 1.], "per_seed_success_rates": [1.],
                        "failures": {}
                    },
                }
            }, "heldout": {}}
        }}}
    }
    path = to_csv(sample, tmp_path / "results.csv")
    import csv
    records = list(csv.DictReader(path.open()))
    assert len(records) == 2
    ppo = next(row for row in records if row["method"] == "ppo_v9")
    assert float(ppo["delta_success_vs_pn"]) == pytest.approx(-.2)
    assert ppo["distribution"] == "in_distribution"


def test_precheckpoint_interruption_is_archived_not_deleted(tmp_path):
    """Power loss during DAgger can safely restart from the parent checkpoint."""
    plan = json.loads(
        Path("configs/experiments/05_v9_generalization.json").read_text()
    )
    c = quick()
    profiles = {}
    for name in plan["profiles"]:
        profile = copy.deepcopy(c)
        profile["task"]["name"] = name
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(profile))
        profiles[name] = str(path)
    plan["profiles"] = profiles
    plan["num_envs"] = 8
    plan["stages"]["pilot"]["seeds"] = [42]
    plan["stages"]["pilot"]["variants"] = ["joint_ppo"]
    plan["stages"]["pilot"]["final_eval_episodes"] = 2
    plan["stages"]["pilot"]["training"] = dict(
        physics_steps=80, rollout_steps=2, batch_size=16, epochs=1,
        validation_every=80, validation_episodes=2, critic_updates=0,
        mc_episodes=2, mc_calibration_every_updates=0,
    )
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    checkpoint = tmp_path / "parent.pt"
    save(checkpoint, new_model(c, optimizers=False), c, {
        "physics_steps": 0, "ppo_updates": 0,
        "cumulative_ppo_physics_steps": 0,
    }, "smoke")
    root = tmp_path / "campaign"
    root.mkdir()
    run_dir = root / "pilot/joint_ppo/seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text('{"incomplete": true}')
    (run_dir / "partial_log.txt").write_text("preserve-me")
    args = [
        "--plan", str(plan_file), "--parent-checkpoint", str(checkpoint),
        "--static-checkpoint", str(checkpoint), "--out", str(root),
        "--stage", "pilot", "--device", "cpu", "--resume-existing",
    ]
    from scripts.run_generalization_campaign import checksum
    from pursuit_rl.ctbr.checkpoint import source_fingerprint
    from pursuit_rl.ctbr.checkpoint import write_json
    write_json(root / "campaign_manifest.json", {
        "schema": plan["schema"], "plan_sha256": checksum(plan_file),
        "parent_sha256": checksum(checkpoint),
        "static_teacher_sha256": checksum(checkpoint),
        "source_fingerprint": source_fingerprint(),
    })
    result = run(args)
    assert result["stages"]["pilot"]["variants"]["joint_ppo"][0]["status"] == "completed"
    archived = root / "pilot/joint_ppo/seed_42__precheckpoint_attempt_01"
    assert (archived / "partial_log.txt").read_text() == "preserve-me"
    assert (archived / "run_manifest.json").exists()
    assert (run_dir / "checkpoints/latest.pt").exists()
