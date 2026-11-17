"""End-to-end first-entry PPO, evaluation, checkpoint and curriculum tests."""
import copy
import json
import pytest
import torch
from pursuit_rl.ctbr.config import DEFAULT, SCHEMA, load, validate
from pursuit_rl.ctbr.checkpoint import new_model, save, load as load_checkpoint
from pursuit_rl.ctbr.curriculum import gate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.evaluation import evaluate, score
from pursuit_rl.ctbr.training import train

@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(123)
    torch.set_num_threads(1)

def quick():
    c = load("configs/first_entry_highspeed/01_static.json")
    c["num_envs"] = 2
    c["prediction"]["enabled"] = False
    c["task"].update(duration=0.02, distance=[0.65, 0.75])
    c["training"].update(
        bc_updates=0, critic_updates=0, physics_steps=32, rollout_steps=8,
        batch_size=8, mc_episodes=2, validation_episodes=2,
        validation_every=16, mc_calibration_every_updates=0,
        prediction_updates_enabled=False,
    )
    return validate(c)

def test_evaluation_mc_returns_and_trace(tmp_path):
    c = quick()
    model = new_model(c, optimizers=False)
    report, data = evaluate(
        model, c, 2, 89, stochastic=True, collect_mc=True,
        trace_dir=tmp_path, trace_count=1,
    )
    assert report["schema"] == "pursuit-first-entry-evaluation-v1"
    assert report["episodes"] == 2
    assert "unsafe_entry_rate" in report
    assert len(report["episodes_detail"]) == 2
    assert data[0].shape[0] == data[1].shape[0]
    assert list(tmp_path.glob("trace_seed_*.npz"))
    assert len(score(report)) == 3

def test_ppo_train_and_checkpoint(tmp_path):
    c = quick()
    result = train(c, tmp_path / "training")
    assert result["physics_steps"] >= 32
    assert result["ppo_updates"] >= 1
    model, restored, saved = load_checkpoint(
        tmp_path / "training/checkpoints/latest.pt"
    )
    assert restored["schema"] == SCHEMA
    assert model.obs_dim > 0
    assert saved["progress"]["ppo_updates"] >= 1

def test_unsafe_crossing_during_ppo_rollout(tmp_path, monkeypatch):
    c = quick()
    c["task"]["distance"] = [0.55, 0.56]
    c["policy"]["decision_steps"] = 1
    original_reset = CTBRBatch.reset
    def reset_fast(self, mask=None):
        original_reset(self, mask)
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        u, t = self.plant.pursuer, self.plant.target
        u.pos[mask] = t.pos[mask].float() + u.pos.new_tensor([0.505, 0, 0])
        u.vel[mask] = u.vel.new_tensor([-17.0, 0, 0])
        return self.observe()
    monkeypatch.setattr(CTBRBatch, "reset", reset_fast)
    result = train(c, tmp_path / "unsafe")
    assert result["physics_steps"] >= 32

def test_checkpoint_loader_rejects_historical_versions(tmp_path):
    c = quick()
    model = new_model(c, optimizers=False)
    path = tmp_path / "new.pt"
    save(path, model, c, {"physics_steps": 0}, "test")
    _, saved_c, _ = load_checkpoint(path)
    assert saved_c["schema"] == SCHEMA
    bad = tmp_path / "old.pt"
    data = torch.load(path, weights_only=False)
    data["schema"] = "pursuit-direct-ctbr-checkpoint-v4"
    torch.save(data, bad)
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(bad)

def test_first_entry_retention_uses_success_not_tail_metrics(tmp_path, monkeypatch):
    c = load("configs/first_entry_highspeed/01_static.json")
    def report_stub(*args, **kwargs):
        return {
            "seed": args[3], "success_rate": 1.0,
            "unsafe_entry_rate": 0.0, "failures": {}
        }, None
    monkeypatch.setattr("pursuit_rl.ctbr.curriculum.evaluate", report_stub)
    plan = {"gate_seed_offsets": [100, 200]}
    anchors = [{"name": "anchor", "config": c, "reference": [{"seed": 123}]}]
    result = gate(None, c, anchors, tmp_path, 4, 42, 0.9, plan, "test")
    assert result["passed"]
    assert result["retention"][0]["success_rate"] == 1.0

def test_prediction_input_and_actor_logprob():
    c = load("configs/first_entry_highspeed/01_static.json")
    env = CTBRBatch(c, n=2)
    actor, critic = env.observe()
    model = new_model(c, optimizers=False)
    act, raw, lp, _ = model.sample_tensor(actor, critic)
    mean, std = model.distribution_parameters(actor)
    assert act.shape == (2, 4)
    assert torch.allclose(model.logprob(mean, std, raw), lp, atol=1e-6)

def test_oracle_actor_excludes_random_motion_latents():
    env = CTBRBatch(load("configs/first_entry_highspeed/01_static.json"), n=2)
    actor, critic = env.observe()
    env.command_velocity.fill_(9)
    env.ou.fill_(4)
    actor2, critic2 = env.observe()
    assert torch.equal(actor, actor2)
    assert not torch.equal(critic, critic2)

def test_cli_has_no_legacy_migration():
    from main import parser
    cli = parser()
    args = cli.parse_args(["curriculum", "--out", "runs/x"])
    assert args.plan.endswith("first_entry_highspeed/plan.json")
    with pytest.raises(SystemExit):
        cli.parse_args(["migrate"])

def test_highspeed_tail_full_ppo_update(tmp_path):
    c = load("configs/first_entry_highspeed/08_high_tail.json")
    c["num_envs"] = 2
    c["prediction"]["enabled"] = False
    c["task"]["duration"] = 0.03
    c["training"].update(
        bc_updates=0, critic_updates=0, physics_steps=40,
        rollout_steps=8, batch_size=8, mc_episodes=2,
        validation_episodes=2, validation_every=40,
        mc_calibration_every_updates=0, prediction_updates_enabled=False,
    )
    result = train(validate(c), tmp_path / "highspeed_ppo")
    assert result["ppo_updates"] >= 1
    assert result["decision_transitions"] >= 2


def test_two_stage_first_entry_curriculum_smoke(tmp_path):
    import json
    from pursuit_rl.ctbr.curriculum import run
    c = quick()
    c["training"]["physics_steps"] = 96
    c["training"]["minimum_success"] = 0.0
    phases = []
    for index in (1, 2):
        copy_c = __import__("copy").deepcopy(c)
        name = f"0{index}_smoke"
        copy_c["task"]["name"] = name
        copy_c["task"]["distance"] = [0.65 + .1 * (index-1), 0.75 + .1 * (index-1)]
        (tmp_path / f"{name}.json").write_text(json.dumps(copy_c))
        phases.append(dict(
            name=name, profile=f"{name}.json",
            difficulty_axis="distance", retention_anchor=index == 1,
        ))
    plan = dict(
        schema="pursuit-first-entry-curriculum-v1",
        prescreen=True, early_promotion=True,
        gate_seed_offsets=[79000, 179000],
        retention_seed_offsets=[89000, 189000],
        phases=phases,
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    result = run(path, tmp_path / "course", episodes=2, minimum_success=0.0)
    assert len(result) == 2
    assert all(r["passed"] for r in result)


def test_init_from_runs_current_phase_dagger_bc_before_ppo(tmp_path):
    parent_config = quick()
    parent_model = new_model(parent_config, optimizers=False)
    parent_path = tmp_path / "parent.pt"
    save(
        parent_path,
        parent_model,
        parent_config,
        {"physics_steps": 0, "cumulative_ppo_physics_steps": 0, "ppo_updates": 0},
        "test_parent",
    )

    child = load("configs/first_entry_highspeed/02_low_tail.json")
    child["prediction"]["enabled"] = False
    child["num_envs"] = 2
    child["task"].update(duration=0.02, distance=[0.65, 0.75])
    child["training"].update(
        physics_steps=32, rollout_steps=8, batch_size=8, epochs=1,
        bc_updates=2, bc_samples=8, bc_stride=1, bc_segment_seconds=0.01,
        initialization_episodes=2, initialization_success_floor=0.0,
        critic_updates=0, mc_episodes=2, validation_episodes=2,
        validation_every=16, mc_calibration_every_updates=0,
        prediction_updates_enabled=False,
    )
    child["training"]["teacher_method"] = "pd"
    child["reward"]["progress"] = 6.0
    result = train(validate(child), tmp_path / "transfer_bc", init_from=parent_path)
    assert result["bc_updates"] == 2
    assert result["accepted_bc_updates"] == 2
    assert result["imitation_physics_steps"] > 0
    assert (tmp_path / "transfer_bc/initialization_bc.json").exists()
    assert (tmp_path / "transfer_bc/checkpoints/latest.pt").exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "transfer_bc/metrics.jsonl").read_text().splitlines()
    ]
    bc = next(row for row in rows if row["event"] == "bc")
    assert bc["mode"] == "dagger_actor_states"
    assert bc["teacher_method"] == "pd"
