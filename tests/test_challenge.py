import copy, json, math, hashlib
from pathlib import Path
import pytest
import torch
from pursuit_rl.config.rescue import load_rescue_profile, rescue_fingerprint
from pursuit_rl.config.challenge import (
    apply_scenario,
    assert_transfer_compatible,
    action_envelope,
    SCENARIOS,
)
from pursuit_rl.evaluation.challenge import run_challenge, initial_state_digest
from pursuit_rl.learning.rescue_checkpoint import (
    new_rescue_model,
    save_rescue_checkpoint,
    load_rescue_checkpoint,
)
from pursuit_rl.training.rescue import train_rescue
from pursuit_rl.simulation.rescue_environment import RescueBatch

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_rescue_profile(
        ROOT / "configs/rescue/velocity_50hz_2m.json", device="cpu"
    )


def tiny():
    cfg, p = config()
    cfg["curriculum"][0]["time"] = 0.006
    cfg["training"]["num_envs"] = 2
    p.update(
        num_envs=2,
        bc_updates=0,
        dagger_rounds=0,
        bc_min_success=0.0,
        validation_episodes=2,
        critic_mc_episodes=2,
        critic_pretrain_updates=2,
        critic_reanchor_updates=2,
        budget_physics_steps=8,
        validation_every_physics_steps=8,
        rollout_decisions=2,
        hidden_actor=[8],
        hidden_critic=[8],
        ppo_epochs=1,
        ppo_batch_size=4,
        decision_interval_steps=2,
    )
    cfg["physics"]["policy_interval_steps"] = 2
    return cfg, p


def test_scenarios_validate_and_leave_source_unchanged():
    cfg, p = config()
    before = copy.deepcopy(cfg)
    for name in SCENARIOS:
        c, q, m = apply_scenario(cfg, p, name)
        assert_transfer_compatible(cfg, p, c, q)
        assert c is not cfg and q is not p
    assert cfg == before


def test_baseline_regression_is_original_task_after_curriculum():
    cfg, p = config()
    cfg["curriculum"][0].update(
        modes=["singer"], distance=[10.0, 100.0], speed=[0.0, 8.0]
    )
    original, _, _ = apply_scenario(cfg, p, "baseline")
    same, _, _ = apply_scenario(cfg, p, "training_task")
    assert original["curriculum"][0]["modes"] == ["straight"]
    assert original["curriculum"][0]["distance"] == [10.0, 30.0] and original[
        "curriculum"
    ][0]["speed"] == [0.0, 5.0]
    assert same == cfg


def test_transfer_rejects_output_units_not_only_shapes():
    cfg, p = config()
    c, q, _ = apply_scenario(cfg, p, "singer_a1")
    assert_transfer_compatible(cfg, p, c, q)
    q["velocity_scale"] = [30.0, 30.0, 10.0]
    with pytest.raises(ValueError, match="velocity_scale"):
        assert_transfer_compatible(cfg, p, c, q)
    q = copy.deepcopy(p)
    q["action_mode"] = "acceleration"
    with pytest.raises(ValueError, match="action_mode"):
        assert_transfer_compatible(cfg, p, c, q)


def test_old_fingerprint_remains_compatible_with_optional_default():
    cfg, p = config()
    old = copy.deepcopy(p)
    old.pop("require_init_checkpoint")
    assert rescue_fingerprint(cfg, p) == rescue_fingerprint(cfg, old)
    p["require_init_checkpoint"] = True
    assert rescue_fingerprint(cfg, p) != rescue_fingerprint(cfg, old)


def test_command_envelope_exposes_high_speed_limit():
    cfg, p = config()
    c, q, _ = apply_scenario(cfg, p, "speed_20_25_limit")
    info = action_envelope(c, q)
    assert info["horizontal_command_warning"] and info[
        "max_horizontal_command"
    ] == pytest.approx(math.sqrt(450))


def test_metric_snapshot_freezes_at_first_event_and_clears_reset():
    cfg, p = tiny()
    env = RescueBatch(cfg, p)
    env.record_task_metrics = True
    _, _, done, _, records = env.step(torch.zeros(2, 3))
    assert not done.any()
    _, _, done, _, records = env.step(torch.zeros(2, 3))
    assert not done.any()
    _, _, done, _, records = env.step(torch.zeros(2, 3))
    assert done.all() and len(records) == 2
    assert all(
        "target_max_accel_observed" in r and r["target_boundary_active_steps"] == 0
        for r in records
    )
    assert env.step(torch.zeros(2, 3))[-1] == []
    env.target_boundary_steps.fill_(9)
    env.reset(torch.tensor([True, False]))
    assert env.target_boundary_steps.tolist() == [0, 9]


def test_explicit_transfer_preserves_actor_resets_critic_and_counts_lineage(tmp_path):
    cfg, p = tiny()
    source = new_rescue_model(cfg, p)
    with torch.no_grad():
        source.actor[0].weight[:, 32:].fill_(0.125)
        source.critic[-1].bias.fill_(300.0)
    parent = tmp_path / "parent.pt"
    save_rescue_checkpoint(
        parent, source, cfg, p, {"physics_steps": 100, "ppo_updates": 2}, "ppo"
    )
    cfg["curriculum"][0]["modes"] = ["singer"]
    cfg["curriculum"][0]["max_accel"] = 0.5
    p["require_init_checkpoint"] = True
    p["budget_physics_steps"] = 0
    result = train_rescue(cfg, p, tmp_path / "child", init_from=parent)
    child, c, q, data = load_rescue_checkpoint(
        tmp_path / "child/checkpoints/latest.pt", device="cpu"
    )
    assert all(
        torch.equal(v, child.actor.state_dict()[k])
        for k, v in source.actor.state_dict().items()
    )
    assert not torch.equal(source.critic[-1].bias, child.critic[-1].bias)
    assert data["origin"] == "transfer_initial" and result["ppo_updates"] == 0
    assert (
        result["parent_ppo_physics_steps"]
        == result["cumulative_ppo_physics_steps"]
        == 100
    )
    assert data["actor_optimizer"]["state"] == {} and data["critic_optimizer"]["state"]
    assert (
        data["progress"]["parent_checkpoint"]["sha256"]
        == hashlib.sha256(parent.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="requires --init-from"):
        train_rescue(cfg, p, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_challenge_records_paired_cases_and_unknown_actual_plant(tmp_path, monkeypatch):
    import pursuit_rl.evaluation.challenge as module

    cfg, p = tiny()
    model = new_rescue_model(cfg, p)
    parent = tmp_path / "parent.pt"
    save_rescue_checkpoint(
        parent, model, cfg, p, {"physics_steps": 8, "ppo_updates": 1}, "ppo"
    )
    captured = []

    def fake(
        model,
        cfg,
        p,
        episodes,
        seed,
        reference=False,
        stochastic=False,
        environment_factory=None,
    ):
        env = environment_factory(cfg, p, n=2, seed=seed, device="cpu")
        captured.append(
            (float(env.env.u.tau_rate.mean()), cfg["physics"]["tau_rate"], reference)
        )
        rows = [
            {
                "episode": i,
                "start_rel_pos": [i, 0, 0],
                "start_target_vel": [0, 0, 0],
                "time": 0.01,
                "success": True,
                "reason": "success",
            }
            for i in range(episodes)
        ]
        return {
            "episodes_detail": rows,
            "successes": episodes,
            "success_rate": 1.0,
            "mean_capture_time_success": 0.01,
            "failures": {},
        }, None

    monkeypatch.setattr(module, "evaluate_rescue", fake)
    result = run_challenge(
        parent, ["actuator_lag_150"], 2, [17], tmp_path / "cases", device="cpu"
    )
    assert len(captured) == 2 and all(
        nominal == 0.13 and actual > nominal * 1.2 for actual, nominal, _ in captured
    )
    case = result["cases"][0]
    assert (
        case["learned"]["initial_state_digest"]
        == case["reference"]["initial_state_digest"]
    )
    assert (tmp_path / "cases/summary.json").is_file()
