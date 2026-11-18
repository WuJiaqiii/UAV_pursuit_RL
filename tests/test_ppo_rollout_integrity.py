"""Regression tests for PPO rollout value integrity and numerical fail-fast guards."""

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import new_model
from pursuit_rl.ctbr.config import load
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.training import emit, train
from pursuit_rl.learning.buffers import TensorRolloutBuffer
from pursuit_rl.learning.ppo import AsymmetricPPO


def _quick_config():
    config = load("configs/first_entry_highspeed/01_static.json")
    config["num_envs"] = 2
    config["prediction"]["enabled"] = False
    config["task"].update(duration=0.02, distance=[0.65, 0.75])
    config["training"].update(
        bc_updates=0,
        critic_updates=0,
        physics_steps=32,
        rollout_steps=8,
        batch_size=8,
        epochs=1,
        mc_episodes=2,
        validation_episodes=2,
        validation_every=16,
        mc_calibration_every_updates=0,
        prediction_updates_enabled=False,
    )
    return config


def test_rollout_values_are_not_overwritten_by_reward_components(tmp_path, monkeypatch):
    """The buffer must retain the Critic prediction for each pre-action state."""
    original_update = AsymmetricPPO.update

    def verify_buffer_values(model, buffer, aux_pd_coef=0.0):
        expected = torch.stack(
            [model.values_tensor(buffer.priv[tick]) for tick in range(buffer.priv.shape[0])]
        )
        # Batched GEMM kernels can differ by a few float32 ulps when the leading
        # dimension changes; the per-tick call mirrors sample_tensor's shape.
        assert torch.allclose(buffer.values, expected, rtol=0.0, atol=1e-6)
        return original_update(model, buffer, aux_pd_coef=aux_pd_coef)

    monkeypatch.setattr(AsymmetricPPO, "update", verify_buffer_values)
    result = train(_quick_config(), tmp_path / "value_integrity")
    assert result["ppo_updates"] >= 1


def test_gae_matches_hand_calculation_for_termination_and_truncation():
    buffer = TensorRolloutBuffer(
        3, 1, 1, 1, "cpu", diagnostics={"enabled": False}
    )
    buffer.reward[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    buffer.values[:, 0] = torch.tensor([0.5, 0.6, 0.7])
    buffer.next_values[:, 0] = torch.tensor([0.6, 0.7, 0.8])
    buffer.terminated[:, 0] = torch.tensor([False, False, True])
    buffer.truncated[:, 0] = torch.tensor([False, True, False])

    buffer.compute_gae(gamma=0.9, lam=0.8)

    expected_advantage = torch.tensor([2.5016, 2.03, 2.3])
    expected_returns = expected_advantage + torch.tensor([0.5, 0.6, 0.7])
    assert torch.allclose(
        buffer.advantage[:, 0], expected_advantage, atol=1e-5, rtol=0.0
    )
    assert torch.allclose(
        buffer.returns[:, 0], expected_returns, atol=1e-5, rtol=0.0
    )
    assert buffer.bootstrap_values[:, 0].tolist() == pytest.approx([0.6, 0.7, 0.0])


def test_ppo_fails_fast_on_finite_but_exploding_returns():
    config = _quick_config()
    env = CTBRBatch(config, n=2)
    obs, critic_obs = env.observe()
    model = new_model(config)
    _, raw, logprob, value = model.sample_tensor(obs, critic_obs)
    buffer = TensorRolloutBuffer(
        2,
        2,
        obs.shape[-1],
        critic_obs.shape[-1],
        "cpu",
        diagnostics={
            "value_abs_threshold": 100.0,
            "target_abs_threshold": 100.0,
            "advantage_abs_threshold": 100.0,
        },
    )
    for tick in range(2):
        buffer.obs[tick] = obs
        buffer.priv[tick] = critic_obs
        buffer.raw[tick] = raw
        buffer.logprob[tick] = logprob
        buffer.values[tick] = value
        buffer.next_values[tick] = value
        buffer.reward[tick] = 0.0
        buffer.returns[tick] = 101.0

    with pytest.raises(FloatingPointError, match="returns.*exceeds configured limit"):
        model.update(buffer)


def test_popart_rebase_preserves_raw_critic_predictions():
    config = _quick_config()
    env = CTBRBatch(config, n=4)
    _, critic_obs = env.observe()
    model = new_model(config)
    before = model.values_tensor(critic_obs).clone()
    targets = torch.linspace(-4.0, 3.0, 128, device=model.device)
    model.update_value_statistics(targets)
    after = model.values_tensor(critic_obs)
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)


def test_emit_identifies_nonfinite_nested_metric_path(tmp_path):
    with pytest.raises(ValueError, match=r"event\.metrics\.critic\.mean"):
        emit(
            tmp_path,
            "ppo_update",
            metrics={"critic": {"mean": float("nan")}},
        )
    assert not (tmp_path / "metrics.jsonl").exists()

def test_factorial_plan_build_configs_and_gate_policy():
    import json
    from pathlib import Path
    from scripts.run_reward_ablation import build_configs

    plan = json.loads(
        Path("configs/experiments/03_low_tail_factorial.json").read_text()
    )
    assert plan["schema"] == "pursuit-first-entry-reward-factorial-v2"
    assert len(plan["variants"]) == 14
    variant = next(item for item in plan["variants"] if item["name"] == "p6_b05_ev05")
    config, replay = build_configs(plan, variant, "screening", 42, "cpu")
    assert config["training"]["bc_updates"] == 500
    assert config["training"]["bc_rounds"] == 2
    assert config["training"]["initialization_success_floor"] == 0.0
    assert config["reward"]["progress"] == 6
    assert config["reward"]["braking"] == 0.5
    assert config["reward"]["entry_velocity"] == 0.5
    assert replay["reward"] == config["reward"]

    gated, _ = build_configs(
        plan, variant, "screening", 42, "cpu", enforce_initialization_gate=True
    )
    assert gated["training"]["initialization_success_floor"] == 0.5

    no_bc = next(item for item in plan["variants"] if item["name"] == "baseline_no_bc")
    no_bc_config, _ = build_configs(plan, no_bc, "screening", 42, "cpu")
    assert no_bc_config["training"]["bc_updates"] == 0
    assert no_bc_config["training"]["bc_rounds"] == 1

def test_checkpoint_loader_accepts_v8_config_without_bc_rounds(tmp_path):
    from pursuit_rl.ctbr.checkpoint import save, load as load_checkpoint
    from pursuit_rl.ctbr.config import fingerprint

    config = _quick_config()
    model = new_model(config, optimizers=False)
    path = tmp_path / "legacy_v8_without_bc_rounds.pt"
    save(path, model, config, {"physics_steps": 0}, "legacy_compatible")
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["config"]["training"].pop("bc_rounds")
    data["fingerprint"] = fingerprint(data["config"])
    torch.save(data, path)

    _, restored, _ = load_checkpoint(path)
    assert restored["training"]["bc_rounds"] == 1

def test_factorial_launcher_dry_run_writes_matrix_without_starting_training(tmp_path):
    from pursuit_rl.ctbr.checkpoint import save
    from scripts.run_reward_ablation import main

    config = _quick_config()
    parent_model = new_model(config, optimizers=False)
    parent = tmp_path / "parent.pt"
    save(parent, parent_model, config, {"physics_steps": 0}, "dry_run_parent")

    output = tmp_path / "dry_run"
    result = main(
        [
            "--plan",
            "configs/experiments/03_low_tail_factorial.json",
            "--parent-checkpoint",
            str(parent),
            "--out",
            str(output),
            "--stage",
            "screening",
            "--variants",
            "p6_b05_ev05",
            "--dry-run",
            "--device",
            "cpu",
        ]
    )
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "dry_run"
    assert result["results"][0]["ppo_physics_steps_budget"] == 4_000_000
    assert (output / "ablation_matrix.json").is_file()
    assert not (output / "p6_b05_ev05" / "seed_42").exists()

def test_dagger_rounds_retain_prior_round_labels(tmp_path, monkeypatch):
    from pursuit_rl.ctbr.training import run_bc_initialization

    config = _quick_config()
    config["training"].update(bc_updates=2, bc_rounds=2)
    model = new_model(config)
    root = tmp_path
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    progress = {
        "bc_updates": 0,
        "accepted_bc_updates": 0,
        "imitation_physics_steps": 0,
        "bc_rounds_completed": 0,
    }
    fit_sizes = []

    def fake_imitation_data(c, model=None, seed=0):
        state = torch.full((2, model.obs_dim), float(seed % 1000))
        label = torch.zeros((2, 4))
        return state, label, 3

    def fake_fit_actor(model, data, updates, batch_size):
        fit_sizes.append((len(data[0]), updates))
        return 0.0

    def fake_validation(model, c, root, progress, label):
        return {"success_rate": 1.0}

    monkeypatch.setattr("pursuit_rl.ctbr.training.imitation_data", fake_imitation_data)
    monkeypatch.setattr("pursuit_rl.ctbr.training.fit_actor", fake_fit_actor)
    monkeypatch.setattr(
        "pursuit_rl.ctbr.training.initialization_validation", fake_validation
    )

    origin, report = run_bc_initialization(
        config,
        model,
        root,
        checkpoint_dir,
        progress,
        transfer=True,
        seed_offset=20000,
    )
    assert origin == "transfer_bc_initial"
    assert report["success_rate"] == 1.0
    assert fit_sizes == [(2, 1), (4, 1)]
    assert progress["bc_rounds_completed"] == 2
    assert progress["bc_updates"] == 2

def test_long_horizon_parameter_matrix_is_well_formed():
    import json
    from pathlib import Path
    from scripts.run_reward_ablation import build_configs
    path = Path("configs/experiments/04_low_tail_long_optimization.json")
    plan = json.loads(path.read_text())
    variants = {variant["name"]: variant for variant in plan["variants"]}
    assert len(variants) == 15
    assert len(plan["stages"]) == 3
    assert plan["stages"]["extended"]["training"]["physics_steps"] == 80_000_000
    assert len(plan["stages"]["extended"]["seeds"]) == 5
    assert plan["default_extended_variants"] == [
        "reference_p6_bc2", "p6_bc3_750"
    ]
    for stage in ("screening", "full", "extended"):
        for variant in plan["variants"]:
            c, replay = build_configs(plan, variant, stage, 42, "cpu")
            assert c["task"]["radius"] == 0.5
            assert c["task"]["entry_relative_speed_max_mps"] == 15
            assert c["physics"]["max_load_factor_g"] == 2.5
            assert c["physics"]["max_speed"] == 60
            assert c["training"]["replay_recent_fraction"] == pytest.approx(.2)
            assert c["training"]["replay_anchor_fraction"] == 0
            assert c["reward"]["completion_bonus"] == 10
            assert c["reward"]["failure"] == 10
            assert replay["reward"] == c["reward"]
            assert replay["training"]["discount_time_s"] == c["training"]["discount_time_s"]
            assert replay["training"]["advantage_time_s"] == c["training"]["advantage_time_s"]
            assert c["training"]["bc_rounds"] <= max(1, c["training"]["bc_updates"])
    baseline, _ = build_configs(plan, variants["reference_p6_bc2"], "screening", 42, "cpu")
    actor, _ = build_configs(plan, variants["p6_actor_lr_1e5"], "screening", 42, "cpu")
    gae, replay = build_configs(plan, variants["p6_gae_5s"], "screening", 42, "cpu")
    assert actor["training"]["actor_lr"] == pytest.approx(1e-5)
    assert baseline["training"]["actor_lr"] == pytest.approx(3e-5)
    assert gae["training"]["advantage_time_s"] == pytest.approx(5)
    assert replay["training"]["advantage_time_s"] == pytest.approx(5)


def test_multi_parameter_variants_cannot_change_unsafe_flight_contract():
    import copy
    import json
    from pathlib import Path
    from scripts.run_reward_ablation import build_configs
    plan = json.loads(
        Path("configs/experiments/04_low_tail_long_optimization.json").read_text()
    )
    variant = {"name": "bad_hard_speed", "training": {"max_load_factor_g": 999}}
    with pytest.raises(ValueError, match="Unsupported variant training"):
        build_configs(plan, variant, "screening", 42, "cpu")
    variant = {"name": "bad_reward", "reward": {"unknown_key": 1}}
    with pytest.raises(ValueError, match="Unknown reward"):
        build_configs(plan, variant, "screening", 42, "cpu")


def test_long_horizon_dry_run_is_reproducible(tmp_path):
    from pursuit_rl.ctbr.checkpoint import save
    from scripts.run_reward_ablation import main
    config = _quick_config()
    parent = tmp_path / "parent.pt"
    save(parent, new_model(config, optimizers=False),
         config, {"physics_steps": 0}, "dry_run_parent")
    plan = "configs/experiments/04_low_tail_long_optimization.json"
    out = tmp_path / "long"
    result = main([
        "--plan", plan,
        "--parent-checkpoint", str(parent),
        "--out", str(out),
        "--stage", "extended",
        "--variants", "reference_p6_bc2",
        "--seeds", "42",
        "--dry-run", "--device", "cpu",
    ])
    assert len(result["results"]) == 1
    run = result["results"][0]
    assert run["status"] == "dry_run"
    assert run["ppo_physics_steps_budget"] == 80_000_000
    assert run["task_mix"]["current_envs"] == 51
    assert run["task_mix"]["replay_envs"] == 13
    assert (out / "ablation_matrix.json").exists()
    assert not (out / "reference_p6_bc2" / "seed_42").exists()
