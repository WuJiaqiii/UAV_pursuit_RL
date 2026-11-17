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
