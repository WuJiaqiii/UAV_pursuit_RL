"""Behavioral checks for the new clocks, causal forecasts and value learning."""

import copy
import json
import math
from pathlib import Path
import pytest
import torch

from pursuit_rl.ctbr.config import load, validate, observation_dims
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.checkpoint import new_model, save, load as restore
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.training import train
from pursuit_rl.ctbr.value_learning import calibrate_critic
from pursuit_rl.ctbr.mixture import TaskMixtureBatch
from pursuit_rl.learning.buffers import TensorRolloutBuffer
from pursuit_rl.learning.prediction import supervised_windows, fit_predictor
from pursuit_rl.learning.valuenorm import ValueNorm


def config():
    return load("configs/ctbr/smoke.json")


def zero_actor(model):
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()


def rollout(model, c, steps=24):
    env = CTBRBatch(c)
    obs, priv = env.observe()
    buffer = TensorRolloutBuffer(
        steps, env.n, env.actor_dim, env.critic_dim, env.device
    )
    positions = []
    for tick in range(steps):
        action, raw, lp, value = model.sample_tensor(obs, priv)
        (nobs, npriv), reward, done, trunc, _ = env.step(action, raw)
        for key, value in dict(
            obs=obs,
            priv=priv,
            raw=raw,
            logprob=lp,
            values=value,
            reward=reward,
            terminated=done,
            truncated=trunc,
            next_values=model.values_tensor(npriv),
        ).items():
            getattr(buffer, key)[tick] = value
        positions.append(env.plant.target.pos.float().clone())
        obs, priv = env.reset(done) if bool(done.any()) else (nobs, npriv)
    buffer.compute_gae(env.gamma, env.gae_lambda)
    return buffer, torch.stack(positions)


def test_ten_substeps_match_one_ms_integration_and_discounted_reward():
    c = config()
    c["prediction"]["enabled"] = False
    c["reward"]["action_derivative"] = 0
    one = copy.deepcopy(c)
    one["policy"]["decision_steps"] = 1
    a, b = CTBRBatch(c), CTBRBatch(one)
    action = torch.full((2, 4), 0.05)
    (_, _), reward, _, _, _ = a.step(action)
    expected = torch.zeros(2)
    for tick in range(10):
        expected += b.physics_gamma**tick * b.step(action)[1]
    for key in ("pos", "vel", "acc", "rot", "rate", "thrust"):
        assert torch.allclose(
            getattr(a.plant.pursuer, key), getattr(b.plant.pursuer, key)
        )
    assert torch.allclose(reward, expected, atol=1e-7)
    assert a.last_physics_steps == 20
    assert torch.equal(a.raw_history[:, 1:], torch.zeros_like(a.raw_history[:, 1:]))
    assert math.isclose(a.gamma, b.gamma**10)
    assert math.isclose(a.gamma * a.gae_lambda, math.exp(-0.01 / 3))


def test_action_derivative_uses_decision_interval_once():
    c = config()
    env = CTBRBatch(c)
    action = torch.full((2, 4), 0.1)
    env.step(action)
    expected = -c["reward"]["scale"] * c["reward"]["action_derivative"] * 0.04 / 0.01
    assert torch.allclose(
        env.reward_components["action_derivative"], torch.full((2,), expected)
    )
    env.step(action)
    assert torch.allclose(
        env.reward_components["action_derivative"], torch.zeros(2), atol=1e-10
    )


def test_diagnostics_use_decision_time_and_current_thrust_mapping():
    c = config()
    model = new_model(c)
    buffer, _ = rollout(model, c)
    buffer.physics = c["physics"]
    buffer.decision_dt = 0.01
    buffer.action_semantics = "ctbr-held-v3"
    buffer.diagnostic_steps = 24
    summary = buffer.diagnostic_summary()
    temporal = summary["actions"]["command_autocorrelation"]
    assert temporal["1"]["lag_seconds"] == 0.01
    assert temporal["5"]["lag_seconds"] == 0.05
    assert temporal["13"]["lag_seconds"] == 0.13
    assert "command_thrust" in summary["actions"]
    assert summary["physics_dt"] == 0.001 and summary["decision_dt"] == 0.01


def test_terminal_inside_decision_freezes_physics_and_history():
    c = config()
    env = CTBRBatch(c)
    env.plant.pursuer.vel[:, 0] = 40
    _, _, done, _, records = env.step(torch.zeros(2, 4))
    assert bool(done.all()) and env.last_physics_steps == 2
    assert env.steps.tolist() == [1, 1]
    assert all(r["time"] == pytest.approx(0.001) for r in records)
    previous = {k: getattr(env.plant.pursuer, k).clone() for k in ("pos", "vel", "rot")}
    history = env.target_history.clone()
    obs = env.observe()
    _, reward, done, _, records = env.step(torch.ones(2, 4) * 0.5)
    assert not bool(done.any()) and not records and env.last_physics_steps == 0
    assert torch.equal(reward, torch.zeros(2))
    assert torch.equal(history, env.target_history)
    assert all(
        torch.equal(v, getattr(env.plant.pursuer, k)) for k, v in previous.items()
    )
    assert all(torch.equal(a, b) for a, b in zip(obs, env.observe()))


def test_non_aligned_deadline_and_mc_return_remain_exact():
    c = config()
    c["task"].update(duration=0.123, tail_seconds=0.05, distance=[0, 0])
    model = new_model(c)
    zero_actor(model)
    report, _ = evaluate(model, c, 2, 987)
    gamma = math.exp(-0.001 / 20)
    expected = 0.1 * (
        0.001 * 2.6 * sum(gamma**k for k in range(123)) + 3 + 5 * gamma**122
    )
    assert report["mean_mc_return"] == pytest.approx(expected, rel=3e-6)
    assert report["decision_hz"] == 100 and report["physics_hz"] == 1000
    assert all(
        e["time"] == pytest.approx(0.123) and e["tail_steps_observed"] == 50
        for e in report["episodes_detail"]
    )


def test_history_reset_is_selective_and_contains_only_observed_states():
    env = CTBRBatch(config())
    env.step(torch.zeros(2, 4))
    env.step(torch.zeros(2, 4))
    previous = env.target_history[1].clone()
    env.reset(torch.tensor([True, False]))
    assert int(env.target_history[0, :, 9].sum()) == 1
    assert torch.equal(previous, env.target_history[1])
    obs, priv = env.observe()
    env.desired_acc.fill_(7)
    env.command_velocity.fill_(10)
    env.ou.fill_(3)
    changed, _ = env.observe()
    assert torch.equal(obs, changed)
    assert not torch.equal(priv, env.observe()[1])


def test_forecaster_is_translation_invariant_with_nonzero_lstm_weights():
    model = new_model(config())
    history = torch.randn(3, 8, 10)
    history[:, :, 9] = 1
    with torch.no_grad():
        model.predictor.head.weight.normal_(0, 0.1)
        expected = model.predictor(history)
        shifted = history.clone()
        shifted[:, :, :3] += 100
        assert torch.allclose(expected, model.predictor(shifted), atol=2e-6)


def test_supervised_targets_do_not_cross_resets_or_bootstrap_future():
    history = torch.zeros(5, 1, 2, 10)
    history[:, 0, -1, 0] = torch.arange(5) * 3
    history[:, :, -1, 9] = 1
    positions = torch.zeros(5, 1, 3)
    positions[:, 0, 0] = torch.arange(1, 6) * 3
    done = torch.tensor([[False], [True], [False], [False], [False]])
    x, y = supervised_windows(history, positions, done, [1, 2], 100)
    assert x[:, -1, 0].tolist() == [6, 9]
    assert y[:, :, 0].tolist() == [[3, 6], [3, 6]]
    assert supervised_windows(history[:1], positions[:1], done[:1], [2], 100) is None


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_predictor_is_frozen_during_ppo_and_has_separate_optimizer(device):
    c = config()
    c["device"] = device
    model = new_model(c)
    buffer, positions = rollout(model, c)
    original = copy.deepcopy(model.predictor.state_dict())
    stats = model.update(buffer)
    assert stats["diagnostics"]["pre_update_logprob_error_max"] < 1e-4
    assert all(
        torch.equal(value, model.predictor.state_dict()[k])
        for k, value in original.items()
    )
    actor = copy.deepcopy(model.actor.state_dict())
    x = buffer.obs[:, :, observation_dims(c)[0] :].reshape(24, 2, 8, 10)
    data = supervised_windows(x, positions, buffer.terminated, [2, 4, 8], 128)
    data = (data[0], data[1] + 0.1)
    fit = fit_predictor(model, data)
    assert fit["samples"] > 0 and fit["fit_rmse_after_m"] < fit["fit_rmse_before_m"]
    assert any(
        not torch.equal(value, model.predictor.state_dict()[k])
        for k, value in original.items()
    )
    assert all(
        torch.equal(value, model.actor.state_dict()[k]) for k, value in actor.items()
    )


def test_value_norm_debiases_moments_and_roundtrips_raw_units():
    normalizer = ValueNorm(beta=0.995)
    values = torch.tensor([1.0, 3.0])
    normalizer.update(values)
    mean, std = normalizer.statistics()
    assert float(mean) == pytest.approx(2) and float(std) == pytest.approx(1)
    assert torch.allclose(normalizer.normalize(values), torch.tensor([-1.0, 1.0]))
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(values)), values)
    normalizer.update(values)
    assert float(normalizer.statistics()[0]) == pytest.approx(2)


def test_prediction_fit_guard_restores_damaging_candidate(monkeypatch):
    model = new_model(config())
    before = copy.deepcopy(model.predictor.state_dict())
    history = torch.zeros(10, 8, 10)
    history[:, -1, 9] = 1

    def damage():
        with torch.no_grad():
            model.predictor.head.bias.fill_(100)

    monkeypatch.setattr(model.predictor_optimizer, "step", damage)
    report = fit_predictor(model, (history, torch.zeros(10, 3, 3)))
    assert not report["accepted"] and report["updates"] == 0
    assert report["attempted_updates"] == 1
    assert all(
        torch.equal(v, model.predictor.state_dict()[k]) for k, v in before.items()
    )


def test_updating_value_statistics_preserves_raw_values(small):
    model = new_model(small)
    x = torch.randn(20, 110)
    before = model.values_tensor(x)
    model.update_value_statistics(torch.tensor([100.0, 200.0, 300.0]))
    assert torch.allclose(before, model.values_tensor(x), atol=3e-5)
    model.update_value_statistics(torch.tensor([-1000.0, 3000.0]))
    # Float32 output-head rebasing at thousand-unit scale has finite rounding.
    assert torch.allclose(before, model.values_tensor(x), atol=2e-4)


def test_checkpoint_restores_prediction_value_norm_and_all_optimizers(tmp_path):
    c = config()
    model = new_model(c)
    buffer, _ = rollout(model, c)
    model.update(buffer)
    h = buffer.obs[0, :, observation_dims(c)[0] :].reshape(2, 8, 10)
    fit_predictor(model, (h, torch.ones(2, 3, 3)))
    p = tmp_path / "model.pt"
    save(p, model, c, {}, "test")
    restored, _, saved = restore(p, optimizers=True)
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
    )
    assert restored.predictor_optimizer.state_dict()["state"]
    assert restored.critic_optimizer.state_dict()["state"]
    saved["schema"] = "pursuit-direct-ctbr-checkpoint-v2"
    torch.save(saved, p)
    with pytest.raises(ValueError, match="incompatible"):
        restore(p)


def test_continuous_mc_updates_only_critic_and_uses_identical_holdout(tmp_path):
    c = config()
    model = new_model(c)
    actor = copy.deepcopy(model.actor.state_dict())
    predictor = copy.deepcopy(model.predictor.state_dict())
    progress = dict(
        ppo_updates=2,
        physics_steps=1280,
        mc_physics_steps=0,
        mc_fit_updates=0,
        mc_calibration_rounds=0,
        last_mc_calibration_update=0,
    )
    report = calibrate_critic(model, c, tmp_path, progress)
    assert report["train_seed"] != report["heldout_seed"]
    assert report["before"]["samples"] == report["candidate_after"]["samples"]
    assert all(torch.equal(v, model.actor.state_dict()[k]) for k, v in actor.items())
    assert all(
        torch.equal(v, model.predictor.state_dict()[k]) for k, v in predictor.items()
    )
    assert progress["mc_fit_updates"] == 4 and progress["mc_calibration_rounds"] == 1
    assert progress["mc_physics_steps"] > 0


def test_rejected_mc_correction_restores_normalizer_and_optimizer(
    tmp_path, monkeypatch
):
    c = config()
    model = new_model(c)
    original = copy.deepcopy(model.state_dict())

    def damage(model, data, updates, size):
        model.update_value_statistics(torch.tensor([100.0, 200.0]))
        with torch.no_grad():
            model.critic[-1].bias.fill_(1e5)
        return dict(updates=updates)

    monkeypatch.setattr("pursuit_rl.ctbr.value_learning.fit_critic", damage)
    progress = dict(
        ppo_updates=1,
        physics_steps=640,
        mc_physics_steps=0,
        mc_fit_updates=0,
        mc_calibration_rounds=0,
        last_mc_calibration_update=0,
    )
    report = calibrate_critic(model, c, tmp_path, progress)
    assert not report["accepted"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert not model.critic_optimizer.state_dict()["state"]


def test_task_mixture_collects_fresh_current_recent_anchor_transitions():
    c = config()
    c["num_envs"] = 10
    recent, anchor = copy.deepcopy(c), copy.deepcopy(c)
    recent["task"].update(name="recent", duration=0.2, tail_seconds=0.1)
    anchor["task"].update(name="anchor", duration=0.1, tail_seconds=0.05)
    env = TaskMixtureBatch(c, recent, anchor, 42)
    assert env.task_counts == {"hover_small": 6, "recent": 2, "anchor": 2}
    action = torch.zeros(10, 4)
    for tick in range(10):
        result = env.step(action, action)
    obs, reward, done, trunc, records = result
    assert len(records) == 2 and all(
        r["task"] == "anchor" and r["slot"] >= 8 for r in records
    )
    assert env.last_physics_steps == 100 and env.target_positions.shape == (10, 3)
    env.reset(done)
    assert env.steps.tolist() == [100] * 8 + [0, 0]


def test_curriculum_changes_one_axis_after_foundations_and_keeps_acceptance():
    plan = json.loads(Path("configs/ctbr/plan.json").read_text())
    configs = [
        load(Path("configs/ctbr") / phase["profile"]) for phase in plan["phases"]
    ]
    for previous, current, phase in zip(configs[3:], configs[4:], plan["phases"][4:]):
        keys = [
            key
            for key in previous["task"]
            if key != "name" and previous["task"][key] != current["task"][key]
        ]
        assert keys == [phase["difficulty_axis"]]
    for c in configs[3:]:
        assert c["task"]["radius"] == 2 and c["task"]["tail_seconds"] == 10
        assert c["task"]["minimum_tail_fraction"] == 0.9
        assert (
            c["task"]["response_time_s"]
            >= 2 * c["task"]["max_accel"] / c["task"]["jerk_limit"]
        )


def test_training_100hz_predictor_continuous_mc_and_resume(tmp_path):
    c = config()
    root = tmp_path / "run"
    result = train(c, root)
    assert result["physics_steps"] == 2560 and result["decision_transitions"] == 256
    assert result["prediction_updates"] == 4 and result["mc_calibration_rounds"] == 2
    assert result["decision_hz"] == 100 and result["completed_episodes"] == 6
    c["training"]["physics_steps"] = 3200
    resumed = train(c, root, resume=root / "checkpoints/latest.pt")
    assert resumed["ppo_updates"] == 5 and resumed["prediction_updates"] == 5
    assert resumed["mc_calibration_rounds"] == 3
    assert not (root / "first_anomaly.pt").exists()
    report, _ = evaluate(restore(root / "checkpoints/latest.pt")[0], c, 2, 121000)
    assert report["forecast"]["samples_by_horizon"][0] > 0


def test_early_promotion_stops_and_curriculum_resume_does_not_retrain(small, tmp_path):
    c = copy.deepcopy(small)
    c["training"]["physics_steps"] = 2048
    root = tmp_path / "early"

    def ready(model, config, progress):
        return progress["physics_steps"] >= 480

    result = train(c, root, promotion_check=ready)
    assert result["physics_steps"] == 512 and result["promotion_passed"]
    again = train(c, root, resume=root / "checkpoints/latest.pt", promotion_check=ready)
    assert result == again


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("policy", "decision_steps", 0),
        ("prediction", "horizons_s", [0.015]),
        ("critic", "norm_beta", 1),
        ("task", "response_time_s", 0.0001),
        ("training", "replay_recent_fraction", 0.9),
    ],
)
def test_new_configuration_rejects_invalid_time_and_value_settings(section, key, value):
    c = config()
    c[section][key] = value
    with pytest.raises(ValueError):
        validate(c)
