"""Control units, stochastic likelihoods, episode boundaries and independent execution."""

import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from pursuit_rl.config.rescue import load_rescue_profile, rescue_contract
from pursuit_rl.simulation.rescue_environment import RescueBatch
from pursuit_rl.simulation.rescue_control import action_to_ctbr, reference_action
from pursuit_rl.learning.rescue_checkpoint import (
    new_rescue_model,
    load_rescue_checkpoint,
)
from pursuit_rl.learning.buffers import TensorRolloutBuffer, RolloutBuffer
from pursuit_rl.evaluation.rescue import evaluate_rescue
from pursuit_rl.training.rescue import train_rescue

ROOT = Path(__file__).resolve().parents[1]


def settings(mode="velocity"):
    filename = "ctbr_1khz_bc.json" if mode == "ctbr" else "velocity_50hz.json"
    return load_rescue_profile(ROOT / "configs" / "rescue" / filename, device="cpu")


def zero_actor(model):
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()


def hover_case(interval=20, deadline=0.003):
    cfg, profile = settings()
    profile.update(num_envs=2, randomize=False, decision_interval_steps=interval)
    cfg["training"]["num_envs"] = 2
    cfg["curriculum"][0]["time"] = deadline
    for key in cfg["reward"]:
        if key.startswith("w_"):
            cfg["reward"][key] = 0.0
    cfg["reward"]["success_bonus"] = 0.0
    cfg["reward"]["failure_penalty"] = 0.0
    env = RescueBatch(cfg, profile)
    env.env.u.pos[:] = torch.tensor([0.0, 0.0, -100.0])
    env.env.u.vel.zero_()
    env.env.t.pos[:] = torch.tensor([20.0, 0.0, -100.0])
    env.env.t.vel.zero_()
    env.env.t.max_accel.zero_()
    return cfg, profile, env


@pytest.mark.parametrize(
    "mode,actor,critic,dimensions", [("velocity", 35, 43, 3), ("ctbr", 36, 44, 4)]
)
def test_action_observation_interfaces(mode, actor, critic, dimensions):
    cfg, profile = settings(mode)
    env = RescueBatch(cfg, profile, n=2)
    obs, priv = env.observe()
    assert obs.shape == (2, actor) and priv.shape == (2, critic)
    model = new_rescue_model(cfg, profile)
    action, raw, lp, values = model.sample_tensor(obs, priv)
    assert action.shape == (2, dimensions) and torch.equal(action, torch.tanh(raw))
    assert lp.shape == values.shape == (2,)


def test_high_level_decoder_uses_self_state_without_target_steering():
    cfg, profile, env = hover_case(deadline=10.0)
    action = torch.zeros(2, 3)
    first = action_to_ctbr(env.env, action, profile)
    assert torch.allclose(first, torch.zeros_like(first), atol=1e-6)
    env.env.t.pos += 500.0
    env.env.t.vel += 20.0
    env.env.t.acc += 7.0
    assert torch.equal(first, action_to_ctbr(env.env, action, profile))
    action[:, 0] = 0.5
    command = action_to_ctbr(env.env, action, profile)
    assert (
        command[:, 2] < 0
    ).all()  # positive NED x acceleration requires negative pitch rate
    assert (command.abs() <= 1).all()


def test_macro_reward_stops_at_first_true_deadline_and_caches_final_state():
    cfg, profile, env = hover_case()
    (final, priv), reward, term, trunc, records = env.step(torch.zeros(2, 3))
    expected = -profile["timeout_penalty"] * profile["reward_scale"] * env.gamma_tick**2
    assert torch.allclose(reward, torch.full_like(reward, expected), atol=1e-6)
    assert term.all() and not trunc.any() and len(records) == 2
    assert int(env.active_physics_steps) == 6
    assert all(
        r["reason"] == "timeout" and r["time"] == pytest.approx(0.003, abs=1e-7)
        for r in records
    )
    assert torch.allclose(final[:, 31], torch.zeros(2), atol=1e-6)
    (second, _), r2, t2, _, rec2 = env.step(torch.zeros(2, 3))
    assert torch.equal(final, second) and not r2.any() and not t2.any() and rec2 == []
    env.reset(torch.tensor([True, False]))
    obs, _ = env.observe()
    assert obs[0, 31] == 1 and obs[1, 31] == pytest.approx(0.0, abs=1e-6)
    assert not env.finished[0] and env.finished[1]


def test_macro_reward_equals_discounted_physics_rewards():
    cfg, p1, e1 = hover_case(interval=20, deadline=10.0)
    cfg["reward"]["w_time"] = 2.0
    p2 = copy.deepcopy(p1)
    p2["decision_interval_steps"] = 1
    e2 = RescueBatch(cfg, p2, n=2)
    for attr in (
        "pos",
        "vel",
        "rot",
        "rate",
        "thrust",
        "drag",
        "tau_rate",
        "tau_thrust",
        "mass_ratio",
    ):
        setattr(e2.env.u, attr, getattr(e1.env.u, attr).clone())
    for attr in ("pos", "vel", "acc", "max_accel", "mode"):
        setattr(e2.env.t, attr, getattr(e1.env.t, attr).clone())
    action = torch.zeros(2, 3)
    _, macro, _, _, _ = e1.step(action)
    summed = torch.zeros(2)
    for k in range(20):
        _, one, _, _, _ = e2.step(action)
        summed += e1.gamma_tick**k * one
    assert torch.allclose(macro, summed, atol=1e-7)
    assert e1.gamma == pytest.approx(e1.gamma_tick**20)
    assert e1.gamma * e1.gae_lambda == pytest.approx(math.exp(-0.02 / 3.0))


def test_current_thrust_and_previous_raw_are_observed_and_reset():
    cfg, p = settings("ctbr")
    env = RescueBatch(cfg, p, n=2)
    raw = torch.tensor([[0.2, -0.1, 0.3, 0.1], [0.1, 0.2, 0.0, -0.4]])
    (obs, _), _, _, _, _ = env.step(torch.tanh(raw), raw)
    assert torch.equal(obs[:, 32:], raw)
    assert torch.allclose(obs[:, 24], env.env.u.thrust / cfg["physics"]["hover_thrust"])
    env.reset(torch.tensor([True, False]))
    obs, _ = env.observe()
    assert not obs[0, 32:].any() and torch.equal(obs[1, 32:], raw[1])


def test_ar_likelihood_matches_the_actual_conditional_sampler():
    cfg, p = settings("ctbr")
    model = new_rescue_model(cfg, p)
    zero_actor(model)
    obs = torch.zeros(5, 36)
    obs[:, 32:] = torch.tensor([0.7, -0.4, 0.2, 0.1])
    priv = torch.zeros(5, 44)
    action, raw, lp, _ = model.sample_tensor(obs, priv)
    expected_mean = model.rho * obs[:, 32:]
    expected_std = math.sqrt(1 - model.rho**2) * p["initial_action_std"]
    jacobian = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
    expected = (
        torch.distributions.Normal(expected_mean, expected_std).log_prob(raw) - jacobian
    ).sum(-1)
    assert torch.allclose(lp, expected, atol=2e-5, rtol=0)
    assert torch.equal(action, raw.tanh())
    assert model.rho == pytest.approx(math.exp(-0.001 / 0.02))


def test_ar_noise_has_configured_marginal_scale_and_physical_correlation():
    cfg, p = settings("ctbr")
    model = new_rescue_model(cfg, p)
    zero_actor(model)
    torch.manual_seed(731)
    obs = torch.zeros(64, 36)
    priv = torch.zeros(64, 44)
    history = []
    for k in range(1600):
        _, raw, _, _ = model.sample_tensor(obs, priv)
        obs[:, 32:] = raw
        if k >= 200:
            history.append(raw[:, 0].clone())
    noise = torch.stack(history)
    corr = np.corrcoef(noise[:-1].reshape(-1).numpy(), noise[1:].reshape(-1).numpy())[
        0, 1
    ]
    assert float(noise.std()) == pytest.approx(p["initial_action_std"], rel=0.06)
    assert corr == pytest.approx(model.rho, abs=0.01)


@pytest.mark.parametrize("mode", ["velocity", "ctbr"])
def test_ppo_recomputes_collected_likelihood_and_updates_both_networks(mode):
    cfg, p = settings(mode)
    p.update(ppo_epochs=1, ppo_batch_size=24)
    env = RescueBatch(cfg, p, n=3)
    model = new_rescue_model(cfg, p)
    obs, priv = env.observe()
    buf = TensorRolloutBuffer(
        8, 3, env.actor_dim, env.critic_dim, "cpu", action_dim=env.action_dim
    )
    for k in range(8):
        action, raw, lp, values = model.sample_tensor(obs, priv)
        buf.obs[k] = obs
        buf.priv[k] = priv
        buf.raw[k] = raw
        buf.logprob[k] = lp
        buf.values[k] = values
        (obs, priv), reward, term, trunc, _ = env.step(action, raw)
        buf.reward[k] = reward
        buf.terminated[k] = term
        buf.truncated[k] = trunc
        buf.next_values[k] = model.values_tensor(priv)
    buf.compute_gae(env.gamma, env.gae_lambda)
    actor = [x.detach().clone() for x in model.actor.parameters()]
    critic = [x.detach().clone() for x in model.critic.parameters()]
    stats = model.update(buf)
    assert stats["diagnostics"]["pre_update_logprob_error_max"] < 1e-5
    assert any(not torch.equal(a, b) for a, b in zip(actor, model.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic, model.critic.parameters()))


def test_kl_stop_retains_critic_updates():
    cfg, p = settings()
    p.update(ppo_epochs=2, ppo_batch_size=8, ppo_target_kl=1e-8)
    model = new_rescue_model(cfg, p)
    buf = TensorRolloutBuffer(
        8, 2, 35, 43, "cpu", action_dim=3, diagnostics={"enabled": False}
    )
    buf.obs.zero_()
    buf.priv.zero_()
    buf.raw.zero_()
    buf.values.zero_()
    buf.next_values.zero_()
    buf.reward.fill_(1.0)
    with torch.no_grad():
        mean, std = model.distribution_parameters(buf.obs.reshape(-1, 35))
        buf.logprob.copy_(
            model.logprob(mean, std, buf.raw.reshape(-1, 3)).reshape(8, 2) - 1.0
        )
    buf.compute_gae(0.99, 0.95)
    actor = [x.detach().clone() for x in model.actor.parameters()]
    critic = [x.detach().clone() for x in model.critic.parameters()]
    stats = model.update(buf)
    assert stats["early_stop"] and stats["updates"] == 4
    assert all(torch.equal(a, b) for a, b in zip(actor, model.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic, model.critic.parameters()))


def test_mc_targets_ignore_bad_bootstrap_and_restore_training_rng():
    cfg, p, _ = hover_case(interval=20, deadline=0.003)
    p["num_envs"] = 2
    model = new_rescue_model(cfg, p)
    zero_actor(model)
    with torch.no_grad():
        for parameter in model.critic.parameters():
            parameter.zero_()
        model.critic[-1].bias.fill_(1000.0)
    rng = torch.get_rng_state().clone()
    report, data = evaluate_rescue(
        model, cfg, p, 2, 731, stochastic=True, collect_mc=True
    )
    assert torch.equal(torch.get_rng_state(), rng)
    expected = (
        -p["timeout_penalty"]
        * p["reward_scale"]
        * math.exp(-0.002 / p["discount_time_s"])
    )
    assert torch.allclose(data[1], torch.full_like(data[1], expected), atol=1e-6)
    assert report["critic_rmse"] > 1000 and report["successes"] == 0


def test_three_dimensional_buffer_diagnostics_do_not_report_thrust_mapping():
    for cls, args in ((RolloutBuffer, ()), (TensorRolloutBuffer, ("cpu",))):
        buf = cls(2, 1, 35, 43, *args, action_dim=3)
        for name in ("obs", "priv", "raw"):
            getattr(buf, name)[...] = 0
        buf.physics = {"dt": 0.02}
        buf.action_semantics = "velocity"
        buf.diagnostic_steps = 2
        buf.compute_gae(0.99, 0.95)
        report = buf.diagnostic_summary()
        assert len(report["actions"]["normalized"]) == 3
        assert "command_thrust" not in report["actions"]


def test_training_evaluation_checkpoint_and_resume_without_teacher_at_runtime(
    tmp_path, monkeypatch
):
    import pursuit_rl.training.rescue as training
    import pursuit_rl.evaluation.rescue as evaluation

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected teacher call during learned execution")

    monkeypatch.setattr(training, "reference_action", forbidden)
    monkeypatch.setattr(evaluation, "reference_action", forbidden)
    cfg, p, _ = hover_case(interval=2, deadline=0.006)
    p.update(
        num_envs=2,
        bc_updates=0,
        dagger_rounds=0,
        bc_min_success=0.0,
        validation_episodes=2,
        critic_mc_episodes=2,
        critic_pretrain_updates=2,
        critic_reanchor_updates=2,
        budget_physics_steps=16,
        validation_every_physics_steps=8,
        rollout_decisions=4,
        hidden_actor=[8],
        hidden_critic=[8],
        ppo_epochs=1,
        ppo_batch_size=8,
    )
    out = tmp_path / "rescue"
    summary = train_rescue(cfg, p, out)
    assert summary["physics_steps"] == 16 and summary["ppo_updates"] == 1
    checkpoint = out / "checkpoints" / "latest.pt"
    model, c2, p2, saved = load_rescue_checkpoint(checkpoint, device="cpu")
    assert saved["origin"] == "ppo"
    report, _ = evaluate_rescue(model, c2, p2, 2, 821)
    assert report["controller"] == "learned_actor" and report["failures"] == {
        "timeout": 2
    }
    wrong = rescue_contract(cfg, p)
    wrong["action_mode"] = "ctbr"
    with pytest.raises(ValueError, match="Wrong action"):
        load_rescue_checkpoint(checkpoint, expected_contract=wrong)
    with pytest.raises(ValueError, match="not a rescue"):
        torch.save({"model": {}}, tmp_path / "legacy.pt")
        load_rescue_checkpoint(tmp_path / "legacy.pt")
    p["budget_physics_steps"] = 32
    summary = train_rescue(cfg, p, out, resume=checkpoint)
    assert summary["physics_steps"] == 32 and summary["ppo_updates"] == 2
    records = [json.loads(s) for s in (out / "metrics.jsonl").read_text().splitlines()]
    assert any(s["event"] == "resume" for s in records)


def test_finite_timeout_zeros_potential_endpoint():
    cfg, profile, env = hover_case()
    profile["timeout_penalty"] = 0.0
    cfg["reward"]["w_cpa"] = 1.0
    cfg["reward"]["cpa_mode"] = "potential"
    env.env.last_cpa_potential.fill_(-20 / cfg["reward"]["distance_scale"])
    _, reward, term, _, _ = env.step(torch.zeros(2, 3))
    expected = profile["reward_scale"] * 20 / cfg["reward"]["distance_scale"]
    assert term.all() and torch.allclose(
        reward, torch.full_like(reward, expected), atol=1e-6
    )


def test_anomaly_preserves_pre_update_network_for_likelihood_replay(
    tmp_path, monkeypatch
):
    from pursuit_rl.learning.ppo import AsymmetricPPO

    original = AsymmetricPPO.update

    def flag(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result["diagnostics"]["anomaly_reasons"].append("forced_test")
        return result

    monkeypatch.setattr(AsymmetricPPO, "update", flag)
    cfg, p, _ = hover_case(interval=2, deadline=0.006)
    p.update(
        num_envs=2,
        bc_updates=0,
        dagger_rounds=0,
        bc_min_success=0.0,
        validation_episodes=2,
        critic_mc_episodes=2,
        critic_pretrain_updates=2,
        critic_reanchor_updates=2,
        budget_physics_steps=16,
        validation_every_physics_steps=8,
        rollout_decisions=4,
        hidden_actor=[8],
        hidden_critic=[8],
        ppo_epochs=1,
        ppo_batch_size=8,
    )
    out = tmp_path / "rescue"
    train_rescue(cfg, p, out)
    packet = torch.load(
        out / "first_anomaly.pt", map_location="cpu", weights_only=False
    )
    pre = new_rescue_model(cfg, p)
    pre.load_state_dict(packet["before_update"]["model"])
    fields = packet["buffer"]["tensors"]
    obs = fields["obs"].reshape(-1, 35)
    raw = fields["raw"].reshape(-1, 3)
    with torch.no_grad():
        mean, std = pre.distribution_parameters(obs)
        error = (
            (pre.logprob(mean, std, raw) - fields["logprob"].reshape(-1)).abs().max()
        )
    assert error < 1e-6 and packet["before_update"]["progress"]["ppo_updates"] == 0
    assert packet["progress"]["ppo_updates"] == 1
    assert packet["before_update"]["actor_optimizer"]["state"] == {}
