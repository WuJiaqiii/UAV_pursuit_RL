import copy
import json
from pathlib import Path
import math
import pytest
import torch
from pursuit_rl.ctbr.config import load, validate, interface
from pursuit_rl.ctbr.control import thrust_from_action, thrust_to_action, teacher
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.checkpoint import new_model, save, load as load_model
from pursuit_rl.ctbr.evaluation import evaluate, improved, score
from pursuit_rl.ctbr.training import train
from pursuit_rl.ctbr.curriculum import run
from pursuit_rl.learning.buffers import TensorRolloutBuffer


def test_all_profiles_and_single_clock():
    for p in Path("configs/ctbr").glob("*.json"):
        if p.stem == "plan":
            continue
        c = load(p)
        env = CTBRBatch(c, n=2)
        assert env.actor_dim == 32 + 4 * c["policy"]["history_steps"] + 2
        assert env.critic_dim == env.actor_dim + 44
        assert env.dt == 0.001
        assert math.isclose(env.gamma * env.gae_lambda, math.exp(-0.001 / 3))


def test_strict_config(small, tmp_path):
    for field, value in [("dt", 0.01), ("substeps", 2), ("tau_rate", 0)]:
        c = copy.deepcopy(small)
        c["physics"][field] = value
        with pytest.raises(ValueError):
            validate(c)
    c = copy.deepcopy(small)
    c["task"]["duration"] = 0.1204
    with pytest.raises(ValueError):
        validate(c)
    p = tmp_path / "bad.json"
    p.write_text('{"training":{"rolluot_steps":5}}')
    with pytest.raises(ValueError, match="Unknown"):
        load(p)


def test_ctbr_endpoints_no_zero_thrust_deadzone(small):
    p = small["physics"]
    u = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    thrust = thrust_from_action(u, p)
    assert torch.allclose(thrust, torch.tensor([0.0, 4.905, 9.81, 19.905, 30.0]))
    assert torch.allclose(thrust_to_action(thrust, p), u)
    assert thrust_from_action(torch.tensor(-0.9), p) > 0


def test_actuator_response_and_ned_hover(small):
    env = CTBRBatch(small)
    u = env.plant.pursuer
    start = u.pos.clone()
    env.step(torch.zeros(env.n, 4))
    assert torch.equal(u.pos, start)
    action = torch.zeros(env.n, 4)
    action[:, 0] = 1
    action[:, 1] = 1
    env.step(action)
    p = small["physics"]
    expected_thrust = 9.81 + (30 - 9.81) * (1 - math.exp(-0.001 / 0.1))
    expected_rate = 1.2 * (1 - math.exp(-0.001 / 0.13))
    assert torch.allclose(u.thrust, torch.full((env.n,), expected_thrust))
    assert torch.allclose(u.rate[:, 0], torch.full((env.n,), expected_rate))
    assert bool((u.acc[:, 2] < 0).all())
    assert env.steps.tolist() == [2, 2]


def test_acquisition_is_not_terminal_and_frozen_completed_state(small):
    env = CTBRBatch(small)
    zeros = torch.zeros(env.n, 4)
    (obs, priv), reward, done, trunc, records = env.step(zeros)
    assert not bool(done.any()) and not records
    assert bool((env.first_acquisition == 0.001).all())
    assert torch.allclose(reward, sum(env.reward_components.values()))
    for _ in range(env.max_steps - 1):
        result = env.step(zeros)
    (last, critic), reward, done, trunc, records = result
    assert bool(done.all()) and not bool(trunc.any())
    assert all(r["success"] and r["time"] == pytest.approx(0.12) for r in records)
    assert all(r["tail_within_fraction"] == 1.0 for r in records)
    frozen, reward, done, _, records = env.step(zeros)
    assert torch.equal(frozen[0], last) and torch.equal(frozen[1], critic)
    assert (
        not bool(done.any())
        and not records
        and torch.equal(reward, torch.zeros_like(reward))
    )


def test_first_entry_does_not_imply_tail_success(small):
    env = CTBRBatch(small)
    zeros = torch.zeros(env.n, 4)
    env.step(zeros)
    env.plant.target.pos[:, 0] += 5
    for _ in range(env.max_steps - 1):
        result = env.step(zeros)
    assert all(r["reason"] == "tracking_failed" for r in result[-1])
    assert all(r["first_acquisition_time"] is not None for r in result[-1])


def test_terminal_failure_priority_and_invalid_target(small):
    env = CTBRBatch(small)
    env.steps.fill_(env.max_steps - 1)
    env.plant.pursuer.vel[:, 0] = 40
    records = env.step(torch.zeros(env.n, 4))[-1]
    assert all(r["reason"] == "speed" and not r["success"] for r in records)
    env = CTBRBatch(small)
    env.plant.target.pos[:, 0] = small["physics"]["workspace_xy"]
    assert all(
        r["reason"] == "invalid_target" for r in env.step(torch.zeros(env.n, 4))[-1]
    )


def test_actor_has_no_future_target_latent_information(small):
    env = CTBRBatch(small)
    a, v = env.observe()
    env.command_velocity.fill_(9)
    env.ou.fill_(4)
    env.desired_acc.fill_(3)
    env.timer.fill_(0.123)
    env.pattern.fill_(3)
    other, critic = env.observe()
    assert torch.equal(a, other) and not torch.equal(v, critic)


def test_random_target_is_reproducible_bounded_and_no_pursuer_pullback():
    c = load("configs/ctbr/10_random_high.json")
    a, b = CTBRBatch(c, n=4, seed=77), CTBRBatch(c, n=4, seed=77)
    b.plant.pursuer.pos.add_(10000)
    changed = False
    initial = a.pattern.clone()
    for _ in range(1200):
        old_acc = a.plant.target.acc.clone()
        a.advance_target()
        b.advance_target()
        assert bool(
            (
                torch.linalg.vector_norm(a.plant.target.acc - old_acc, dim=1) / 0.001
                <= 20.001
            ).all()
        )
        assert torch.equal(a.plant.target.pos, b.plant.target.pos)
        assert bool(
            (torch.linalg.vector_norm(a.plant.target.vel, dim=1) <= 10.0001).all()
        )
        assert bool((torch.linalg.vector_norm(a.plant.target.acc, dim=1) <= 5.02).all())
        changed = changed or not torch.equal(initial, a.pattern)
    assert changed


def test_gae_terminal_timeout_and_rollout_edge():
    b = TensorRolloutBuffer(3, 1, 2, 3, "cpu")
    b.reward[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    b.values.fill_(0.5)
    b.next_values.fill_(10.0)
    b.terminated[0] = True
    b.truncated[1] = True
    b.compute_gae(0.9, 0.8)
    assert b.bootstrap_values[:, 0].tolist() == [0.0, 10.0, 10.0]
    assert torch.allclose(b.advantage[:, 0], torch.tensor([0.5, 10.5, 11.5]))
    b.terminated.zero_()
    b.truncated.zero_()
    b.compute_gae(0.9, 0.8)
    assert b.advantage[1, 0] == pytest.approx(10.5 + 0.9 * 0.8 * 11.5)


def test_ar_likelihood_recomputes_with_observed_raw_history(small):
    env = CTBRBatch(small)
    model = new_model(small)
    obs, priv = env.observe()
    for _ in range(12):
        action, raw, lp, _ = model.sample_tensor(obs, priv)
        mean, std = model.distribution_parameters(obs)
        assert torch.allclose(model.logprob(mean, std, raw), lp, atol=1e-6)
        (obs, priv), _, _, _, _ = env.step(action, raw)
        assert torch.equal(obs[:, 32:36], raw)
    with pytest.raises(ValueError, match="disagree"):
        env.step(action, -raw)


def test_mc_returns_complete_episode_and_rng_pairing(small):
    model = new_model(small)
    # A precisely hovering deterministic policy, including AR memory.
    for parameter in model.actor.parameters():
        parameter.data.zero_()
    rng = torch.get_rng_state().clone()
    report, data = evaluate(model, small, 2, 999, collect_mc=True)
    assert torch.equal(torch.get_rng_state(), rng)
    again, _ = evaluate(model, small, 2, 999)
    assert report == again
    assert report["successes"] == 2
    gamma = math.exp(-0.001 / 20)
    steps = 120
    expected = 0.1 * (
        0.001 * (0.1 + 2.0 + 0.5) * sum(gamma**k for k in range(steps))
        + 3.0
        + 5.0 * gamma ** (steps - 1)
    )
    assert report["mean_mc_return"] == pytest.approx(expected, rel=2e-6)
    # Sparse MC samples include the initial state and the final 20 ms block.
    x, y = data
    for slot in range(2):
        initial = (x[:, 31] > 0.0019) & (x[:, 32:64].abs().sum(1) == 0)
        assert bool(torch.isclose(y[initial], torch.tensor(expected), atol=2e-6).any())
    assert not report["critic_matches_policy"]


def test_reference_is_causal_and_full_thrust_mapping(small):
    env = CTBRBatch(small)
    assert torch.allclose(teacher(env), torch.zeros(env.n, 4), atol=1e-6)
    env.plant.pursuer.mass_ratio.fill_(2)
    assert torch.allclose(teacher(env), torch.zeros(env.n, 4), atol=1e-6)


def test_no_false_best_from_floating_point_noise(small):
    model = new_model(small)
    r, _ = evaluate(model, small, 2, 9)
    baseline = score(r)
    perturbed = copy.deepcopy(r)
    for e in perturbed["episodes_detail"]:
        e["tail_position_rmse"] -= 1e-7
    assert not improved(perturbed, baseline)


def test_checkpoint_contract_rejects_legacy_and_tampering(small, tmp_path):
    model = new_model(small)
    p = tmp_path / "a.pt"
    save(p, model, small, {"x": 1}, "test")
    restored, c, data = load_model(p)
    assert c == small
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
    )
    data["config"]["physics"]["hover_thrust"] = 10.0
    torch.save(data, p)
    with pytest.raises(ValueError, match="contract"):
        load_model(p)
    torch.save({"model": model.state_dict()}, p)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(p)


def test_training_resume_and_transfer_end_to_end(small, tmp_path):
    c = copy.deepcopy(small)
    root = tmp_path / "run"
    result = train(c, root)
    assert result["ppo_updates"] == 2 and result["completed_episodes"] == 2
    assert result["physics_steps"] == 256 and result["unfinished_episodes"] == 2
    initial, _, _ = load_model(root / "checkpoints/initial.pt")
    latest, _, data = load_model(root / "checkpoints/latest.pt")
    assert any(
        not torch.equal(v, latest.actor.state_dict()[k])
        for k, v in initial.actor.state_dict().items()
    )
    metrics = [json.loads(x) for x in (root / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-2]["metrics"]["aux_pd_coef"] == 0
    assert all(
        r["diagnostics"]["pre_update_logprob_error_max"] < 1e-4
        for r in metrics
        if r["event"] == "ppo_update"
    )
    assert not (root / "first_anomaly.pt").exists()
    c["training"]["physics_steps"] = 384
    resumed = train(c, root, resume=root / "checkpoints/latest.pt")
    assert resumed["ppo_updates"] == 3 and resumed["physics_steps"] == 384
    target = copy.deepcopy(c)
    target["training"]["physics_steps"] = 0
    target["task"]["name"] = "transfer"
    new = train(target, tmp_path / "transfer", init_from=root / "checkpoints/best.pt")
    assert new["parent"] is not None and new["physics_steps"] == 0
    target["physics"]["tau_rate"] = 0.2
    with pytest.raises(ValueError, match="contract"):
        train(target, tmp_path / "bad_transfer", init_from=root / "checkpoints/best.pt")


def test_curriculum_rejects_skipped_prerequisites_and_short_budget(tmp_path):
    with pytest.raises(ValueError, match="prefix"):
        run("configs/ctbr/plan.json", tmp_path / "skip", phases=["10_random_high"])
    with pytest.raises(ValueError, match="two full"):
        run("configs/ctbr/plan.json", tmp_path / "short", steps=100)


def test_curriculum_two_stages_retention_and_restart(small, tmp_path):
    phases = []
    for index in range(2):
        name = f"stage_{index}"
        c = copy.deepcopy(small)
        c["task"]["name"] = name
        c["training"]["physics_steps"] = 512
        (tmp_path / f"{name}.json").write_text(json.dumps(c))
        phases.append(dict(name=name, profile=f"{name}.json", retention_anchor=True))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(dict(schema="pursuit-direct-ctbr-curriculum-v2", phases=phases))
    )
    result = run(plan, tmp_path / "experiment", episodes=2)
    assert len(result) == 2 and all(r["passed"] for r in result)
    assert len(result[-1]["retention"]) == 1
    hashes = [r["selected_checkpoint_sha256"] for r in result]
    again = run(plan, tmp_path / "experiment", episodes=2)
    assert [r["selected_checkpoint_sha256"] for r in again] == hashes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_sampling_and_environment(small):
    small["device"] = "cuda"
    env = CTBRBatch(small)
    model = new_model(small)
    obs, priv = env.observe()
    action, raw, _, _ = model.sample_tensor(obs, priv)
    (obs, priv), _, _, _, _ = env.step(action, raw)
    assert obs.device.type == "cuda"
