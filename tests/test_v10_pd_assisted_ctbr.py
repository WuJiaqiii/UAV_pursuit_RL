"""v10 direct-CTBR executed-action imitation, fair PD and safety regressions."""
import json
import math
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, new_model, save
from pursuit_rl.ctbr.config import fingerprint, load, validate
from pursuit_rl.ctbr.control import observed_pd, teacher
from pursuit_rl.ctbr.environment import CTBRBatch, tilt_guard_cost
from pursuit_rl.ctbr.training import fit_actor, imitation_data, train


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(17)
    torch.set_num_threads(1)


def config():
    c = load("configs/first_entry_highspeed/01_static.json", device="cpu")
    c["num_envs"] = 4
    c["prediction"]["enabled"] = False
    c["training"]["teacher_method"] = "pd_observed"
    return validate(c)


def test_observed_pd_excludes_privileged_random_parameters():
    c = config()
    env = CTBRBatch(c)
    before_obs, _ = env.observe()
    baseline = observed_pd(env)
    env.plant.pursuer.mass_ratio *= 1.12
    env.plant.pursuer.drag *= 1.35
    after_obs, _ = env.observe()
    assert torch.allclose(before_obs, after_obs)
    assert torch.allclose(baseline, observed_pd(env), atol=1e-6)
    assert torch.allclose(baseline, teacher(env), atol=1e-6)


def test_observed_pd_smoothing_matches_actor_filter():
    c = config()
    env = CTBRBatch(c)
    target = observed_pd(env)
    env.config["training"]["teacher_method"] = "pd_observed_smoothed"
    smoothed = teacher(env)
    rho = math.exp(-env.decision_dt / c["policy"]["exploration_time_s"])
    assert torch.allclose(
        smoothed, torch.tanh((1 - rho) * torch.atanh(target)), atol=1e-6
    )


def test_soft_tilt_guard_warns_before_hard_boundary():
    radians = torch.tensor([40.0, 60.0, 68.0]) * (math.pi / 180)
    from pursuit_rl.simulation.rigid_body import exp_so3
    vectors = torch.stack((radians, torch.zeros_like(radians),
                           torch.zeros_like(radians)), 1)
    angles = tilt_guard_cost(radians, exp_so3(vectors),
                             torch.zeros(3, 3), 70.0, 55.0, 0.12)
    assert angles[0] == 0
    assert 0 < angles[1] < angles[2] < 1
    c = config()
    c["reward"]["tilt_guard_start_deg"] = 71.0
    with pytest.raises(ValueError, match="tilt guard"):
        validate(c)


def test_bc_trains_filtered_executed_actions_from_full_observation():
    c = config()
    model = new_model(c)
    n = 160
    obs = torch.randn(n, model.obs_dim) * 0.1
    obs[:, 21:25] *= 3
    obs[:, 32:36] = torch.randn(n, 4) * 0.4
    target = torch.tanh(0.5 * obs[:, 21:25] + 0.2)
    with torch.no_grad():
        before = (model.predict_tensor(obs) - target).square().mean()
    fit_actor(model, (obs, target), updates=110, batch_size=64)
    with torch.no_grad():
        after = (model.predict_tensor(obs) - target).square().mean()
    assert after < before * 0.8
    assert model.actor[0].weight[:, 21:].abs().max() > 1e-5


def test_dagger_keeps_actuator_and_history_features():
    c = config()
    c["training"]["bc_samples"] = 128
    states, labels, physics_steps = imitation_data(c, model=new_model(c), seed=381)
    assert states.shape == (128, 34 + 4*c["policy"]["history_steps"])
    assert labels.shape == (128, 4)
    assert physics_steps > 0
    assert states[:, 21:].abs().sum() > 0


def test_ppo_auxiliary_loss_receives_teacher_labels(tmp_path):
    c = config()
    c["task"].update(duration=0.03, distance=[0.65, 0.75])
    c["training"].update(
        physics_steps=80, rollout_steps=2, batch_size=8, epochs=1,
        validation_every=80, validation_episodes=2, mc_episodes=2,
        critic_updates=0, mc_calibration_every_updates=0,
        aux_pd_coef=0.02, prediction_updates_enabled=False,
    )
    train(validate(c), tmp_path / "aux_run")
    rows = [
        json.loads(line) for line in
        (tmp_path / "aux_run/metrics.jsonl").read_text().splitlines()
    ]
    updates = [r for r in rows if r["event"] == "ppo_update"]
    assert updates
    assert updates[0]["metrics"]["aux_pd_coef"] == pytest.approx(0.02)
    assert updates[0]["metrics"]["aux_pd_loss"] >= 0


def test_old_checkpoint_fingerprint_still_loads(tmp_path):
    c = config()
    filename = tmp_path / "legacy.pt"
    save(filename, new_model(c, optimizers=False), c, {
        "physics_steps": 0, "ppo_updates": 0
    }, "test")
    data = torch.load(filename, weights_only=False)
    data["config"]["training"].pop("aux_pd_coef")
    for key in ("tilt_guard", "tilt_guard_start_deg", "tilt_guard_lookahead_s"):
        data["config"]["reward"].pop(key)
    data["fingerprint"] = fingerprint(data["config"])
    torch.save(data, filename)
    _, restored, _ = load_checkpoint(filename)
    assert restored["training"]["aux_pd_coef"] == 0
    assert restored["reward"]["tilt_guard"] == 0


def test_matched_pd_auxiliary_experiment_plan():
    from scripts.run_generalization_campaign import model_configs
    plan = json.loads(Path("configs/experiments/06_v10_pd_assisted_ctbr.json").read_text())
    assert plan["teacher_method"] == "pd_observed"
    assert len(plan["variants"]) == 2
    arms = plan["variants"]
    assert arms[0]["reward"] == arms[1]["reward"]
    assert [v["training"]["aux_pd_coef"] for v in arms] == [0, 0.02]
    for arm in arms:
        primary, other, all_profiles = model_configs(
            config(), plan, arm, "pilot", 42, "cpu"
        )
        assert primary["reward"]["tilt_guard"] == 3.0
        assert len(other) == 3 and len(all_profiles) == 4
