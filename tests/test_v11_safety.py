"""V11: safety predictor, cost constraints, legacy warm start and matched plan."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, new_model, save
from pursuit_rl.ctbr.config import load, validate, fingerprint, interface
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.safety import predict_attitude_risk
from pursuit_rl.ctbr.training import train
from scripts.run_generalization_campaign import model_configs


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(19)
    torch.set_num_threads(1)


def tiny(safety=True):
    c = load("configs/first_entry_highspeed/01_static.json", device="cpu")
    c["num_envs"] = 4
    c["prediction"]["enabled"] = False
    c["training"]["mc_episodes"] = 2
    c["training"]["validation_episodes"] = 2
    c["training"]["initialization_episodes"] = 2
    c["training"]["validation_every"] = 80
    c["training"]["physics_steps"] = 80
    c["training"]["rollout_steps"] = 2
    c["training"]["batch_size"] = 8
    c["training"]["epochs"] = 1
    c["training"]["critic_updates"] = 0
    c["training"]["mc_calibration_every_updates"] = 0
    c["training"]["prediction_updates_enabled"] = False
    c["safety"]["predictive_enabled"] = safety
    c["safety"]["constraint_enabled"] = safety
    c["reward"]["actuator_risk"] = 0.5 if safety else 0.0
    return validate(c)


def test_predictor_increases_risk_near_tilt_boundary():
    c = tiny()
    env = CTBRBatch(c)
    env.plant.pursuer.rot[:] = torch.eye(3)
    env.plant.pursuer.rate.zero_()
    action = torch.zeros(env.n, 4)
    safe, peak, margin = predict_attitude_risk(env, action)
    assert torch.all(safe == 0)
    assert torch.all(margin > 0)
    from pursuit_rl.simulation.rigid_body import exp_so3
    tilt = torch.full((env.n, 3), 0.0)
    tilt[:, 0] = math.radians(63.0)
    env.plant.pursuer.rot[:] = exp_so3(tilt)
    critical, peak_critical, margin_critical = predict_attitude_risk(env, action)
    assert torch.all(critical > safe)
    assert torch.all(peak_critical > peak)
    assert torch.all(margin_critical < margin)


def test_prediction_uses_current_rate_and_actuator_lag():
    c = tiny()
    env = CTBRBatch(c)
    from pursuit_rl.simulation.rigid_body import exp_so3
    tilt = torch.zeros(env.n, 3)
    tilt[:, 0] = math.radians(54)
    env.plant.pursuer.rot[:] = exp_so3(tilt)
    action = torch.zeros(env.n, 4)
    env.plant.pursuer.rate[:, 0] = 1.0
    risk_moving, _, _ = predict_attitude_risk(env, action)
    env.plant.pursuer.rate.zero_()
    risk_still, _, _ = predict_attitude_risk(env, action)
    assert torch.all(risk_moving > risk_still)
    env.plant.pursuer.rate[:, 0] = 1.0
    env.plant.pursuer.tau_rate[:] *= 1.5
    risk_slow, _, _ = predict_attitude_risk(env, action)
    assert torch.all(risk_slow >= risk_moving - 1e-7)


def test_critical_curriculum_produces_safe_but_nonlevel_states():
    c = tiny()
    c["safety"]["critical_reset_fraction"] = 0.5
    env = CTBRBatch(validate(c))
    assert env.critical_reset_fraction == 0.0
    env.critical_reset_fraction = 1.0
    env.reset()
    angles = torch.acos(env.plant.pursuer.rot[:, 2, 2].clamp(-1, 1))
    assert bool((angles > math.radians(30)).all())
    assert bool((angles < math.radians(c["physics"]["max_tilt_deg"])).all())
    assert bool((env.plant.pursuer.rate.norm(dim=1) > 0).all())


def test_legacy_model_without_safety_can_load_in_v11(tmp_path):
    c = tiny(False)
    filename = tmp_path / "legacy.pt"
    save(filename, new_model(c, optimizers=False), c,
         {"physics_steps": 0, "ppo_updates": 0}, "test")
    raw = torch.load(filename, weights_only=False)
    raw["config"].pop("safety")
    raw["config"]["reward"].pop("actuator_risk")
    raw["fingerprint"] = fingerprint(raw["config"])
    raw["interface"] = interface(raw["config"])
    torch.save(raw, filename)
    model, restored, _ = load_checkpoint(filename, device="cpu")
    assert model.cost_critic is None
    assert restored["safety"]["constraint_enabled"] is False


def test_cost_critic_is_saved_and_loaded_with_optimizer(tmp_path):
    c = tiny(True)
    model = new_model(c, optimizers=True)
    assert model.cost_critic is not None
    with torch.no_grad():
        model.lagrange_multiplier.fill_(0.4)
    filename = tmp_path / "safe.pt"
    save(filename, model, c, {"physics_steps": 80}, "test")
    restored, loaded, _ = load_checkpoint(filename, device="cpu", optimizers=True)
    assert loaded["safety"]["constraint_enabled"]
    assert restored.lagrange_multiplier.item() == pytest.approx(0.4)
    assert restored.cost_optimizer is not None


def test_tiny_constrained_training_has_cost_metrics(tmp_path):
    c = tiny(True)
    train(c, tmp_path / "run")
    rows = [
        json.loads(x) for x in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    ppo = [x for x in rows if x["event"] == "ppo_update"]
    assert ppo
    m = ppo[0]["metrics"]
    assert m["safety_enabled"] is True
    assert m["cost_critic_loss"] >= 0
    assert m["lagrange_multiplier"] >= 0
    ckpt = tmp_path / "run/checkpoints/latest.pt"
    assert ckpt.exists()
    load_checkpoint(ckpt, device="cpu", optimizers=True)


def test_campaign_ablation_is_matched():
    plan = json.loads(
        Path("configs/experiments/08_v11_actuator_safe_ppo.json").read_text()
    )
    assert len(plan["variants"]) == 4
    assert plan["num_envs"] == 32
    for v in plan["variants"]:
        c, others, all_profiles = model_configs(
            tiny(False), plan, v, "pilot", 42, "cpu"
        )
        assert len(others) == 3 and len(all_profiles) == 4
        assert c["training"]["physics_steps"] == 2_000_000
        assert c["policy"]["history_steps"] == tiny(False)["policy"]["history_steps"]
        assert c["training"]["teacher_method"] == "pd_observed"
    arms = {v["name"]: v for v in plan["variants"]}
    assert not arms["v10_control"]["safety"]["predictive_enabled"]
    assert arms["v11a_predictive_reward"]["safety"]["predictive_enabled"]
    assert not arms["v11a_predictive_reward"]["safety"]["constraint_enabled"]
    assert arms["v11b_constrained_ppo"]["safety"]["constraint_enabled"]
    assert arms["v11c_constrained_curriculum"]["safety"]["critical_reset_fraction"] > 0



def test_recovery_probe_has_identical_start_states_for_ppo_and_pd():
    from scripts.run_v11_recovery_probe import rollout
    c = tiny(False)
    c["task"]["duration"] = 0.03
    c["training"]["teacher_method"] = "pd_observed"
    validate(c)
    model = new_model(c, optimizers=False)
    actor, actor_init = rollout(
        model, c, total=4, batch_size=4, seed=511,
        critical_fraction=0.5, reference=False,
    )
    expert, expert_init = rollout(
        model, c, total=4, batch_size=4, seed=511,
        critical_fraction=0.5, reference=True,
    )
    assert actor_init == expert_init
    assert len(actor) == len(expert) == 4
    assert all(0 <= item["maximum_tilt_deg"] <= 70 for item in actor)
