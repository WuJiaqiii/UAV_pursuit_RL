"""Regression coverage for v7.1 2.5g direct-CTBR first-entry curriculum."""
import copy
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.config import interface, load
from pursuit_rl.ctbr.control import guidance_acceleration, teacher
from pursuit_rl.ctbr.curriculum import gate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.mixture import TaskMixtureBatch


PHASES = sorted(Path("configs/first_entry_highspeed").glob("[0-9]*.json"))


@pytest.mark.parametrize("profile", PHASES, ids=lambda p: p.stem)
def test_high_speed_profiles_are_valid_and_share_interface(profile):
    c = load(profile)
    baseline = load(PHASES[0])
    assert interface(c) == interface(baseline)
    assert c["physics"]["thrust_max"] == pytest.approx(2.5 * 9.81)
    assert c["physics"]["max_load_factor_g"] == pytest.approx(2.5)
    assert c["physics"]["max_speed"] == 60
    assert c["task"]["entry_relative_speed_max_mps"] == 15
    env = CTBRBatch(c, n=2, seed=47)
    obs, priv = env.observe()
    assert obs.shape == (2, env.actor_dim)
    assert priv.shape == (2, env.critic_dim)
    assert bool(torch.isfinite(obs).all())


@pytest.mark.parametrize("profile", [
    "02_low_tail", "03_low_head_on", "08_high_tail", "09_high_head_on",
    "11_extreme_tail", "12_extreme_head_on",
])
def test_encounter_geometry_and_speed(profile):
    c = load(f"configs/first_entry_highspeed/{profile}.json")
    env = CTBRBatch(c, n=4, seed=109)
    u, t = env.plant.pursuer, env.plant.target
    displacement = (u.pos - t.pos).float()
    proj = (displacement * t.vel.float()).sum(1)
    velocity_dot = (u.vel * t.vel.float()).sum(1)
    if "tail" in profile:
        assert bool((proj < 0).all())
        assert bool((velocity_dot > 0).all())
    else:
        assert bool((proj > 0).all())
        assert bool((velocity_dot < 0).all())


def test_trimmed_initialization_balances_drag_and_gravity():
    c = load("configs/first_entry_highspeed/08_high_tail.json")
    env = CTBRBatch(c, n=4, seed=6)
    u = env.plant.pursuer
    gravity = u.vel.new_tensor([0, 0, c["physics"]["gravity"]])
    acceleration = (
        -u.rot[:, :, 2] * (u.thrust / u.mass_ratio)[:, None]
        + gravity
        - u.drag[:, None] * u.vel * torch.linalg.vector_norm(u.vel, dim=1, keepdim=True)
    )
    assert torch.allclose(acceleration, torch.zeros_like(acceleration), atol=2e-4)


def test_max_load_factor_is_enforced_on_actual_mass():
    c = load("configs/first_entry_highspeed/01_static.json")
    env = CTBRBatch(c, n=2)
    u = env.plant.pursuer
    u.mass_ratio[:] = torch.tensor([0.8, 1.0])
    action = torch.zeros(2, 4)
    action[:, 0] = 1.0
    for _ in range(80):
        env.plant.step(action)
    actual_load = u.thrust / u.mass_ratio / c["physics"]["gravity"]
    assert bool((actual_load <= 2.5 + 1e-5).all())


@pytest.mark.parametrize("speed,reason,success", [
    (10.0, "success", True),
    (17.0, "unsafe_entry", False),
])
def test_entry_speed_threshold_is_terminal(speed, reason, success):
    c = load("configs/first_entry_highspeed/01_static.json")
    c["policy"]["decision_steps"] = 1
    env = CTBRBatch(c, n=1)
    u, t = env.plant.pursuer, env.plant.target
    u.pos.copy_(t.pos.float())
    u.pos[0, 0] += 0.505
    u.vel.zero_()
    u.vel[0, 0] = -speed
    _, reward, done, truncated, records = env.step(torch.zeros(1, 4))
    assert done.item() and not truncated.item() and len(records) == 1
    assert records[0]["reason"] == reason
    assert records[0]["success"] == success
    assert records[0]["entry_relative_speed_mps"] == pytest.approx(speed, rel=2e-3)
    assert (reward.item() > 0) == success


def test_pn_guidance_is_finite_and_ctbr_normalized():
    c = load("configs/first_entry_highspeed/09_high_head_on.json")
    env = CTBRBatch(c, n=4, seed=17)
    for method in ("pd", "pn", "pn_hybrid"):
        env.config["training"]["teacher_method"] = method
        acceleration = guidance_acceleration(env)
        action = teacher(env)
        assert acceleration.shape == (4, 3)
        assert action.shape == (4, 4)
        assert bool(torch.isfinite(acceleration).all())
        assert bool(torch.isfinite(action).all())
        assert bool((action.abs() <= 1).all())


def test_first_entry_mixture_and_objective_mismatch():
    c = load("configs/first_entry_highspeed/01_static.json")
    c["num_envs"] = 10
    recent = copy.deepcopy(c)
    recent["task"]["name"] = "recent"
    recent["task"]["distance"] = [3.0, 5.0]
    env = TaskMixtureBatch(c, recent, None, 11)
    a = torch.zeros(10, 4)
    (_, _), reward, _, _, _ = env.step(a, a)
    assert torch.allclose(reward, sum(env.reward_components.values()), atol=1e-5)
    other = copy.deepcopy(recent)
    other["task"]["objective"] = "sustained"
    with pytest.raises(ValueError, match="terminal objectives"):
        TaskMixtureBatch(c, other, None, 11)


def test_legacy_interface_omits_disabled_optional_features():
    c = load("configs/first_entry/static.json")
    original = copy.deepcopy(c)
    original["physics"].pop("max_load_factor_g")
    for key in ("braking", "entry_velocity", "time"):
        original["reward"].pop(key)
    assert interface(original) == interface(c)


def test_first_entry_retention_uses_success_not_tail_rmse(tmp_path, monkeypatch):
    c = load("configs/first_entry_highspeed/01_static.json")
    def evaluate_stub(*args, **kwargs):
        return {
            "seed": args[3],
            "success_rate": 1.0,
            "mean_tail_position_rmse": None,
            "failures": {},
        }, None
    monkeypatch.setattr("pursuit_rl.ctbr.curriculum.evaluate", evaluate_stub)
    plan = {"gate_seed_offsets": [100, 200], "retention_rmse_relative_budget": 0.2}
    anchors = [{
        "name": "anchor", "config": c,
        "reference": [{"seed": 123, "tail_rmse": None}]
    }]
    result = gate(None, c, anchors, tmp_path, 4, 42, 0.9, plan, "test")
    assert result["passed"]
    assert result["retention"][0]["rmse_budget"] is None

def test_pn_brakes_excessive_closing_speed():
    c = load("configs/first_entry_highspeed/01_static.json")
    c["training"]["teacher_method"] = "pn"
    c["training"]["teacher_accel_limit"] = 24.0
    env = CTBRBatch(c, n=1)
    u, target = env.plant.pursuer, env.plant.target
    target.pos.zero_()
    target.vel.zero_()
    target.acc.zero_()
    u.pos.zero_()
    u.pos[0, 0] = -2.0
    u.vel.zero_()
    u.vel[0, 0] = 20.0
    acceleration = guidance_acceleration(env)
    assert acceleration[0, 0] < 0, "PN longitudinal feedback must brake excessive closing"

def test_reference_method_cli_is_explicit():
    from main import parser
    parsed = parser().parse_args([
        "evaluate", "--reference", "--reference-method", "pn",
        "--config", "configs/first_entry_highspeed/01_static.json",
        "--out", "runs/unused.json"
    ])
    assert parsed.reference_method == "pn"
