"""High-speed first-entry dynamics, teacher and configuration regressions."""
import copy
from pathlib import Path
import pytest
import torch
from pursuit_rl.ctbr.config import SCHEMA, DEFAULT, interface, transfer_interface, load, merge
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.control import guidance_acceleration, teacher
from pursuit_rl.ctbr.mixture import TaskMixtureBatch

PROFILES = sorted(Path("configs/first_entry_highspeed").glob("[0-9]*.json"))

@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    torch.manual_seed(123)

@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_all_thirteen_profiles_share_interface(path):
    c = load(path)
    assert c["schema"] == SCHEMA
    assert c["physics"]["thrust_max"] == pytest.approx(24.525)
    assert c["physics"]["max_load_factor_g"] == 2.5
    assert c["physics"]["max_speed"] == 60
    assert c["task"]["entry_relative_speed_max_mps"] == 15
    assert interface(c) == interface(load(PROFILES[0]))
    env = CTBRBatch(c, n=2, seed=47)
    obs, critic = env.observe()
    assert obs.shape == (2, env.actor_dim)
    assert critic.shape == (2, env.critic_dim)
    assert env.critic_dim - env.actor_dim == 32
    assert torch.isfinite(obs).all()
    assert obs[0, env.base_actor_dim - 1] == pytest.approx(1.0)

@pytest.mark.parametrize("name", [
    "02_low_tail", "03_low_head_on", "05_medium_tail",
    "06_medium_head_on", "08_high_tail", "09_high_head_on",
    "11_extreme_tail", "12_extreme_head_on",
])
def test_encounter_initialization(name):
    env = CTBRBatch(load(f"configs/first_entry_highspeed/{name}.json"), n=4, seed=101)
    u, t = env.plant.pursuer, env.plant.target
    relative = (u.pos - t.pos).float()
    projection = (relative * t.vel.float()).sum(1)
    alignment = (u.vel * t.vel.float()).sum(1)
    if "tail" in name:
        assert (projection < 0).all() and (alignment > 0).all()
    else:
        assert (projection > 0).all() and (alignment < 0).all()

def test_highspeed_trim_with_random_mass():
    env = CTBRBatch(load("configs/first_entry_highspeed/10_high_random.json"), n=8)
    u, p = env.plant.pursuer, env.config["physics"]
    gravity = u.vel.new_tensor([0., 0., p["gravity"]])
    acceleration = (
        -u.rot[:, :, 2] * (u.thrust / u.mass_ratio)[:, None] + gravity
        - u.drag[:, None] * u.vel * torch.linalg.vector_norm(u.vel, dim=1, keepdim=True)
    )
    assert torch.allclose(acceleration, torch.zeros_like(acceleration), atol=2e-4)

def test_actual_mass_respects_2p5g():
    env = CTBRBatch(load(PROFILES[0]), n=2)
    u = env.plant.pursuer
    u.mass_ratio[:] = torch.tensor([0.8, 1.1])
    action = torch.zeros(2, 4)
    action[:, 0] = 1
    for _ in range(250):
        env.plant.step(action)
    assert (u.thrust / u.mass_ratio / 9.81 <= 2.50001).all()

@pytest.mark.parametrize("speed,reason,success", [
    (10., "success", True), (17., "unsafe_entry", False)
])
def test_safe_entry_speed_limit(speed, reason, success):
    c = load(PROFILES[0])
    c["policy"]["decision_steps"] = 1
    env = CTBRBatch(c, n=1)
    u, t = env.plant.pursuer, env.plant.target
    u.pos.copy_(t.pos.float())
    u.pos[0, 0] += 0.505
    u.vel.zero_()
    u.vel[0, 0] = -speed
    _, reward, done, truncated, records = env.step(torch.zeros(1, 4))
    assert done.item() and not truncated.item()
    assert records[0]["reason"] == reason
    assert records[0]["success"] == success
    assert (reward.item() > 0) == success

def test_random_target_retains_highspeed_band():
    c = load("configs/first_entry_highspeed/10_high_random.json")
    env = CTBRBatch(c, n=5, seed=4)
    for _ in range(200):
        env.advance_target()
        speed = torch.linalg.vector_norm(env.plant.target.vel, dim=1)
        assert (speed <= c["task"]["speed"][1] + .001).all()
        assert (speed >= c["task"]["speed"][0] - .2).all()

def test_pn_pd_and_hybrid_bounded_actions():
    env = CTBRBatch(load("configs/first_entry_highspeed/09_high_head_on.json"), n=4)
    for method in ("pd", "pn", "pn_hybrid"):
        env.config["training"]["teacher_method"] = method
        assert torch.isfinite(guidance_acceleration(env)).all()
        action = teacher(env)
        assert action.shape == (4, 4) and (action.abs() <= 1).all()

def test_mixture_reward_components_are_consistent():
    c = load(PROFILES[0])
    c["num_envs"] = 10
    recent = copy.deepcopy(c)
    recent["task"].update(name="recent", distance=[3., 5.])
    env = TaskMixtureBatch(c, recent, None, 11)
    action = torch.zeros(10, 4)
    (_, _), reward, _, _, _ = env.step(action, action)
    assert torch.allclose(reward, sum(env.reward_components.values()), atol=1e-5)
    assert env.task_counts == {"01_static": 8, "recent": 2}

def test_old_sustained_config_keys_are_rejected():
    c = load(PROFILES[0])
    c["task"]["objective"] = "sustained"
    with pytest.raises(ValueError, match="Unknown"):
        merge(copy.deepcopy(DEFAULT), c)
    c = load(PROFILES[0])
    c["reward"]["survival"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        merge(copy.deepcopy(DEFAULT), c)

def test_highspeed_rollout_keeps_ctbr_and_observations_finite():
    from pursuit_rl.ctbr.checkpoint import new_model
    c = load("configs/first_entry_highspeed/09_high_head_on.json")
    c["num_envs"] = 4
    env = CTBRBatch(c, seed=501)
    model = new_model(c, optimizers=False)
    obs, critic = env.observe()
    for _ in range(25):
        action, raw, _, _ = model.sample_tensor(obs, critic)
        (obs, critic), reward, _, _, _ = env.step(action, raw)
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all()
        assert torch.isfinite(reward).all()
        assert (action.abs() <= 1).all()
        assert (env.plant.pursuer.thrust / env.plant.pursuer.mass_ratio / 9.81 <= 2.50001).all()


def test_random_highspeed_acceleration_and_jerk_bounds():
    c = load("configs/first_entry_highspeed/13_extreme_random.json")
    env = CTBRBatch(c, n=6, seed=18)
    for _ in range(400):
        old_acc = env.plant.target.acc.clone()
        env.advance_target()
        accel = torch.linalg.vector_norm(env.plant.target.acc, dim=1)
        jerk = torch.linalg.vector_norm(env.plant.target.acc - old_acc, dim=1) / .001
        assert (accel <= c["task"]["max_accel"] + 0.2).all()
        assert (jerk <= c["task"]["jerk_limit"] + 0.2).all()


def test_cruise_trim_rejects_physically_unreachable_drag():
    c = load("configs/first_entry_highspeed/11_extreme_tail.json")
    c["physics"]["drag"] = 0.02
    with pytest.raises(ValueError, match="Trimmed cruise"):
        CTBRBatch(c, n=6, seed=88)


def test_scaled_reward_components_are_the_ppo_reward():
    c = load("configs/first_entry_highspeed/01_static.json")
    c["policy"]["decision_steps"] = 1
    c["task"]["duration"] = 0.001
    env = CTBRBatch(c, n=2, seed=707)
    action = torch.zeros(2, 4)
    (_, _), reward, done, _, records = env.step(action, action)
    scaled = sum(env.reward_components.values())
    raw = sum(env.reward_components_unscaled.values())
    assert torch.allclose(reward, scaled, atol=1e-7, rtol=1e-6)
    assert torch.allclose(scaled, raw * c["reward"]["scale"], atol=1e-7, rtol=1e-6)
    assert bool(done.all())
    assert len(records) == 2
    for record in records:
        assert record["return_scaled"] == pytest.approx(
            record["return_unscaled"] * c["reward"]["scale"], rel=1e-5, abs=1e-7
        )
    assert torch.allclose(env.episode_return_scaled, env.episode_return_unscaled * c["reward"]["scale"], atol=1e-6)


def test_reward_ablation_keeps_hard_entry_and_flight_constraints():
    c = load("configs/first_entry_highspeed/02_low_tail.json")
    c["reward"].update(
        braking=0.0, entry_velocity=0.0, tilt=0.0, rate=0.0,
        action_derivative=0.0, time=0.0,
    )
    env = CTBRBatch(c, n=1, seed=708)
    assert env.config["task"]["radius"] == pytest.approx(0.5)
    assert env.config["task"]["entry_relative_speed_max_mps"] == pytest.approx(15.0)
    assert env.config["physics"]["max_tilt_deg"] == pytest.approx(70.0)
    assert env.config["physics"]["max_speed"] == pytest.approx(60.0)
    assert env.config["physics"]["max_load_factor_g"] == pytest.approx(2.5)


def test_reward_transfer_contract_allows_reward_ablations_only():
    base = load("configs/first_entry_highspeed/02_low_tail.json")
    changed_reward = copy.deepcopy(base)
    changed_reward["reward"]["braking"] = 0.0
    changed_reward["reward"]["progress"] = 6.0
    assert transfer_interface(base) == transfer_interface(changed_reward)
    changed_physics = copy.deepcopy(base)
    changed_physics["physics"]["drag"] += 0.001
    assert transfer_interface(base) != transfer_interface(changed_physics)


def test_duplicate_recent_and_anchor_replay_is_counted_once():
    current = load("configs/first_entry_highspeed/02_low_tail.json")
    current["num_envs"] = 10
    current["training"]["replay_recent_fraction"] = 0.2
    current["training"]["replay_anchor_fraction"] = 0.2
    old = load("configs/first_entry_highspeed/01_static.json")
    old["num_envs"] = 10
    old["reward"] = copy.deepcopy(current["reward"])
    env = TaskMixtureBatch(current, old, copy.deepcopy(old), seed=709)
    assert env.task_counts == {"02_low_tail": 8, "01_static": 2}


def test_discounted_decision_reward_matches_discounted_components():
    c = load("configs/first_entry_highspeed/01_static.json")
    c["task"]["duration"] = c["policy"]["decision_steps"] * c["physics"]["dt"]
    env = CTBRBatch(c, n=2, seed=710)
    action = torch.zeros(2, 4)
    (_, _), reward, done, _, records = env.step(action, action)
    scaled = sum(env.reward_components.values())
    raw = sum(env.reward_components_unscaled.values())
    assert bool(done.all())
    assert torch.allclose(reward, scaled, atol=1e-7, rtol=1e-6)
    assert torch.allclose(scaled, raw * c["reward"]["scale"], atol=1e-7, rtol=1e-6)
    assert len(records) == 2
    for record in records:
        assert record["return_scaled"] == pytest.approx(
            record["return_unscaled"] * c["reward"]["scale"], rel=1e-5, abs=1e-7
        )


def test_reward_ablation_plan_preserves_terminal_terms_and_has_factorial_controls():
    import json
    plan = json.loads(
        Path("configs/experiments/02_low_tail_reward_ablation.json").read_text()
    )
    variants = {item["name"]: item for item in plan["variants"]}
    assert "baseline_pd_bc" in variants and "baseline_no_bc" in variants
    assert "no_braking_pd_bc" in variants
    assert "no_entry_velocity_pd_bc" in variants
    assert "no_braking_no_entry_velocity_pd_bc" in variants
    assert "progress_only_diagnostic_pd_bc" in variants
    assert len(plan["final_gate_seed_offsets"]) == 2
    assert len(plan["retention_seed_offsets"]) == 2
    for variant in variants.values():
        reward = dict(plan["base_reward"])
        reward.update(variant.get("reward", {}))
        assert reward["completion_bonus"] == 10.0
        assert reward["failure"] == 10.0
    progress_only = dict(plan["base_reward"])
    progress_only.update(variants["progress_only_diagnostic_pd_bc"]["reward"])
    assert all(progress_only[k] == 0.0 for k in (
        "braking", "entry_velocity", "tilt", "rate", "action_derivative", "time"
    ))


def test_zero_soft_shaping_does_not_disable_unsafe_entry_terminal():
    c = load("configs/first_entry_highspeed/01_static.json")
    c["policy"]["decision_steps"] = 1
    c["reward"].update(
        progress=0.0, braking=0.0, entry_velocity=0.0, tilt=0.0,
        rate=0.0, action_derivative=0.0, time=0.0,
    )
    env = CTBRBatch(c, n=1, seed=711)
    u, target = env.plant.pursuer, env.plant.target
    u.pos.copy_(target.pos.float())
    u.pos[0, 0] += 0.505
    u.vel.zero_()
    u.vel[0, 0] = -17.0
    _, reward, done, _, records = env.step(torch.zeros(1, 4))
    assert done.item()
    assert records[0]["reason"] == "unsafe_entry"
    assert not records[0]["success"]
    assert reward.item() < 0
