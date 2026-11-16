"""High-speed first-entry dynamics, teacher and configuration regressions."""
import copy
from pathlib import Path
import pytest
import torch
from pursuit_rl.ctbr.config import SCHEMA, DEFAULT, interface, load, merge
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
