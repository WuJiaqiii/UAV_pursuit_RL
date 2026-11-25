"""V12.1 chase learning: reward, observation, checkpoint selection and CTBR sign."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from pursuit_rl.ctbr.config import load, validate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.control import observed_pd
from pursuit_rl.ctbr.evaluation import score, improved
from scripts import run_pure_ppo as pure
from scripts import run_pure_ppo_campaign as campaign
from scripts.audit_pursuit_learnability import ZeroCTBRPolicy


@pytest.fixture(autouse=True)
def cpu_single_thread():
    torch.set_num_threads(1)


def config(task="static", mode="legacy_saturating"):
    c = load(pure.TASKS[task], device="cpu", seed=42)
    c["num_envs"] = 2
    c["prediction"]["enabled"] = False
    c["reward"]["progress_mode"] = mode
    c["reward"]["progress_distance_floor_m"] = 1.0
    c["policy"]["history_steps"] = 1
    return validate(c)


def _set_straight_encounter(env, distance=15.0, target_speed=-3.0):
    """Set target due east and stationary, level pursuer in NED frame."""
    u, t = env.plant.pursuer, env.plant.target
    u.pos.zero_()
    u.pos[:, 2] = -100
    u.vel.zero_()
    u.rate.zero_()
    u.rot.copy_(torch.eye(3).repeat(env.n, 1, 1))
    u.thrust.fill_(env.config["physics"]["hover_thrust"])
    u.drag.zero_()
    u.mass_ratio.fill_(1.0)
    t.pos.copy_(u.pos.double())
    t.pos[:, 0] += distance
    t.vel.zero_()
    t.vel[:, 0] = target_speed
    env.initial_rel_pos.copy_((t.pos - u.pos).float())
    env.minimum_distance.fill_(distance)
    return u, t


def test_asinh_position_observation_is_invertible_and_finite():
    c = config()
    c["policy"]["relative_position_mode"] = "asinh"
    c["policy"]["relative_position_scale_m"] = 8.0
    c["policy"]["relative_velocity_scale_mps"] = 15.0
    env = CTBRBatch(validate(c), n=2, seed=17)
    obs, _ = env.observe()
    actual = (env.plant.target.pos - env.plant.pursuer.pos).float()
    restored = torch.sinh(obs[:, :3]) * 8.0
    assert torch.allclose(restored, actual, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(obs).all()
    assert obs.shape[-1] == 38
    # Ref control decodes the altered observation into physical meters.
    ctrl = observed_pd(env)
    assert ctrl.shape == (2, 4)
    assert torch.isfinite(ctrl).all()
    assert bool((ctrl.abs() <= 1).all())


def test_active_pursuer_closure_does_not_reward_passive_head_on_motion():
    c = config("head_on", "pursuer_closure_normalized")
    env = CTBRBatch(c, n=2, seed=30)
    _set_straight_encounter(env, 15.0, target_speed=-4.0)
    noop = torch.zeros(2, 4)
    env.step(noop)
    active_reward = env.reward_components_unscaled["progress"]
    assert active_reward.abs().max().item() < 3e-5

    c_legacy = config("head_on", "legacy_saturating")
    old = CTBRBatch(c_legacy, n=2, seed=30)
    _set_straight_encounter(old, 15.0, target_speed=-4.0)
    old.step(noop)
    legacy_reward = old.reward_components_unscaled["progress"]
    assert (legacy_reward > 1e-5).all()


def test_straight_line_progress_rewards_pursuer_motion_and_pitch_sign():
    c = config("static", "pursuer_closure_normalized")
    env = CTBRBatch(c, n=2, seed=30)
    u, _ = _set_straight_encounter(env, distance=3.0, target_speed=0.0)
    # Negative body pitch rate makes level thrust point towards +NED x.
    action = torch.tensor([[0.0, 0.0, -0.35, 0.0]] * env.n)
    reward_total = 0.0
    for _ in range(15):
        env.step(action)
        reward_total += float(env.reward_components_unscaled["progress"].sum())
    assert (u.pos[:, 0] > 1e-3).all(), u.pos
    assert (u.vel[:, 0] > 0.01).all(), u.vel
    assert reward_total > 0.001


def _report(min_distance, safe_fraction=1.0, successes=0):
    n = 4
    safe = round(safe_fraction * n)
    rows = [
        dict(
            reason="timeout" if i < safe else "tilt",
            start_rel_pos=[10.0, 0.0, 0.0],
            minimum_distance=float(min_distance),
        ) for i in range(n)
    ]
    return dict(
        episodes_detail=rows,
        successes=successes,
        success_rate=successes / n,
        mean_minimum_distance_m=min_distance,
        mean_entry_time_s=3.0 if successes else None,
    )


def test_checkpoint_pursuit_progress_prioritizes_closure_with_risk_penalty():
    stationary = _report(9.8, 1.0)
    approaching = _report(3.0, 0.75)
    # Historical selector picks passive timeout over more useful approach.
    assert score(stationary, mode="legacy") > score(approaching, mode="legacy")
    assert score(approaching, mode="pursuit_progress") > score(
        stationary, mode="pursuit_progress"
    )
    assert improved(
        approaching, score(stationary, mode="pursuit_progress"),
        mode="pursuit_progress",
    )
    # A reckless model that mostly crashes without closing is penalized.
    reckless = _report(9.0, 0.0)
    assert score(stationary, mode="pursuit_progress") > score(
        reckless, mode="pursuit_progress"
    )


def test_invalid_chase_config_is_rejected():
    c = config()
    c["reward"]["progress_mode"] = "arbitrary"
    with pytest.raises(ValueError, match="progress_mode"):
        validate(c)
    c = config()
    c["policy"]["relative_position_scale_m"] = 0
    with pytest.raises(ValueError, match="relative_position_scale_m"):
        validate(c)
    c = config()
    c["training"]["checkpoint_selection"] = "arbitrary"
    with pytest.raises(ValueError, match="checkpoint_selection"):
        validate(c)


def test_campaign_passes_exact_selection_observation_and_reward_settings():
    plan = dict(
        schema=campaign.SCHEMA,
        defaults=dict(
            preset="smoke", num_envs=2, rollout_steps=2,
            batch_size=4, epochs=1, history_steps=1,
            progress_mode="pursuer_closure_normalized",
            checkpoint_selection="pursuit_progress",
            relative_position_mode="asinh",
            relative_position_scale_m=8.0,
            relative_velocity_scale_mps=15.0,
        ),
        stages=[dict(
            name="Near Static", task="near_static", seeds=[42],
            physics_steps=100,
        )],
    )
    item = campaign.expand_plan(plan, "cpu")[0]
    assert item["config"]["reward"]["progress_mode"] == "pursuer_closure_normalized"
    assert item["config"]["policy"]["relative_position_mode"] == "asinh"
    assert item["config"]["training"]["checkpoint_selection"] == "pursuit_progress"
    args = campaign.run_command(item, "unused", "cpu")
    for flag, value in (
        ("--progress-mode", "pursuer_closure_normalized"),
        ("--relative-position-mode", "asinh"),
        ("--relative-position-scale-m", "8.0"),
        ("--relative-velocity-scale-mps", "15.0"),
        ("--checkpoint-selection", "pursuit_progress"),
    ):
        assert args[args.index(flag) + 1] == value


def test_zero_policy_returns_safe_neutral_action_without_supervision():
    policy = ZeroCTBRPolicy(torch.device("cpu"))
    obs = torch.zeros(4, 38)
    priv = torch.zeros(4, 70)
    action, raw, lp, values = policy.sample_tensor(obs, priv)
    assert torch.count_nonzero(action) == 0
    assert torch.equal(action, raw)
    assert torch.count_nonzero(lp) == 0
    assert torch.count_nonzero(values) == 0


def test_new_close_range_task_does_not_start_inside_capture_sphere():
    c = config("near_static")
    assert c["task"]["distance"][0] > c["task"]["radius"]
    assert c["task"]["duration"] < config("static")["task"]["duration"]
