"""Regression tests for V13 pursuit-first PPO fixes and V12 compatibility."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pursuit_rl.ctbr.config import load as load_config, validate
from pursuit_rl.ctbr.checkpoint import new_model, save, load as load_checkpoint
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.control import observed_pd
from pursuit_rl.ctbr.evaluation import score
from scripts.run_pure_ppo import make_config
from scripts.run_pure_ppo_campaign import run_command
from scripts.run_pure_ppo_curriculum import validate_plan


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def config(task="static", v13=True):
    args = SimpleNamespace(
        task=task, preset="quick", device="cpu", seed=42,
        num_envs=4, rollout_steps=2, batch_size=8, epochs=1,
        history_steps=1, physics_steps=400,
        legacy_terminal_records=False, pursuit_v13=v13,
    )
    return make_config(args)


def test_legacy_default_unchanged_and_v13_is_explicit_optin():
    old, new = config(v13=False), config(v13=True)
    assert old["policy"]["relative_position_encoding"] == "linear"
    assert old["policy"]["relative_position_scale_m"] == 100.0
    assert old["policy"]["relative_velocity_scale_mps"] == 50.0
    assert old["training"]["selection_mode"] == "legacy"
    assert old["reward"]["progress_mode"] == "saturated"
    assert new["policy"]["relative_position_encoding"] == "asinh"
    assert new["policy"]["relative_frame"] == "body"
    assert new["policy"]["relative_position_scale_m"] == 5.0
    assert new["policy"]["relative_velocity_scale_mps"] == 10.0
    assert new["reward"]["progress_mode"] == "initial_normalized"
    assert new["training"]["selection_mode"] == "distance_first"


def test_near_static_has_valid_short_range_first_entry():
    cfg = config("near_static")
    assert cfg["task"]["radius"] < cfg["task"]["distance"][0]
    assert cfg["task"]["distance"] == [0.75, 1.5]
    assert cfg["task"]["duration"] == 6


def test_state_direction_is_visible_at_one_meter_and_pd_decodes_exactly():
    old, new = config(v13=False), config(v13=True)
    a, b = CTBRBatch(old, n=4, seed=432), CTBRBatch(new, n=4, seed=432)
    obs_old, _ = a.observe()
    obs_new, _ = b.observe()
    rel = (b.plant.target.pos - b.plant.pursuer.pos).float()
    rel_body = b.plant.pursuer.rot.transpose(1, 2).bmm(
        rel.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(
        obs_new[:, :3], torch.asinh(rel_body / 5), atol=1e-6
    )
    assert torch.allclose(obs_old[:, :3], rel / 100, atol=1e-6)
    assert obs_new[:, :3].abs().mean() > obs_old[:, :3].abs().mean() * 4
    assert torch.allclose(
        observed_pd(a), observed_pd(b), atol=2e-5, rtol=2e-5
    )


def test_distance_first_selection_does_not_prefer_waiting_over_approach():
    safe_timeout = {
        "success_rate": 0.0, "successes": 0, "mean_entry_time_s": None,
        "mean_minimum_distance_m": 125.0,
        "episodes_detail": [{"reason": "timeout"}] * 16,
    }
    closer_with_few_faults = {
        "success_rate": 0.0, "successes": 0, "mean_entry_time_s": None,
        "mean_minimum_distance_m": 50.0,
        "episodes_detail": [{"reason": "timeout"}] * 12 +
                           [{"reason": "tilt"}] * 4,
    }
    assert score(safe_timeout) > score(closer_with_few_faults)
    assert score(closer_with_few_faults, mode="distance_first") > score(
        safe_timeout, mode="distance_first"
    )


def test_actor_cap_uses_correct_physical_units():
    cfg = config()
    cfg["policy"]["position_error_cap_m"] = 2.0
    model = new_model(validate(cfg), optimizers=False)
    sample = torch.zeros(1, model.obs_dim)
    sample[0, :3] = torch.asinh(torch.tensor([10., 0., 0.]) / 5.)
    bounded = model.bound_actor_input(sample)
    assert torch.sinh(bounded[0, 0]).item() * 5.0 == pytest.approx(2.0)


def test_initial_normalized_progress_is_positive_for_real_approach():
    c = config()
    env = CTBRBatch(c, n=1, seed=14)
    pursuer, target = env.plant.pursuer, env.plant.target
    pursuer.pos[0] = target.pos[0].float() - torch.tensor([2., 0., 0.])
    pursuer.vel[0] = torch.tensor([1., 0., 0.])
    env.initial_rel_pos[0] = target.pos[0].float() - pursuer.pos[0]
    env.minimum_distance[0] = 2.
    _, reward, done, _, records = env.step(torch.zeros(1, 4))
    assert not bool(done[0])
    assert env.reward_components["progress"].item() > 0.
    assert reward.item() > -0.01


def test_ppo_training_config_and_launch_command_stay_consistent():
    cfg = config()
    item = {
        "task": "static", "preset": "quick", "seed": 42,
        "budget": 400, "config": cfg,
    }
    command = run_command(item, Path("/tmp/not-a-real-run"), "cpu")
    assert "--pursuit-v13" in command
    assert "--init-from" not in command


def test_two_stage_v13_plan_transfers_policy_and_reward_semantics():
    path = Path("configs/experiments/16_v13_near_static_probe.json")
    plan = json.loads(path.read_text(encoding="utf8"))
    runs, evaluation = validate_plan(plan, "cpu")
    assert len(runs) == 2
    assert runs[1]["parent_id"] == runs[0]["id"]
    assert runs[0]["config"]["policy"]["relative_position_encoding"] == "asinh"
    assert all(r["config"]["training"]["selection_mode"] == "distance_first"
               for r in runs)
    assert "near_static" in evaluation["profiles"]


def test_v13_checkpoint_keeps_observation_units_on_reload(tmp_path):
    cfg = config()
    model = new_model(cfg, optimizers=False)
    checkpoint = tmp_path / "v13.pt"
    save(checkpoint, model, cfg, dict(critic_needs_calibration=False), "ppo")
    restored, loaded, _ = load_checkpoint(checkpoint, device="cpu")
    assert loaded["policy"]["relative_position_scale_m"] == 5.0
    assert restored.relative_position_encoding == "asinh"
    assert restored.relative_position_scale_m == pytest.approx(5.0)
