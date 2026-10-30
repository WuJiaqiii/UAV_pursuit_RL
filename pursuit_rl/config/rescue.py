"""Explicit, versioned settings for the new control interfaces; legacy configs stay intact."""

import copy
import hashlib
import json
import math
from pathlib import Path
from pursuit_rl.config.loader import load_config

SCHEMA = "pursuit-rescue-profile-v1"
DEFAULTS = {
    "schema": SCHEMA,
    "stage": 0,
    "require_init_checkpoint": False,
    "action_mode": "velocity",
    "decision_interval_steps": 20,
    "randomize": True,
    "velocity_scale": [15.0, 15.0, 5.0],
    "acceleration_limit": 8.0,
    "velocity_gain": 2.0,
    "attitude_gain": 4.0,
    "tilt_limit_deg": 35.0,
    "position_teacher_gain": 0.4,
    "closing_speed_max": 6.0,
    "position_scale": 30.0,
    "velocity_obs_scale": 15.0,
    "discount_time_s": 20.0,
    "advantage_time_s": 3.0,
    "reward_scale": 0.1,
    "timeout_penalty": 25.0,
    "exploration_time_s": 0.0,
    "initial_action_std": 0.1,
    "min_action_std": 0.02,
    "hidden_actor": [128, 128],
    "hidden_critic": [128, 128],
    "bc_samples": 40000,
    "bc_stride_physics_steps": 20,
    "bc_updates": 1200,
    "bc_batch_size": 1024,
    "bc_lr": 0.001,
    "dagger_rounds": 1,
    "dagger_samples": 20000,
    "dagger_updates": 500,
    "critic_mc_episodes": 64,
    "critic_pretrain_updates": 600,
    "critic_reanchor_rmse": 1.5,
    "critic_reanchor_updates": 300,
    "rollout_decisions": 256,
    "ppo_actor_lr": 0.00001,
    "ppo_critic_lr": 0.0001,
    "ppo_batch_size": 1024,
    "ppo_epochs": 3,
    "ppo_target_kl": 0.01,
    "ppo_entropy_coef": 0.0,
    "validation_every_physics_steps": 2000000,
    "validation_episodes": 64,
    "validation_seed": 78000,
    "test_seed": 990000,
    "test_episodes": 256,
    "bc_min_success": 0.8,
    "budget_physics_steps": 20000000,
    "num_envs": 64,
}


def load_rescue_profile(path, device=None):
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(raw, dict) or "base_config" not in raw:
        raise ValueError("Rescue profile must contain base_config")
    unknown = set(raw) - set(DEFAULTS) - {"base_config"}
    if unknown:
        raise ValueError("Unknown rescue settings: " + str(sorted(unknown)))
    profile = copy.deepcopy(DEFAULTS)
    profile.update({k: v for k, v in raw.items() if k != "base_config"})
    if profile["schema"] != SCHEMA or profile["action_mode"] not in (
        "velocity",
        "acceleration",
        "ctbr",
    ):
        raise ValueError("Unknown rescue profile/action encoding")
    for k, default in DEFAULTS.items():
        value = profile[k]
        if isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = type(value) is int and value >= 0
        elif isinstance(default, float):
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        elif isinstance(default, list):
            valid = (
                isinstance(value, list)
                and bool(value)
                and all(
                    type(x) in (int, float) and math.isfinite(x) and x > 0
                    for x in value
                )
            )
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError("Invalid rescue setting: " + k)
    positive = (
        "decision_interval_steps",
        "acceleration_limit",
        "velocity_gain",
        "attitude_gain",
        "tilt_limit_deg",
        "position_teacher_gain",
        "closing_speed_max",
        "position_scale",
        "velocity_obs_scale",
        "discount_time_s",
        "advantage_time_s",
        "reward_scale",
        "initial_action_std",
        "min_action_std",
        "bc_samples",
        "bc_stride_physics_steps",
        "bc_batch_size",
        "bc_lr",
        "rollout_decisions",
        "ppo_actor_lr",
        "ppo_critic_lr",
        "ppo_batch_size",
        "ppo_epochs",
        "validation_every_physics_steps",
        "validation_episodes",
        "test_episodes",
        "num_envs",
    )
    if any(profile[k] <= 0 for k in positive):
        raise ValueError("Required positive rescue setting is zero")
    if not 0 <= profile["bc_min_success"] <= 1:
        raise ValueError("bc_min_success must be a probability")
    for key in ("hidden_actor", "hidden_critic"):
        if not all(type(x) is int for x in profile[key]):
            raise ValueError("Hidden layer widths must be integers")
    if len(profile["velocity_scale"]) != 3:
        raise ValueError("velocity_scale needs three axes")
    if profile["initial_action_std"] > math.e:
        raise ValueError("initial_action_std exceeds the policy upper bound")
    if profile["critic_mc_episodes"] < 1:
        raise ValueError("critic_mc_episodes must be positive")
    if profile["min_action_std"] > profile["initial_action_std"]:
        raise ValueError("Invalid exploration floor")
    cfg = load_config(path.parent / raw["base_config"])
    if device is not None:
        cfg["device"] = device
    cfg["simulation_backend"] = "torch"
    if profile["stage"] >= len(cfg["curriculum"]):
        raise ValueError("Rescue stage is outside the curriculum")
    cfg["training"]["initial_stage"] = profile["stage"]
    cfg["training"]["max_stage"] = max(cfg["training"]["max_stage"], profile["stage"])
    cfg["physics"]["policy_interval_steps"] = profile["decision_interval_steps"]
    cfg["training"]["num_envs"] = profile["num_envs"]
    cfg["training"]["gamma"] = math.exp(
        -cfg["physics"]["dt"] / profile["discount_time_s"]
    )
    cfg["training"]["aux_pd_coef"] = 0.0
    if cfg["physics"]["dt"] != 0.001:
        raise ValueError("Rescue physics must run at 1 kHz")
    if profile["tilt_limit_deg"] >= cfg["physics"]["max_tilt_deg"]:
        raise ValueError("Inner tilt limit must leave a safety margin")
    dt = cfg["physics"]["dt"] * profile["decision_interval_steps"]
    if (
        math.exp(-dt / profile["advantage_time_s"])
        / math.exp(-dt / profile["discount_time_s"])
        > 1
    ):
        raise ValueError("Advantage time cannot exceed discount time")
    profile["base_config"] = str((path.parent / raw["base_config"]).resolve())
    return cfg, profile


def rescue_contract(cfg, profile):
    """Reject loading a network under another output unit, rate, observation, or plant."""
    return {
        "schema": SCHEMA,
        "action_mode": profile["action_mode"],
        "decision_interval_steps": profile["decision_interval_steps"],
        "observation_encoding": "ned-full-actuator-boundary-raw-memory-v1",
        "target_accel_scale": cfg["observation"]["target_accel_scale"],
        "stage": profile["stage"],
        "task": copy.deepcopy(cfg["curriculum"][profile["stage"]]),
        "target": copy.deepcopy(cfg["target"]),
        "reward": copy.deepcopy(cfg["reward"]),
        "position_scale": profile["position_scale"],
        "velocity_obs_scale": profile["velocity_obs_scale"],
        "velocity_scale": profile["velocity_scale"],
        "acceleration_limit": profile["acceleration_limit"],
        "velocity_gain": profile["velocity_gain"],
        "attitude_gain": profile["attitude_gain"],
        "tilt_limit_deg": profile["tilt_limit_deg"],
        "reward_scale": profile["reward_scale"],
        "timeout_penalty": profile["timeout_penalty"],
        "discount_time_s": profile["discount_time_s"],
        "exploration_time_s": profile["exploration_time_s"],
        "hidden_actor": profile["hidden_actor"],
        "hidden_critic": profile["hidden_critic"],
        "physics": copy.deepcopy(cfg["physics"]),
    }


def rescue_fingerprint(cfg, profile):
    clean = {
        k: v
        for k, v in profile.items()
        if k != "base_config" and not (k == "require_init_checkpoint" and v is False)
    }
    return hashlib.sha256(
        json.dumps({"base": cfg, "rescue": clean}, sort_keys=True).encode()
    ).hexdigest()
