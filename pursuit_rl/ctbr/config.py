"""Configuration contract for high-speed first safe entry with direct CTBR."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "pursuit-first-entry-ctbr-v1"
KINDS = ("hover", "cruise", "random")
DEFAULT = {
    "schema": SCHEMA,
    "seed": 42,
    "device": "cpu",
    "num_envs": 64,
    "physics": {
        "dt": 0.001,
        "substeps": 1,
        "gravity": 9.81,
        "hover_thrust": 9.81,
        "thrust_min": 0.0,
        "thrust_max": 24.525,
        "max_load_factor_g": 2.5,
        "rate_xy": 2.5,
        "rate_yaw": 1.5,
        "tau_thrust": 0.065,
        "tau_rate": 0.060,
        "drag": 0.005,
        "max_speed": 60.0,
        "max_tilt_deg": 70.0,
        "workspace_xy": 5000.0,
        "min_altitude": 4.0,
        "max_altitude": 230.0,
    },
    "policy": {
        "decision_steps": 10,
        "position_error_cap_m": 0.0,
        "hidden_actor": [128, 128],
        "hidden_critic": [128, 128],
        "history_steps": 8,
        "initial_std": 0.07,
        "min_std": 0.02,
        "max_std": 0.3,
        "exploration_time_s": 0.02,
    },
    "prediction": {
        "enabled": True,
        "mode": "lstm",
        "history_steps": 40,
        "horizons_s": [0.05, 0.1, 0.2],
        "hidden_size": 64,
        "position_scale": 5.0,
        "lr": 1e-4,
        "epochs": 2,
        "batch_size": 256,
        "max_samples": 16384,
    },
    "critic": {
        "value_norm": True,
        "norm_beta": 0.995,
        "norm_min_variance": 0.01,
        "value_clip": 0.2,
        "huber_delta": 1.0,
    },
    "training": {
        "physics_steps": 20000000,
        "rollout_steps": 512,
        "batch_size": 2048,
        "epochs": 3,
        "actor_lr": 3e-5,
        "critic_lr": 1e-4,
        "discount_time_s": 20.0,
        "advantage_time_s": 3.0,
        "validation_every": 2000000,
        "validation_episodes": 128,
        "validation_seed": 78000,
        "mc_episodes": 16,
        "mc_calibration_every_updates": 16,
        "mc_calibration_updates": 32,
        "mc_max_regression_fraction": 0.02,
        "replay_recent_fraction": 0.2,
        "replay_anchor_fraction": 0.2,
        "critic_updates": 600,
        "bc_samples": 40000,
        "bc_updates": 0,
        "bc_rounds": 1,
        "aux_pd_coef": 0.0,
        "bc_stride": 2,
        "bc_segment_seconds": 0.5,
        "initialization_episodes": 32,
        "initialization_success_floor": 0.0,
        "zero_initial_actor_mean": False,
        "save_every_updates": 1,
        "prediction_updates_enabled": True,
        "minimum_success": 0.95,
        "teacher_method": "pn_hybrid",
        "teacher_pn_gain": 3.0,
        "teacher_switch_distance_m": 12.0,
        "teacher_accel_limit": 16.0,
        "teacher_tilt_limit_deg": 65.0,
    },
    "safety": {
        "predictive_enabled": False,
        "constraint_enabled": False,
        "horizon_s": 0.12,
        "prediction_step_s": 0.01,
        "warning_deg": 55.0,
        "uncertainty_deg": 2.0,
        "recovery_rate_gain": 4.0,
        "risk_cost_weight": 1.0,
        "failure_cost": 1.0,
        "cost_limit": 0.025,
        "lagrangian_lr": 0.05,
        "lagrangian_max": 10.0,
        "cost_critic_lr": 1e-4,
        "discount_time_s": 20.0,
        "advantage_time_s": 3.0,
        "critical_reset_fraction": 0.0,
        "curriculum_ramp_fraction": 0.4,
        "critical_tilt_min_deg": 35.0,
        "critical_tilt_max_deg": 50.0,
        "critical_outward_rate": 0.5,
    },
    "reward": {
        "scale": 0.1,
        "tilt": 0.08,
        "tilt_guard": 0.0,
        "actuator_risk": 0.0,
        "tilt_guard_start_deg": 55.0,
        "tilt_guard_lookahead_s": 0.12,
        "rate": 0.01,
        "progress": 0.3,
        "action_derivative": 1e-5,
        "completion_bonus": 10.0,
        "failure": 10.0,
        "braking": 0.5,
        "entry_velocity": 0.5,
        "time": 0.02,
    },
    "task": {
        "name": "01_static",
        "kind": "hover",
        "duration": 12.0,
        "distance": [2.0, 5.0],
        "speed": [0.0, 0.0],
        "max_accel": 0.0,
        "jerk_limit": 20.0,
        "switch_interval": [0.5, 1.5],
        "ou_time": 0.5,
        "response_time_s": 0.4,
        "radius": 0.5,
        "initial_tilt_deg": 0.0,
        "initial_velocity_noise": 0.0,
        "randomization": 0.0,
        "encounter": "random",
        "initial_pursuer_heading": "random",
        "initial_pursuer_speed": [0.0, 0.0],
        "trimmed_initialization": False,
        "entry_relative_speed_max_mps": 15.0,
        "entry_brake_accel_mps2": 12.0,
        "entry_velocity_distance_m": 8.0,
        "entry_velocity_scale_mps": 15.0,
    },
}


def merge(base, update, path=""):
    """Strict recursive merge: stale or misspelled keys are rejected."""
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration field: {path}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected object: {path}{key}")
            merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = copy.deepcopy(value)
    return base


def decision_dt(c):
    return c["physics"]["dt"] * c["policy"]["decision_steps"]


def observation_dims(c):
    base = 34 + 4 * c["policy"]["history_steps"]
    history = c["prediction"]["history_steps"] * 10 if c["prediction"]["enabled"] else 0
    return base, base + history


def validate(c):
    if c.get("schema") != SCHEMA or c["task"]["kind"] not in KINDS:
        raise ValueError("Expected first-entry CTBR schema and task kind")
    if not isinstance(c["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", c["device"]):
        raise ValueError("Invalid torch device")
    if not isinstance(c["seed"], int) or c["seed"] < 0 or type(c["num_envs"]) is not int or c["num_envs"] < 1:
        raise ValueError("Invalid seed/num_envs")
    def finite(v):
        if isinstance(v, dict):
            return all(finite(x) for x in v.values())
        if isinstance(v, list):
            return all(finite(x) for x in v)
        return not isinstance(v, (int, float)) or math.isfinite(v)
    if not finite(c):
        raise ValueError("Nonfinite configuration")
    p, t, tr, pol = (c[k] for k in ("physics", "task", "training", "policy"))
    reward = c["reward"]
    if p["dt"] != 0.001 or p["substeps"] != 1:
        raise ValueError("Physics integration must be 1 ms")
    if not 0 <= p["thrust_min"] < p["hover_thrust"] < p["thrust_max"]:
        raise ValueError("Invalid CTBR thrust endpoints")
    if p["gravity"] <= 0 or p["max_load_factor_g"] <= 1:
        raise ValueError("Invalid gravity or maximum load factor")
    if p["thrust_max"] > p["gravity"] * p["max_load_factor_g"] + 1e-7:
        raise ValueError("Thrust maximum exceeds declared load factor")
    for k in ("rate_xy", "rate_yaw", "tau_thrust", "tau_rate", "max_speed", "workspace_xy"):
        if p[k] <= 0:
            raise ValueError(f"Invalid physics.{k}")
    if not 0 < p["max_tilt_deg"] < 90 or not p["min_altitude"] < 100 < p["max_altitude"] or p["drag"] < 0:
        raise ValueError("Invalid flight envelope")
    if type(pol["decision_steps"]) is not int or pol["decision_steps"] < 1:
        raise ValueError("Invalid decision_steps")
    if pol["position_error_cap_m"] < 0 or type(pol["history_steps"]) is not int or pol["history_steps"] < 1:
        raise ValueError("Invalid position cap or action history")
    for key in ("hidden_actor", "hidden_critic"):
        if not pol[key] or any(type(v) is not int or v <= 0 for v in pol[key]):
            raise ValueError(f"Invalid policy.{key}")
    if not 0 < pol["min_std"] <= pol["initial_std"] <= pol["max_std"] <= 1 or pol["exploration_time_s"] <= 0:
        raise ValueError("Invalid exploration settings")
    dt = decision_dt(c)
    pr = c["prediction"]
    if pr["mode"] not in ("lstm", "constant_velocity", "constant_acceleration") or type(pr["enabled"]) is not bool:
        raise ValueError("Invalid predictor selection")
    for key in ("history_steps", "hidden_size", "epochs", "batch_size", "max_samples"):
        if type(pr[key]) is not int or pr[key] < 1:
            raise ValueError(f"Invalid prediction.{key}")
    if pr["position_scale"] <= 0 or pr["lr"] <= 0:
        raise ValueError("Invalid predictor learning settings")
    hs = pr["horizons_s"]
    if not hs or any(h <= 0 for h in hs) or sorted(set(hs)) != hs or any(abs(round(h / dt) * dt - h) > 1e-9 for h in hs):
        raise ValueError("Prediction horizons must align with decision ticks")
    cr = c["critic"]
    if type(cr["value_norm"]) is not bool or not 0 <= cr["norm_beta"] < 1 or cr["norm_min_variance"] <= 0 or cr["value_clip"] < 0 or cr["huber_delta"] <= 0:
        raise ValueError("Invalid critic settings")
    for key in ("physics_steps", "bc_updates", "critic_updates", "mc_calibration_every_updates", "mc_calibration_updates"):
        if type(tr[key]) is not int or tr[key] < 0:
            raise ValueError(f"Invalid training.{key}")
    if type(tr["bc_rounds"]) is not int or tr["bc_rounds"] < 1:
        raise ValueError("Invalid training.bc_rounds")
    if tr["bc_updates"] > 0 and tr["bc_rounds"] > tr["bc_updates"]:
        raise ValueError("training.bc_rounds cannot exceed bc_updates")
    if not 0 <= tr["aux_pd_coef"] <= 1:
        raise ValueError("training.aux_pd_coef must be in [0, 1]")
    if (reward["tilt_guard"] < 0
            or not 0 <= reward["tilt_guard_start_deg"] < p["max_tilt_deg"]
            or not 0 <= reward["tilt_guard_lookahead_s"] <= 1.0):
        raise ValueError("Invalid predictive tilt guard parameters")
    safe = c["safety"]
    if any(type(safe[k]) is not bool for k in ("predictive_enabled", "constraint_enabled")):
        raise ValueError("Invalid safety enable flags")
    if safe["constraint_enabled"] and not safe["predictive_enabled"]:
        raise ValueError("Constrained PPO needs the action-aware risk predictor")
    if (safe["horizon_s"] < dt or safe["prediction_step_s"] <= 0
            or abs(round(safe["horizon_s"] / safe["prediction_step_s"])
                   * safe["prediction_step_s"] - safe["horizon_s"]) > 1e-9
            or abs(round(dt / safe["prediction_step_s"])
                   * safe["prediction_step_s"] - dt) > 1e-9):
        raise ValueError("Safety prediction step must divide decision and horizon durations")
    if (not 0 <= safe["warning_deg"] < p["max_tilt_deg"]
            or not 0 <= safe["uncertainty_deg"] < 20
            or safe["recovery_rate_gain"] <= 0):
        raise ValueError("Invalid attitude safety forecast settings")
    for key in ("risk_cost_weight", "failure_cost", "cost_limit",
                "lagrangian_lr", "lagrangian_max", "cost_critic_lr",
                "discount_time_s", "advantage_time_s"):
        if safe[key] <= 0:
            raise ValueError(f"Invalid safety.{key}")
    if safe["advantage_time_s"] > safe["discount_time_s"]:
        raise ValueError("Safety advantage horizon exceeds discount horizon")
    if not 0 <= safe["critical_reset_fraction"] < 1 or not 0 < safe["curriculum_ramp_fraction"] <= 1:
        raise ValueError("Invalid safety curriculum fractions")
    if not (0 <= safe["critical_tilt_min_deg"] <= safe["critical_tilt_max_deg"]
            < p["max_tilt_deg"]) or not 0 <= safe["critical_outward_rate"] <= p["rate_xy"]:
        raise ValueError("Invalid critical attitude reset settings")
    if reward["actuator_risk"] > 0 and not safe["predictive_enabled"]:
        raise ValueError("actuator_risk reward needs predictive_enabled")
    for key in ("rollout_steps", "batch_size", "epochs", "validation_every", "validation_episodes", "mc_episodes", "bc_samples", "bc_stride", "save_every_updates", "initialization_episodes"):
        if type(tr[key]) is not int or tr[key] < 1:
            raise ValueError(f"Invalid training.{key}")
    for key in ("actor_lr", "critic_lr", "discount_time_s", "advantage_time_s", "bc_segment_seconds"):
        if tr[key] <= 0:
            raise ValueError(f"Invalid training.{key}")
    if tr["advantage_time_s"] > tr["discount_time_s"] or abs(round(tr["bc_segment_seconds"] / dt) * dt - tr["bc_segment_seconds"]) > 1e-9:
        raise ValueError("Invalid GAE timescale or BC segment")
    if type(tr["prediction_updates_enabled"]) is not bool or type(tr["zero_initial_actor_mean"]) is not bool:
        raise ValueError("Invalid training switches")
    if not 0 <= tr["initialization_success_floor"] <= 1 or not 0 <= tr["minimum_success"] <= 1:
        raise ValueError("Invalid success thresholds")
    fractions = [tr["replay_recent_fraction"], tr["replay_anchor_fraction"]]
    if any(not 0 <= f < 1 for f in fractions) or sum(fractions) >= 1 or tr["mc_max_regression_fraction"] < 0:
        raise ValueError("Invalid replay/MC calibration fractions")
    if tr["teacher_method"] not in ("pd", "pn", "pn_hybrid", "pd_observed", "pd_observed_smoothed") or tr["teacher_pn_gain"] <= 0 or tr["teacher_switch_distance_m"] <= 0 or tr["teacher_accel_limit"] <= 0 or not 0 < tr["teacher_tilt_limit_deg"] <= p["max_tilt_deg"]:
        raise ValueError("Invalid geometric teacher")
    if t["encounter"] not in ("random", "tail", "head_on") or t["initial_pursuer_heading"] not in ("same", "opposite", "random"):
        raise ValueError("Invalid encounter/initial direction")
    for key in ("distance", "speed", "switch_interval", "initial_pursuer_speed"):
        bounds = t[key]
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
            raise ValueError(f"Invalid task.{key}")
    if t["distance"][0] <= t["radius"] or t["radius"] <= 0 or t["entry_relative_speed_max_mps"] <= 0:
        raise ValueError("First entry must start strictly outside the target sphere")
    if t["initial_pursuer_speed"][1] > p["max_speed"] or not 0 <= t["initial_tilt_deg"] < p["max_tilt_deg"] or t["initial_velocity_noise"] < 0 or not 0 <= t["randomization"] <= 1:
        raise ValueError("Invalid pursuer initialization")
    if type(t["trimmed_initialization"]) is not bool:
        raise ValueError("Invalid cruise trim switch")
    for key in ("duration", "entry_brake_accel_mps2", "entry_velocity_distance_m", "entry_velocity_scale_mps", "jerk_limit", "ou_time", "response_time_s"):
        if t[key] <= 0:
            raise ValueError(f"Invalid task.{key}")
    if abs(round(t["duration"] / p["dt"]) * p["dt"] - t["duration"]) > 1e-9 or t["switch_interval"][0] <= 0:
        raise ValueError("Invalid duration or switching interval")
    if t["max_accel"] < 0 or t["response_time_s"] < max(2*t["max_accel"]/t["jerk_limit"], p["dt"]):
        raise ValueError("Invalid acceleration/jerk profile")
    if t["kind"] == "random" and (t["speed"][1] <= 0 or t["max_accel"] <= 0):
        raise ValueError("Random maneuver needs speed and nonzero acceleration")
    if t["speed"][1] * t["duration"] + t["distance"][1] + 100 >= p["workspace_xy"]:
        raise ValueError("Target trajectory could exceed horizontal workspace")
    if t["kind"] == "random" and .75 * t["duration"] + 15 >= .5 * (p["max_altitude"]-p["min_altitude"]):
        raise ValueError("Random target could exceed altitude envelope")
    r=c["reward"]
    if any(v < 0 for v in r.values()) or r["scale"] <= 0:
        raise ValueError("Invalid first-entry reward weights")
    return c


def load(path=None, device=None, seed=None):
    c=copy.deepcopy(DEFAULT)
    if path:
        merge(c, json.loads(Path(path).read_text(encoding="utf8")))
    if device is not None:
        c["device"]=device
    if seed is not None:
        c["seed"]=seed
    return validate(c)


def fingerprint(c):
    value=copy.deepcopy(c)
    value.pop("device", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def interface(c):
    return {
        "schema": c["schema"],
        "action": "ctbr-normalized-thrust-and-body-rates-v1",
        "observation": "first-entry-oracle-ctbr-prediction-v1",
        "physics": c["physics"],
        "policy": {k:v for k,v in c["policy"].items() if k not in ("initial_std", "min_std")},
        "reward": c["reward"],
        "prediction": c["prediction"],
        "critic": c["critic"],
    }

def transfer_interface(c):
    """Contract required for warm-starting weights across reward ablations.

    Reward weights may change because the transferred Critic is recalibrated
    before PPO. Action, observation, dynamics, predictor and Critic architecture
    must remain identical.
    """
    contract = copy.deepcopy(interface(c))
    contract.pop("reward")
    return contract
