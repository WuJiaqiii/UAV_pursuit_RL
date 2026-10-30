"""Self-contained versioned direct-CTBR experiment configuration."""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "pursuit-ctbr-v2"
DEFAULT = {
    "schema": SCHEMA,
    "seed": 42,
    "device": "cpu",
    "num_envs": 32,
    "physics": {
        "dt": 0.001,
        "substeps": 1,
        "gravity": 9.81,
        "hover_thrust": 9.81,
        "thrust_min": 0.0,
        "thrust_max": 30.0,
        "rate_xy": 1.2,
        "rate_yaw": 1.0,
        "tau_thrust": 0.1,
        "tau_rate": 0.13,
        "drag": 0.007,
        "max_speed": 30.0,
        "max_tilt_deg": 65.0,
        "workspace_xy": 2000.0,
        "min_altitude": 4.0,
        "max_altitude": 230.0,
    },
    "policy": {
        "hidden_actor": [128, 128],
        "hidden_critic": [128, 128],
        "history_steps": 8,
        "initial_std": 0.07,
        "min_std": 0.02,
        "max_std": 0.3,
        "exploration_time_s": 0.02,
    },
    "training": {
        "physics_steps": 20000000,
        "rollout_steps": 4096,
        "batch_size": 4096,
        "epochs": 3,
        "actor_lr": 3e-5,
        "critic_lr": 1e-4,
        "discount_time_s": 20.0,
        "advantage_time_s": 3.0,
        "validation_every": 2000000,
        "validation_episodes": 64,
        "validation_seed": 78000,
        "mc_episodes": 32,
        "critic_updates": 600,
        "bc_samples": 40000,
        "bc_updates": 2000,
        "dagger_rounds": 1,
        "dagger_updates": 500,
        "bc_stride": 20,
        "bc_segment_seconds": 0.5,
        "initialization_episodes": 32,
        "initialization_success_floor": 0.9,
        "save_every_updates": 1,
        "transfer_dagger_updates": 0,
    },
    "reward": {
        "scale": 0.1,
        "position": 2.0,
        "position_length": 4.0,
        "relative_velocity": 0.5,
        "velocity_length": 3.0,
        "survival": 0.1,
        "tilt": 0.15,
        "rate": 0.01,
        "progress": 0.1,
        "action_derivative": 1e-5,
        "acquisition_bonus": 3.0,
        "failure": 10.0,
        "completion_bonus": 5.0,
    },
    "task": {
        "name": "hover_small",
        "kind": "hover",
        "duration": 20.0,
        "distance": [0.0, 0.3],
        "speed": [0.0, 0.0],
        "max_accel": 0.0,
        "jerk_limit": 20.0,
        "switch_interval": [0.5, 1.5],
        "ou_time": 0.5,
        "radius": 0.75,
        "velocity_tolerance": 1.0,
        "tail_seconds": 8.0,
        "minimum_tail_fraction": 0.9,
        "initial_tilt_deg": 5.0,
        "initial_velocity_noise": 0.1,
        "match_initial_velocity": False,
        "randomization": 0.0,
    },
}
KINDS = ("hover", "goto", "cruise", "circle", "maneuver", "random", "random_high")


def merge(base, update, path=""):
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


def validate(c):
    if not isinstance(c["device"], str) or not re.fullmatch(
        r"cpu|cuda(?::[0-9]+)?", c["device"]
    ):
        raise ValueError("Device must be cpu, cuda or cuda:N")
    if c["schema"] != SCHEMA or c["task"]["kind"] not in KINDS:
        raise ValueError("Unknown CTBR schema/task kind")

    def finite(x):
        if isinstance(x, dict):
            return all(finite(v) for v in x.values())
        if isinstance(x, list):
            return all(finite(v) for v in x)
        return (
            not isinstance(x, (float, int))
            or (not isinstance(x, bool) and math.isfinite(x))
            or isinstance(x, bool)
        )

    if not finite(c):
        raise ValueError("Nonfinite configuration")
    for key in ("seed", "num_envs"):
        if type(c[key]) is not int or c[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    p, t, tr, policy = c["physics"], c["task"], c["training"], c["policy"]
    if p["dt"] != 0.001 or p["substeps"] != 1:
        raise ValueError("This contract uses 1 kHz physics AND fresh CTBR decisions")
    if not 0 <= p["thrust_min"] < p["hover_thrust"] < p["thrust_max"]:
        raise ValueError("Invalid thrust mapping endpoints")
    for k in (
        "gravity",
        "hover_thrust",
        "rate_xy",
        "rate_yaw",
        "tau_thrust",
        "tau_rate",
        "max_speed",
        "workspace_xy",
    ):
        if p[k] <= 0:
            raise ValueError(f"Invalid physics.{k}")
    if (
        not 0 < p["max_tilt_deg"] < 90
        or not p["min_altitude"] < 100 < p["max_altitude"]
    ):
        raise ValueError("Invalid flight envelope")
    if p["max_altitude"] - p["min_altitude"] < 40 or p["drag"] < 0:
        raise ValueError("Invalid altitude span or drag")
    if policy["history_steps"] < 1 or type(policy["history_steps"]) is not int:
        raise ValueError("history_steps must be a positive integer")
    if (
        not 0 < policy["min_std"] <= policy["initial_std"] <= policy["max_std"] <= 1
        or policy["exploration_time_s"] <= 0
    ):
        raise ValueError("Invalid correlated exploration")
    for k in ("hidden_actor", "hidden_critic"):
        if not policy[k] or any(type(v) is not int or v < 1 for v in policy[k]):
            raise ValueError("Invalid network architecture")
    for k in (
        "physics_steps",
        "bc_updates",
        "dagger_rounds",
        "dagger_updates",
        "critic_updates",
        "transfer_dagger_updates",
    ):
        if type(tr[k]) is not int or tr[k] < 0:
            raise ValueError(f"Invalid training.{k}")
    for k in (
        "rollout_steps",
        "batch_size",
        "epochs",
        "validation_every",
        "validation_episodes",
        "mc_episodes",
        "bc_samples",
        "bc_stride",
        "save_every_updates",
        "initialization_episodes",
    ):
        if type(tr[k]) is not int or tr[k] < 1:
            raise ValueError(f"Invalid training.{k}")
    for k in ("actor_lr", "critic_lr", "discount_time_s", "advantage_time_s"):
        if tr[k] <= 0:
            raise ValueError(f"Invalid training.{k}")
    if (
        tr["bc_segment_seconds"] <= 0
        or not 0 <= tr["initialization_success_floor"] <= 1
    ):
        raise ValueError("Invalid initialization segment or success floor")
    if (
        abs(
            round(tr["bc_segment_seconds"] / p["dt"]) * p["dt"]
            - tr["bc_segment_seconds"]
        )
        > 1e-9
    ):
        raise ValueError("BC segment must be an integer number of physics ticks")
    if tr["advantage_time_s"] > tr["discount_time_s"]:
        raise ValueError("GAE lambda would exceed one")
    for k in ("distance", "speed", "switch_interval"):
        v = t[k]
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"Invalid task.{k}")
    if (
        t["switch_interval"][0] <= 0
        or t["ou_time"] <= 0
        or t["jerk_limit"] <= 0
        or t["max_accel"] < 0
    ):
        raise ValueError("Invalid target process")
    if (
        not 0 < t["tail_seconds"] <= t["duration"]
        or t["radius"] <= 0
        or t["velocity_tolerance"] <= 0
    ):
        raise ValueError("Invalid task acceptance window")
    for key in ("duration", "tail_seconds"):
        if abs(round(t[key] / p["dt"]) * p["dt"] - t[key]) > 1e-9:
            raise ValueError(f"task.{key} must be an integer number of physics ticks")
    if not 0 <= t["randomization"] <= 1 or not 0 <= t["minimum_tail_fraction"] <= 1:
        raise ValueError("Invalid task acceptance/randomization")
    if (
        t["initial_velocity_noise"] < 0
        or not 0 <= t["initial_tilt_deg"] < p["max_tilt_deg"]
    ):
        raise ValueError("Invalid initial perturbations")
    if t["kind"] in ("circle", "maneuver", "random", "random_high") and (
        t["max_accel"] <= 0 or t["speed"][1] <= 0
    ):
        raise ValueError("Moving maneuver task needs positive speed and acceleration")
    if t["speed"][1] * t["duration"] + t["distance"][1] + 100 >= p["workspace_xy"]:
        raise ValueError(
            "Target travel bound exceeds workspace; no boundary pullback is used"
        )
    if (
        t["kind"] == "circle"
        and t["speed"][1] ** 2 / t["max_accel"] + 100 >= p["workspace_xy"]
    ):
        raise ValueError("Circle radius exceeds workspace")
    if t["kind"] in ("maneuver", "random", "random_high") and 0.75 * t[
        "duration"
    ] + 15 >= 0.5 * (p["max_altitude"] - p["min_altitude"]):
        raise ValueError("Target vertical travel bound exceeds altitude envelope")
    if (
        min(c["reward"].values()) < 0
        or c["reward"]["scale"] <= 0
        or c["reward"]["position_length"] <= 0
        or c["reward"]["velocity_length"] <= 0
    ):
        raise ValueError("Invalid reward settings")
    return c


def load(path=None, device=None, seed=None):
    c = copy.deepcopy(DEFAULT)
    if path:
        merge(c, json.loads(Path(path).read_text()))
    if device is not None:
        c["device"] = device
    if seed is not None:
        c["seed"] = seed
    return validate(c)


def fingerprint(c):
    c = copy.deepcopy(c)
    c.pop("device")
    return hashlib.sha256(
        json.dumps(c, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def interface(c):
    return {
        "schema": SCHEMA,
        "action": "ctbr-piecewise-hover-thrust-v2",
        "observation": "ned-actuator-raw-history-goal-v2",
        "physics": c["physics"],
        "policy": {
            k: v for k, v in c["policy"].items() if k not in ("initial_std", "min_std")
        },
        "reward": c["reward"],
    }
