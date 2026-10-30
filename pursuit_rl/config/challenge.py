"""Explicit OOD tasks and interface compatibility, without reinterpreting old actions."""

import copy
import math
from pursuit_rl.config.rescue import rescue_contract
from pursuit_rl.config.schema import validate

BASELINE_TASK = {
    "modes": ["straight"],
    "distance": [10.0, 30.0],
    "speed": [0.0, 5.0],
    "time": 40.0,
    "max_accel": 2.0,
    "success_radius": 2.0,
    "initial_geometry": "random",
}

BASELINE_TARGET = {
    "max_speed": 25.0,
    "max_accel": 7.0,
    "singer_alpha": 1.0,
    "singer_sigma": 3.0,
    "turn_rate": 0.45,
    "maneuver_interval": [1.5, 5.0],
    "hybrid_pattern": "random",
    "hybrid_smoothing_s": 0.4,
    "strict_speed_ceiling": False,
    "target_workspace_xy": 450.0,
    "target_min_altitude": 30.0,
    "target_max_altitude": 190.0,
}

# Every scenario overrides the saved task rather than silently changing a checkpoint.
SCENARIOS = {
    "baseline": {},
    "training_task": {},
    "singer_a05": {"stage": {"modes": ["singer"], "max_accel": 0.5}},
    "singer_a1": {"stage": {"modes": ["singer"], "max_accel": 1.0}},
    "singer_a2": {"stage": {"modes": ["singer"], "max_accel": 2.0}},
    "dubins_a2": {"stage": {"modes": ["dubins"], "max_accel": 2.0}},
    "hybrid_a2": {
        "stage": {"modes": ["hybrid"], "max_accel": 2.0},
        "target": {"hybrid_pattern": "switchback", "maneuver_interval": [1.5, 3.0]},
    },
    "distance_30_60": {"stage": {"distance": [30.0, 60.0], "time": 60.0}},
    "distance_60_100": {"stage": {"distance": [60.0, 100.0], "time": 60.0}},
    "speed_5_8": {"stage": {"speed": [5.0, 8.0], "time": 60.0}},
    "speed_8_12": {"stage": {"speed": [8.0, 12.0], "time": 60.0}},
    "speed_8_12_open": {
        "stage": {"speed": [8.0, 12.0], "time": 60.0},
        "target": {
            "target_workspace_xy": 2000.0,
            "target_min_altitude": 4.0,
            "target_max_altitude": 400.0,
        },
    },
    "combined_moderate": {
        "stage": {
            "modes": ["straight", "dubins", "singer"],
            "distance": [30.0, 60.0],
            "speed": [5.0, 8.0],
            "max_accel": 1.0,
            "time": 60.0,
        }
    },
    **{
        f"speed_8_12_{geometry}": {
            "stage": {"speed": [8.0, 12.0], "time": 60.0, "initial_geometry": geometry},
            "target": {
                "target_workspace_xy": 2000.0,
                "target_min_altitude": 4.0,
                "target_max_altitude": 400.0,
            },
        }
        for geometry in ("overtaking", "head_on", "crossing")
    },
    "actuator_lag_150": {"plant_multipliers": {"tau_rate": 1.5, "tau_thrust": 1.5}},
    "drag_150": {"plant_multipliers": {"drag": 1.5}},
    "speed_20_25_limit": {
        "stage": {"distance": [30.0, 60.0], "speed": [20.0, 25.0], "time": 60.0}
    },
}


def control_interface(cfg, profile):
    """Exclude task/target/reward; retain every input/output/controller semantic."""
    contract = rescue_contract(cfg, profile)
    for key in (
        "stage",
        "task",
        "target",
        "reward",
        "timeout_penalty",
        "reward_scale",
        "discount_time_s",
        "hidden_critic",
    ):
        contract.pop(key)
    return contract


def assert_transfer_compatible(source_cfg, source_profile, target_cfg, target_profile):
    source = control_interface(source_cfg, source_profile)
    target = control_interface(target_cfg, target_profile)
    different = [key for key in source if source[key] != target.get(key)]
    if different:
        raise ValueError("Transfer changes control interface: " + ", ".join(different))


def apply_scenario(cfg, profile, name):
    if name not in SCENARIOS:
        raise ValueError("Unknown challenge scenario: " + name)
    cfg = copy.deepcopy(cfg)
    profile = copy.deepcopy(profile)
    spec = copy.deepcopy(SCENARIOS[name])
    if name != "training_task":
        cfg["curriculum"][profile["stage"]].update(BASELINE_TASK)
        cfg["target"].update(BASELINE_TARGET)
    cfg["curriculum"][profile["stage"]].update(spec.get("stage", {}))
    cfg["target"].update(spec.get("target", {}))
    validate(cfg)
    return cfg, profile, spec.get("plant_multipliers", {})


def action_envelope(cfg, profile):
    info = {
        "action_mode": profile["action_mode"],
        "acceleration_norm_limit": profile["acceleration_limit"],
        "tilt_limit_deg": profile["tilt_limit_deg"],
        "plant_speed_limit": cfg["physics"]["max_speed"],
        "notice": "Command limits alone are not a proof of capture feasibility; geometry, time, drag and actuator lag matter.",
    }
    if profile["action_mode"] == "velocity":
        scale = profile["velocity_scale"]
        speed = cfg["curriculum"][profile["stage"]]["speed"][1]
        horizontal = math.hypot(*scale[:2])
        info.update(velocity_axis_limits=scale, max_horizontal_command=horizontal)
        info["target_initial_speed_max"] = speed
        info["configured_target_global_speed_limit"] = cfg["target"]["max_speed"]
        info["horizontal_command_warning"] = speed >= min(scale[:2])
    return info
