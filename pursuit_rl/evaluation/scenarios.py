"""Shared named holdout scenarios and Wilson binomial interval."""

import copy

SEED_HOLDOUT = 930000
# Named holdout tasks use a fixed, independently versioned distribution.
BENCHMARK_VERSION = "v6-fixed-holdout"
SCENARIOS = {
    "ood_random_2m": {
        "description": "Unseen rapid hybrid, 2m",
        "target": {
            "max_accel": 9.0,
            "maneuver_interval": [0.6, 1.4],
            "hybrid_smoothing_s": 0.3,
            "strict_speed_ceiling": True,
        },
        "stage": {
            "modes": ["hybrid"],
            "distance": [300.0, 500.0],
            "speed": [18.0, 25.0],
            "time": 150.0,
            "max_accel": 9.0,
            "success_radius": 2.0,
        },
    },
    "ood_switchback_2m": {
        "description": "Unseen alternating hybrid, 2m",
        "target": {
            "max_accel": 9.0,
            "maneuver_interval": [0.6, 1.4],
            "hybrid_smoothing_s": 0.3,
            "strict_speed_ceiling": True,
            "hybrid_pattern": "switchback",
        },
        "stage": {
            "modes": ["hybrid"],
            "distance": [300.0, 500.0],
            "speed": [18.0, 25.0],
            "time": 150.0,
            "max_accel": 9.0,
            "success_radius": 2.0,
        },
    },
    "ood_pulse_2m": {
        "description": "Unseen pulse hybrid, 2m",
        "target": {
            "max_accel": 9.0,
            "maneuver_interval": [0.6, 1.4],
            "hybrid_smoothing_s": 0.3,
            "strict_speed_ceiling": True,
            "hybrid_pattern": "pulse",
        },
        "stage": {
            "modes": ["hybrid"],
            "distance": [300.0, 500.0],
            "speed": [18.0, 25.0],
            "time": 150.0,
            "max_accel": 9.0,
            "success_radius": 2.0,
        },
    },
    "standard": {"description": "Original Stage 3 distribution (held-out test seeds)"},
    "far_fast": {
        "description": "Initial 350-500 m, target 18-25 m/s",
        "stage": {"distance": [350.0, 500.0], "speed": [18.0, 25.0]},
    },
    "edge_fast": {
        "description": "Near worst case 400-500 m, 22-25 m/s",
        "stage": {"distance": [400.0, 500.0], "speed": [22.0, 25.0]},
    },
    "long_180s": {
        "description": "Long 180 s time allowance; catch still terminates immediately",
        "stage": {"time": 180.0, "distance": [350.0, 500.0], "speed": [18.0, 25.0]},
    },
    "rapid_hybrid": {
        "description": "Hybrid target rapid switch: 0.5-1.5 s",
        "stage": {
            "modes": ["hybrid"],
            "distance": [250.0, 500.0],
            "speed": [18.0, 25.0],
        },
        "target": {"maneuver_interval": [0.5, 1.5]},
    },
    "singer_fast": {
        "description": "Singer maneuver exclusively, sigma 6 m/s2 stationary OU",
        "stage": {
            "modes": ["singer"],
            "distance": [300.0, 500.0],
            "speed": [18.0, 25.0],
        },
        "target": {"singer_sigma": 6.0},
    },
    "plant_lag": {
        "description": "Flight-control lag increased and drag increased",
        "physics": {"tau_thrust": 0.20, "tau_rate": 0.24, "drag": 0.012},
    },
    "high_accel_rapid": {
        "description": "Compound OOD: up to 10 m/s2 and rapid maneuvers",
        "target": {
            "max_accel": 10.0,
            "singer_sigma": 5.0,
            "maneuver_interval": [0.7, 2.0],
        },
        "stage": {"max_accel": 10.0, "distance": [250.0, 500.0], "speed": [18.0, 25.0]},
    },
    "precision_2m": {
        "description": "Interception event radius reduced 5 -> 2 m",
        "stage": {"success_radius": 2.0},
    },
    "precision_3m": {
        "description": "Three-metre capture diagnostic",
        "stage": {"success_radius": 3.0},
    },
    "precision_1m": {
        "description": "First interception within 1m (diagnostic only)",
        "stage": {"success_radius": 1.0},
    },
    "overtaking_edge": {
        "description": "Same-direction chase, 350-500m and 20-25m/s",
        "stage": {
            "distance": [350.0, 500.0],
            "speed": [20.0, 25.0],
            "initial_geometry": "overtaking",
        },
    },
    "head_on_edge": {
        "description": "Head-on approach, 350-500m and 20-25m/s",
        "stage": {
            "distance": [350.0, 500.0],
            "speed": [20.0, 25.0],
            "initial_geometry": "head_on",
        },
    },
    "crossing_edge": {
        "description": "Crossing pursuit, 350-500m and 20-25m/s",
        "stage": {
            "distance": [350.0, 500.0],
            "speed": [20.0, 25.0],
            "initial_geometry": "crossing",
        },
    },
    "switchback_hybrid": {
        "description": "Lateral alternating hybrid maneuvers",
        "target": {"hybrid_pattern": "switchback", "maneuver_interval": [0.5, 1.5]},
        "stage": {
            "modes": ["hybrid"],
            "distance": [250.0, 500.0],
            "speed": [18.0, 25.0],
        },
    },
    "pulse_hybrid": {
        "description": "Alternating hybrid acceleration/braking",
        "target": {"hybrid_pattern": "pulse", "maneuver_interval": [0.5, 1.5]},
        "stage": {
            "modes": ["hybrid"],
            "distance": [250.0, 500.0],
            "speed": [18.0, 25.0],
        },
    },
    "strict_target_speed": {
        "description": "Target speed ceiling enforced at 25 m/s",
        "target": {"strict_speed_ceiling": True},
        "stage": {"distance": [350.0, 500.0], "speed": [20.0, 25.0]},
    },
    "high_corr_singer": {
        "description": "Correlated Singer acceleration with alpha 0.3",
        "target": {"singer_alpha": 0.3, "singer_sigma": 5.0},
        "stage": {
            "modes": ["singer"],
            "distance": [250.0, 500.0],
            "speed": [18.0, 25.0],
        },
    },
}


def evaluation_stage(cfg):
    return next(
        i for i, stage in enumerate(cfg["curriculum"]) if stage["name"] == "stage3"
    )


def configuration(base, name):
    from pursuit_rl.config import DEFAULT, validate

    c = copy.deepcopy(base)
    c["physics"] = copy.deepcopy(DEFAULT["physics"])
    # Fixed benchmark physical limits remain standard, but experimental time
    # resolution and policy scheduling MUST be inherited from the caller.
    # Otherwise a 1 kHz --config would silently run scenarios at 20 Hz.
    for key in ("dt", "substeps", "policy_interval_steps"):
        if key in base["physics"]:
            c["physics"][key] = base["physics"][key]
    c["target"] = copy.deepcopy(DEFAULT["target"])
    c["curriculum"] = copy.deepcopy(DEFAULT["curriculum"])
    stage = evaluation_stage(c)
    changes = SCENARIOS[name]
    for section in ("physics", "target"):
        c[section].update(changes.get(section, {}))
    c["curriculum"][stage].update(changes.get("stage", {}))
    # The base retains the checkpoint's input encoding, including acceleration scale.
    c["training"]["initial_stage"] = 0
    c["training"]["max_stage"] = len(c["curriculum"]) - 1
    validate(c)
    return c
