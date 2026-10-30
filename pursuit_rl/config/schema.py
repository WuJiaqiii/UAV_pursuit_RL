"""Authoritative packaged defaults and simulation/training field validation."""

import math
from importlib.resources import files
import yaml
from .values import as_dict

DEFAULT = yaml.safe_load(
    files("pursuit_rl.config").joinpath("defaults.yaml").read_text(encoding="utf8")
)


def validate(config):
    """Reject malformed values at the configuration boundary with named errors."""
    config = as_dict(config)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def validate_shape(value, default, path):
        if isinstance(default, dict):
            require(isinstance(value, dict), f"{path} must be a map")
            require(set(value) == set(default), f"{path} has missing or unknown fields")
            for key, item in default.items():
                validate_shape(value[key], item, f"{path}.{key}")
        elif isinstance(default, bool):
            require(type(value) is bool, f"{path} must be boolean")
        elif isinstance(default, int):
            require(type(value) is int, f"{path} must be an integer")
        elif isinstance(default, float):
            require(
                type(value) in (int, float) and math.isfinite(value),
                f"{path} must be a finite number",
            )
        elif isinstance(default, str):
            require(isinstance(value, str), f"{path} must be a string")
        elif isinstance(default, list):
            require(isinstance(value, list), f"{path} must be a list")

    validate_shape(config, DEFAULT, "config")
    physics, target, reward, training = (
        config[key] for key in ("physics", "target", "reward", "training")
    )
    require(
        config["seed"] >= 0 and config["torch_threads"] > 0,
        "seed must be nonnegative and torch_threads positive",
    )
    require(config["device"] in ("cpu", "cuda"), "device must be cpu or cuda")
    require(config["simulation_backend"] in ("numpy", "torch"), "simulation_backend must be numpy or torch")
    require(
        config["observation"]["target_accel_scale"] > 0,
        "observation.target_accel_scale must be positive",
    )
    for section, keys in [
        (
            physics,
            [
                "dt",
                "gravity",
                "hover_thrust",
                "thrust_max",
                "thrust_delta",
                "rate_xy",
                "rate_yaw",
                "tau_thrust",
                "tau_rate",
                "max_speed",
                "workspace_xy",
            ],
        ),
        (
            target,
            [
                "max_speed",
                "max_accel",
                "singer_alpha",
                "turn_rate",
                "hybrid_smoothing_s",
                "target_workspace_xy",
            ],
        ),
        (
            reward,
            [
                "distance_scale",
                "speed_scale",
                "cpa_horizon",
                "boundary_soft_distance",
                "precision_scale_m",
                "progress_scale_m",
            ],
        ),
        (
            training,
            [
                "num_envs",
                "n_steps",
                "batch_size",
                "epochs",
                "total_steps",
                "eval_every",
                "eval_episodes",
                "promotion_consecutive",
                "actor_lr",
                "critic_lr",
                "max_grad_norm",
            ],
        ),
    ]:
        for key in keys:
            require(section[key] > 0, f"{key} must be positive")
    require(
        type(physics["policy_interval_steps"]) is int
        and physics["policy_interval_steps"] >= 1,
        "physics.policy_interval_steps must be a positive integer",
    )
    require(
        type(config["evaluation"]["trace_stride"]) is int
        and config["evaluation"]["trace_stride"] >= 1,
        "evaluation.trace_stride must be a positive integer",
    )
    require(
        physics["dt"] <= 0.1 and physics["substeps"] >= 1,
        "dt must be <=0.1 and substeps >=1",
    )
    require(
        0 <= physics["thrust_min"] < physics["hover_thrust"] < physics["thrust_max"],
        "Invalid thrust limits",
    )
    require(
        physics["drag"] >= 0 and 0 < physics["max_tilt_deg"] < 90,
        "Invalid drag or tilt limit",
    )
    require(
        0 <= physics["min_altitude"] < physics["max_altitude"],
        "Invalid pursuer altitude range",
    )
    require(
        0 <= target["target_min_altitude"] < target["target_max_altitude"],
        "Invalid target altitude range",
    )
    require(target["singer_sigma"] >= 0, "target.singer_sigma must be nonnegative")
    require(
        target["hybrid_pattern"] in ("random", "switchback", "pulse"),
        "Unknown hybrid_pattern",
    )

    def interval(values, name, minimum=0):
        require(
            len(values) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in values),
            f"{name} requires two finite numbers",
        )
        require(minimum <= values[0] <= values[1], f"Invalid {name} interval")

    interval(target["maneuver_interval"], "target.maneuver_interval")
    require(target["maneuver_interval"][0] > 0, "maneuver_interval must be positive")
    require(reward["cpa_mode"] in ("direct", "potential"), "Unknown reward.cpa_mode")
    for key, value in reward.items():
        if key.startswith("w_") or key in (
            "success_bonus",
            "failure_penalty",
            "predict_horizon_s",
            "near_action_delta_factor",
            "closing_decay_distance_m",
        ):
            require(value >= 0, f"reward.{key} must be nonnegative")
    for key in ("soft_speed_frac", "soft_tilt_frac"):
        require(0 <= reward[key] < 1, f"reward.{key} must be in [0,1)")
    require(
        -6.0 <= training["log_std_min"] <= training["log_std_init"] <= 1.0,
        "training.log_std_min must be <= log_std_init and within [-6, 1]",
    )
    require(
        0 < training["gamma"] < 1 and 0 < training["gae_lambda"] <= 1,
        "Invalid PPO discount/GAE",
    )
    require(
        0 < training["clip"] < 1
        and training["ent_coef"] >= 0
        and training["target_kl"] >= 0,
        "Invalid PPO clip/entropy/KL",
    )
    require(
        0 <= training["promotion_success"] <= 1 and training["min_stage_steps"] >= 0,
        "Invalid promotion thresholds",
    )
    require(training["eval_seed"] >= 0, "eval_seed must be nonnegative")
    require(
        0 <= training["aux_pd_coef"] <= 10,
        "training.aux_pd_coef must be between 0 and 10",
    )
    require(
        training["aux_pd_schedule"] in ("constant", "linear"),
        "training.aux_pd_schedule must be constant or linear",
    )
    require(
        type(training["aux_pd_decay_steps"]) is int and training["aux_pd_decay_steps"] > 0,
        "training.aux_pd_decay_steps must be positive",
    )
    require(
        type(training["challenge_eval_enabled"]) is bool,
        "training.challenge_eval_enabled must be boolean",
    )
    for key in ("hidden_actor", "hidden_critic"):
        require(
            bool(training[key])
            and all(type(v) is int and v > 0 for v in training[key]),
            f"{key} requires positive integer widths",
        )
    require(
        all(type(v) is int and v > 0 for v in training["snapshot_steps"]),
        "snapshot_steps requires positive integers",
    )
    stages = config["curriculum"]
    require(bool(stages), "curriculum must be a nonempty list")
    require(
        0 <= training["initial_stage"] <= training["max_stage"] < len(stages),
        "Invalid initial_stage/max_stage",
    )
    names = []
    required = {
        "name",
        "distance",
        "speed",
        "time",
        "success_radius",
        "max_accel",
        "modes",
    }
    for index, stage in enumerate(stages):
        label = f"curriculum[{index}]"
        require(
            isinstance(stage, dict)
            and required <= set(stage) <= required | {"initial_geometry"},
            f"{label} has missing or unknown fields",
        )
        require(
            isinstance(stage["name"], str) and bool(stage["name"]),
            f"{label}.name must be nonempty",
        )
        names.append(stage["name"])
        for key in ("distance", "speed"):
            require(isinstance(stage[key], list), f"{label}.{key} must be a list")
            interval(stage[key], f"{label}.{key}")
        require(
            stage["distance"][0] > 0
            and stage["distance"][1] < physics["workspace_xy"] * 1.2,
            f"{label} distance exceeds supported workspace",
        )
        require(
            stage["speed"][1] <= target["max_speed"],
            f"{label}.speed exceeds target.max_speed",
        )
        for key in ("time", "success_radius", "max_accel"):
            value = stage[key]
            require(
                type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"{label}.{key} must be finite and positive",
            )
        require(
            stage["max_accel"] <= target["max_accel"],
            f"{label}.max_accel exceeds target.max_accel",
        )
        require(
            isinstance(stage["modes"], list)
            and bool(stage["modes"])
            and all(
                mode in ("straight", "singer", "dubins", "hybrid")
                for mode in stage["modes"]
            ),
            f"{label}.modes must be nonempty valid modes",
        )
        require(
            stage.get("initial_geometry", "random")
            in ("random", "overtaking", "head_on", "crossing"),
            f"{label} has invalid initial_geometry",
        )
    require(len(names) == len(set(names)), "curriculum names must be unique")
    return config
