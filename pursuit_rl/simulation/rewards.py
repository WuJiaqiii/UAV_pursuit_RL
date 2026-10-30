"""Reward components computed once after terminal classification."""

import numpy as np
from pursuit_rl.simulation.geometry import cpa

REWARD_VERSION = "pursuit-reward-v6"


def potential_reward(previous, current, terminated, gamma, weight):
    # Time-limit truncations retain their potential and value bootstrap.
    return weight * (gamma * np.where(terminated, 0.0, current) - previous)


def boundary_risk(pos, physics, soft_distance):
    limit = physics["workspace_xy"]
    low, high = -physics["max_altitude"], -physics["min_altitude"]
    margin = np.minimum.reduce(
        [
            limit - np.abs(pos[:, 0]),
            limit - np.abs(pos[:, 1]),
            pos[:, 2] - low,
            high - pos[:, 2],
        ]
    )
    return np.clip((soft_distance - margin) / soft_distance, 0.0, 1.0) ** 2


def compute_reward(
    env,
    actions,
    old_action,
    relative,
    relative_velocity,
    distance,
    hit,
    failed,
    terminated,
):
    cfg, reward = env.cfg, env.cfg["reward"]
    physics = cfg["physics"]
    dt, gamma = physics["dt"], cfg["training"]["gamma"]
    closing = -np.sum(relative * relative_velocity, axis=1) / np.maximum(distance, 1e-8)
    closing_weight = reward["w_closing"]
    if reward["closing_decay_distance_m"] > 0:
        closing_weight *= np.minimum(1.0, distance / reward["closing_decay_distance_m"])
    cpa_distance, _ = cpa(relative, relative_velocity, reward["cpa_horizon"])
    cpa_potential = -cpa_distance / reward["distance_scale"]
    near_potential = np.exp(-distance / reward["precision_scale_m"])
    near_factor = reward["near_action_delta_factor"] * near_potential
    speed_fraction = np.linalg.norm(env.pursuer.vel, axis=1) / physics["max_speed"]
    tilt = np.arccos(np.clip(env.pursuer.rot[:, 2, 2], -1.0, 1.0))
    tilt_fraction = tilt / np.deg2rad(physics["max_tilt_deg"])
    speed_soft, tilt_soft = reward["soft_speed_frac"], reward["soft_tilt_frac"]
    speed_risk = (
        np.clip((speed_fraction - speed_soft) / (1 - speed_soft), 0.0, 1.0) ** 2
    )
    tilt_risk = np.clip((tilt_fraction - tilt_soft) / (1 - tilt_soft), 0.0, 1.0) ** 2
    future_pos = env.pursuer.pos + env.pursuer.vel * reward["predict_horizon_s"]
    components = {
        "distance": -reward["w_distance"] * dt * distance / reward["distance_scale"],
        "progress": reward["w_progress"] * (env.previous_relative_distance - distance) / reward["progress_scale_m"],
        "closing": closing_weight
        * dt
        * np.clip(closing / reward["speed_scale"], -2.0, 2.0),
        "cpa": (
            potential_reward(
                env.last_cpa_potential,
                cpa_potential,
                terminated,
                gamma,
                reward["w_cpa"],
            )
            if reward["cpa_mode"] == "potential"
            else -reward["w_cpa"] * dt * cpa_distance / reward["distance_scale"]
        ),
        "action_delta": -reward["w_action_delta"]
        * dt
        * (1.0 + near_factor)
        * np.sum((actions - old_action) ** 2, axis=1),
        "time": np.full(env.n, -reward["w_time"] * dt),
        "speed_risk": -dt * reward["w_speed_risk"] * speed_risk,
        "tilt_risk": -dt * reward["w_tilt_risk"] * tilt_risk,
        "boundary_risk": -dt
        * reward["w_boundary_risk"]
        * boundary_risk(env.pursuer.pos, physics, reward["boundary_soft_distance"]),
        "predictive_risk": -dt
        * reward["w_predictive_risk"]
        * boundary_risk(future_pos, physics, reward["boundary_soft_distance"]),
        "precision": potential_reward(
            env.last_precision_potential,
            near_potential,
            terminated,
            gamma,
            reward["w_precision_potential"],
        ),
        "terminal": reward["success_bonus"] * hit
        - reward["failure_penalty"] * (failed & ~hit),
    }
    env.last_cpa_potential[:] = cpa_potential
    env.last_precision_potential[:] = near_potential
    return sum(components.values()), components
