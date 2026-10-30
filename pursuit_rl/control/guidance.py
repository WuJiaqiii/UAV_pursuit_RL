"""Nonlearning, approximate geometric guidance baseline (not PPO teacher)."""

from dataclasses import dataclass, asdict
from typing import Protocol
import numpy as np


class Controller(Protocol):
    def act(self, env, obs: np.ndarray) -> np.ndarray:
        """Return normalized CTBR actions with shape (N,4)."""
        ...


@dataclass(frozen=True)
class GuidanceParameters:
    N: float = 3.0
    Na: float = 1.0
    velocity_gain: float = 0.8
    attitude_gain: float = 4.0
    speed_fraction: float = 0.85
    closing_goal_max: float = 8.0
    accel_limit: float = 10.0


def pd_intercept(env, kp=0.20, kd=0.75):
    """Heuristic position/velocity guidance into roll/pitch body rates.

    Only for diagnostic comparisons; does not model full nonlinear dynamics.
    """
    u = env.pursuer
    t = env.target
    cfg = env.cfg["physics"]
    rel = t.pos - u.pos
    vrel = t.vel - u.vel
    desired = np.clip(kp * rel + kd * vrel, -7.0, 7.0)
    # gravity-aligned yaw-neutral local small-angle mapping: a_N≈-g*pitch,
    # a_E≈g*roll; with attitude yaw at reset=0.
    target_roll = np.clip(desired[:, 1] / cfg["gravity"], -0.6, 0.6)
    target_pitch = np.clip(-desired[:, 0] / cfg["gravity"], -0.6, 0.6)
    # Approximate roll and pitch directly from body-to-world rotation.
    rot = u.rot
    current_roll = np.arctan2(rot[:, 2, 1], rot[:, 2, 2])
    current_pitch = np.arcsin(np.clip(rot[:, 0, 2], -1, 1))
    # NOTE current_pitch sign positive with y-body rotation, consistent above.
    desired_rates = np.zeros((env.n, 3))
    desired_rates[:, 0] = np.clip(
        3 * (target_roll - current_roll), -cfg["rate_xy"], cfg["rate_xy"]
    )
    desired_rates[:, 1] = np.clip(
        3 * (target_pitch - current_pitch), -cfg["rate_xy"], cfg["rate_xy"]
    )
    thrust = np.clip(
        cfg["hover_thrust"] - desired[:, 2], cfg["thrust_min"], cfg["thrust_max"]
    )
    acts = np.zeros((env.n, 4))
    acts[:, 0] = (thrust - cfg["hover_thrust"]) / cfg["thrust_delta"]
    acts[:, 1] = desired_rates[:, 0] / cfg["rate_xy"]
    acts[:, 2] = desired_rates[:, 1] / cfg["rate_xy"]
    return np.clip(acts, -1.0, 1.0)


def pn_acceleration(env, N=3.0):
    nav = getattr(env, "navigation", None)
    t, u = (nav.target, nav.pursuer) if nav is not None else (env.target, env.pursuer)
    r = t.pos - u.pos
    rv = t.vel - u.vel
    distance = np.maximum(np.linalg.norm(r, axis=1), 1e-8)
    direction = r / distance[:, None]
    closing = -np.sum(direction * rv, axis=1)
    los_rate = np.cross(r, rv) / distance[:, None] ** 2
    return N * np.maximum(closing, 0.0)[:, None] * np.cross(los_rate, direction)


def apn_acceleration(env, N=3.0, Na=1.0):
    nav = getattr(env, "navigation", None)
    t, u = (nav.target, nav.pursuer) if nav is not None else (env.target, env.pursuer)
    r = t.pos - u.pos
    direction = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-8)
    target_acc = t.acc
    transverse = (
        target_acc - np.sum(target_acc * direction, axis=1, keepdims=True) * direction
    )
    return pn_acceleration(env, N) + Na * transverse


def acceleration_to_action(env, acc, attitude_gain=4.0):
    """NED acceleration -> desired attitude/thrust -> FRD body-rate CTBR.

    Uses nominal mass/drag, not privileged randomized dynamics. Fixed current
    yaw avoids an extra heading objective. No real-aircraft guarantees.
    """
    p, u = env.cfg["physics"], env.pursuer
    acc = np.asarray(acc, dtype=float)
    if acc.shape != (env.n, 3) or not np.isfinite(acc).all():
        raise ValueError("Expected finite (N, 3) desired accelerations")
    nav = getattr(env, "navigation", None)
    guided_velocity = nav.pursuer.vel if nav is not None else u.vel
    drag = p["drag"] * guided_velocity * np.linalg.norm(guided_velocity, axis=1, keepdims=True)
    force = acc + drag - np.array([0.0, 0.0, p["gravity"]])
    # Limit desired tilt below the environment's terminal threshold.
    upward = np.maximum(-force[:, 2], 0.1 * p["gravity"])
    horizontal = np.linalg.norm(force[:, :2], axis=1)
    limit = upward * np.tan(np.deg2rad(0.9 * p["max_tilt_deg"]))
    force[:, :2] *= np.minimum(1.0, limit / np.maximum(horizontal, 1e-8))[:, None]
    force[:, 2] = -upward
    b3 = -force / np.maximum(np.linalg.norm(force, axis=1, keepdims=True), 1e-8)
    yaw = np.arctan2(u.rot[:, 1, 0], u.rot[:, 0, 0])
    heading = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(env.n)], axis=1)
    b2 = np.cross(b3, heading)
    b2 /= np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-8)
    b1 = np.cross(b2, b3)
    desired_rot = np.stack([b1, b2, b3], axis=2)
    error_matrix = 0.5 * (
        np.transpose(desired_rot, (0, 2, 1)) @ u.rot
        - np.transpose(u.rot, (0, 2, 1)) @ desired_rot
    )
    error = np.stack(
        [error_matrix[:, 2, 1], error_matrix[:, 0, 2], error_matrix[:, 1, 0]], axis=1
    )
    rates = -attitude_gain * error
    thrust = np.sum(force * (-u.rot[:, :, 2]), axis=1)
    thrust = np.clip(thrust, p["thrust_min"], p["thrust_max"])
    action = np.zeros((env.n, 4))
    action[:, 0] = (thrust - p["hover_thrust"]) / p["thrust_delta"]
    action[:, 1:] = rates / np.array([p["rate_xy"], p["rate_xy"], p["rate_yaw"]])
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _guided_action(env, augmented=False, N=3.0, Na=1.0, parameters=None):
    parameters = parameters or GuidanceParameters(N=N, Na=Na)
    nav = getattr(env, "navigation", None)
    t, u = (nav.target, nav.pursuer) if nav is not None else (env.target, env.pursuer)
    r = t.pos - u.pos
    direction = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-8)
    # PN alone has no longitudinal speed law. Both baselines use this same law.
    budget = parameters.speed_fraction * env.cfg["physics"]["max_speed"]
    closing_goal = np.clip(
        budget - np.sum(t.vel * direction, axis=1),
        0.0,
        parameters.closing_goal_max,
    )
    desired_velocity = t.vel + closing_goal[:, None] * direction
    desired_velocity *= np.minimum(
        1.0, budget / np.maximum(np.linalg.norm(desired_velocity, axis=1), 1e-8)
    )[:, None]
    speed_acc = parameters.velocity_gain * (desired_velocity - u.vel)
    guidance = (
        apn_acceleration(env, parameters.N, parameters.Na)
        if augmented
        else pn_acceleration(env, parameters.N)
    )
    acc = speed_acc + guidance
    acc *= np.minimum(
        1.0, parameters.accel_limit / np.maximum(np.linalg.norm(acc, axis=1), 1e-8)
    )[:, None]
    return acceleration_to_action(env, acc, parameters.attitude_gain)


def proportional_navigation(env, N=3.0):
    return _guided_action(env, N=N)


def augmented_proportional_navigation(env, N=3.0, Na=1.0):
    return _guided_action(env, augmented=True, N=N, Na=Na)


class GuidanceController:
    def __init__(self, algorithm, parameters=None):
        if algorithm not in ("PN", "APN", "PD"):
            raise ValueError(f"Unknown baseline {algorithm}")
        self.algorithm = algorithm
        self.parameters = parameters or GuidanceParameters()

    def configuration(self):
        return asdict(self.parameters)

    def act(self, env, obs):
        if self.algorithm == "PD":
            return pd_intercept(env)
        return _guided_action(
            env, augmented=self.algorithm == "APN", parameters=self.parameters
        )
