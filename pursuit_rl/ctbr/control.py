"""Matched CTBR units; geometric teacher is never mixed into learned actions."""

import math
import torch


def limit(x, maximum):
    return x * torch.clamp(
        torch.as_tensor(maximum, device=x.device)
        / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8),
        max=1.0,
    )


def thrust_from_action(action, p):
    return p["hover_thrust"] + action * torch.where(
        action >= 0,
        action.new_tensor(p["thrust_max"] - p["hover_thrust"]),
        action.new_tensor(p["hover_thrust"] - p["thrust_min"]),
    )


def thrust_to_action(thrust, p):
    delta = thrust - p["hover_thrust"]
    return (
        delta
        / torch.where(
            delta >= 0,
            delta.new_tensor(p["thrust_max"] - p["hover_thrust"]),
            delta.new_tensor(p["hover_thrust"] - p["thrust_min"]),
        )
    ).clamp(-1, 1)


def guidance_acceleration(env):
    """PN, PD, or PN-to-PD hybrid acceleration guidance (NED, m/s^2).

    This teacher produces demonstrations only; PPO still executes its own
    four-dimensional direct CTBR command without teacher blending.
    """
    u, t = env.plant.pursuer, env.plant.target
    tr = env.config["training"]
    r = (t.pos - u.pos).float()
    v = (t.vel - u.vel).float()
    a_target = t.acc.float()
    distance = torch.linalg.vector_norm(r, dim=1, keepdim=True).clamp_min(1e-5)
    los = r / distance
    closing = -(v * los).sum(1, keepdim=True)
    maximum = tr.get("teacher_accel_limit", 8.0)
    pd = limit(2.0 * r + 2.5 * v + a_target, maximum)
    method = tr.get("teacher_method", "pd")
    if method == "pd":
        return pd
    # LOS angular velocity and classical proportional navigation (lateral).
    omega = torch.linalg.cross(r, v, dim=1) / distance.square()
    pn_lateral = (
        tr.get("teacher_pn_gain", 3.0)
        * closing.clamp_min(0)
        * torch.linalg.cross(omega, los, dim=1)
    )
    # Longitudinal guidance targets a finite closing speed that can be
    # arrested before entering the first-entry target sphere.
    task = env.config["task"]
    braking = task.get("entry_brake_accel_mps2", 8.0)
    desired_closing = torch.sqrt(
        2.0 * braking * (distance - task["radius"]).clamp_min(0.0)
    )
    along = (desired_closing - closing) * los
    pn = limit(pn_lateral + 2.0 * along + a_target, maximum)
    if method == "pn":
        return pn
    switch = tr.get("teacher_switch_distance_m", 8.0)
    blend = ((distance - switch) / max(switch, 1.0)).clamp(0.0, 1.0)
    return limit(blend * pn + (1.0 - blend) * pd, maximum)



def observed_pd(env, smooth=False):
    """PD reference from Actor-visible ideal state and nominal physics only.

    Excludes realized random drag/mass, target command latents and Critic state.
    A filtered variant shares the Actor's raw-action autoregressive smoothing.
    """
    obs, _ = env.observe()
    p, tr = env.config["physics"], env.config["training"]
    pc = env.config["policy"]
    r = obs[:, :3]
    if pc["relative_position_mode"] == "asinh":
        r = torch.sinh(r) * pc["relative_position_scale_m"]
    else:
        r = r * pc["relative_position_scale_m"]
    v = obs[:, 3:6] * pc["relative_velocity_scale_mps"]
    a_target, velocity = obs[:, 6:9] * 7.0, obs[:, 9:12] * 50.0
    rot = obs[:, 12:21].reshape(-1, 3, 3)
    acc = limit(2.0 * r + 2.5 * v + a_target,
                tr.get("teacher_accel_limit", 8.0))
    force = acc + p["drag"] * velocity * torch.linalg.vector_norm(
        velocity, dim=1, keepdim=True
    )
    force[:, 2] -= p["gravity"]
    upward = (-force[:, 2]).clamp_min(0.2 * p["gravity"])
    horizontal = torch.linalg.vector_norm(force[:, :2], dim=1)
    max_tilt = tr.get("teacher_tilt_limit_deg", 35.0)
    force[:, :2] *= torch.clamp(
        upward * math.tan(math.radians(max_tilt))
        / horizontal.clamp_min(1e-8), max=1.0
    )[:, None]
    force[:, 2] = -upward
    b3 = -force / torch.linalg.vector_norm(force, dim=1, keepdim=True).clamp_min(1e-8)
    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), 1)
    b2 = torch.linalg.cross(b3, heading, dim=1)
    b2 = b2 / torch.linalg.vector_norm(b2, dim=1, keepdim=True).clamp_min(1e-8)
    desired = torch.stack((torch.linalg.cross(b2, b3, dim=1), b2, b3), 2)
    error = 0.5 * (
        desired.transpose(1, 2).bmm(rot) - rot.transpose(1, 2).bmm(desired)
    )
    rates = -4.0 * torch.stack((error[:, 2, 1], error[:, 0, 2], error[:, 1, 0]), 1)
    thrust = (force * (-rot[:, :, 2])).sum(1).clamp(
        p["thrust_min"], p["thrust_max"]
    )
    if p.get("max_load_factor_g") is not None:
        thrust = thrust.clamp(max=p["gravity"] * p["max_load_factor_g"])
    limits = force.new_tensor([p["rate_xy"], p["rate_xy"], p["rate_yaw"]])
    action = torch.cat((thrust_to_action(thrust, p)[:, None], rates / limits), 1)
    action = action.clamp(-0.999999, 0.999999)
    if smooth:
        rho = math.exp(-env.decision_dt / env.config["policy"]["exploration_time_s"])
        previous_raw = obs[:, 32:36]
        action = torch.tanh(rho * previous_raw + (1.0 - rho) * torch.atanh(action))
    return action


def teacher(env):
    """Convert geometric acceleration guidance to bounded CTBR setpoints."""
    method = env.config["training"]["teacher_method"]
    if method in ("pd_observed", "pd_observed_smoothed"):
        return observed_pd(env, smooth=method == "pd_observed_smoothed")
    u, p = env.plant.pursuer, env.config["physics"]
    acc = guidance_acceleration(env)
    # Desired specific force compensates the modeled quadratic drag.
    force = acc + u.drag[:, None] * u.vel * torch.linalg.vector_norm(
        u.vel, dim=1, keepdim=True
    )
    force = force.float()
    force[:, 2] -= p["gravity"]
    upward = (-force[:, 2]).clamp_min(0.2 * p["gravity"])
    horizontal = torch.linalg.vector_norm(force[:, :2], dim=1)
    max_tilt = env.config["training"].get("teacher_tilt_limit_deg", 35.0)
    force[:, :2] *= torch.clamp(
        upward * math.tan(math.radians(max_tilt))
        / horizontal.clamp_min(1e-8),
        max=1.0,
    )[:, None]
    force[:, 2] = -upward
    b3 = -force / torch.linalg.vector_norm(force, dim=1, keepdim=True).clamp_min(1e-8)
    yaw = torch.atan2(u.rot[:, 1, 0], u.rot[:, 0, 0])
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), 1)
    b2 = torch.linalg.cross(b3, heading, dim=1)
    b2 /= torch.linalg.vector_norm(b2, dim=1, keepdim=True).clamp_min(1e-8)
    desired = torch.stack((torch.linalg.cross(b2, b3, dim=1), b2, b3), 2)
    error = 0.5 * (
        desired.transpose(1, 2).bmm(u.rot) - u.rot.transpose(1, 2).bmm(desired)
    )
    rates = -4.0 * torch.stack(
        (error[:, 2, 1], error[:, 0, 2], error[:, 1, 0]), 1
    )
    thrust = (force * (-u.rot[:, :, 2])).sum(1).clamp(
        p["thrust_min"], p["thrust_max"]
    )
    if p.get("max_load_factor_g") is not None:
        thrust = torch.minimum(
            thrust, u.mass_ratio * p["gravity"] * p["max_load_factor_g"]
        )
    limits = force.new_tensor([p["rate_xy"], p["rate_xy"], p["rate_yaw"]])
    return torch.cat((thrust_to_action(thrust, p)[:, None], rates / limits), 1).clamp(
        -0.999999, 0.999999
    )
