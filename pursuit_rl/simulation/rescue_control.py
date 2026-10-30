"""Nominal self-state velocity/attitude control; no target steering at inference."""

import math
import torch


def limit_norm(x, maximum):
    return x * torch.clamp(
        maximum / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8),
        max=1.0,
    )


def acceleration_to_ctbr(env, acc, profile):
    p, u = env.cfg["physics"], env.pursuer
    drag = p["drag"] * u.vel * torch.linalg.vector_norm(u.vel, dim=1, keepdim=True)
    force = acc + drag
    force = force.clone()
    force[:, 2] -= p["gravity"]
    upward = (-force[:, 2]).clamp_min(0.2 * p["gravity"])
    horizontal = torch.linalg.vector_norm(force[:, :2], dim=1)
    maximum = upward * math.tan(math.radians(profile["tilt_limit_deg"]))
    force[:, :2] *= torch.clamp(maximum / horizontal.clamp_min(1e-8), max=1.0)[:, None]
    force[:, 2] = -upward
    b3 = -force / torch.linalg.vector_norm(force, dim=1, keepdim=True).clamp_min(1e-8)
    yaw = torch.atan2(u.rot[:, 1, 0], u.rot[:, 0, 0])
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), 1)
    b2 = torch.linalg.cross(b3, heading, dim=1)
    b2 = b2 / torch.linalg.vector_norm(b2, dim=1, keepdim=True).clamp_min(1e-8)
    b1 = torch.linalg.cross(b2, b3, dim=1)
    desired = torch.stack((b1, b2, b3), 2)
    error_matrix = 0.5 * (
        desired.transpose(1, 2).bmm(u.rot) - u.rot.transpose(1, 2).bmm(desired)
    )
    error = torch.stack(
        (error_matrix[:, 2, 1], error_matrix[:, 0, 2], error_matrix[:, 1, 0]), 1
    )
    rates = -profile["attitude_gain"] * error
    thrust = (force * (-u.rot[:, :, 2])).sum(1).clamp(p["thrust_min"], p["thrust_max"])
    limits = acc.new_tensor([p["rate_xy"], p["rate_xy"], p["rate_yaw"]])
    return torch.cat(
        (((thrust - p["hover_thrust"]) / p["thrust_delta"])[:, None], rates / limits), 1
    ).clamp(-1, 1)


def action_to_ctbr(env, action, profile):
    if profile["action_mode"] == "ctbr":
        return action
    if profile["action_mode"] == "velocity":
        desired_velocity = action * action.new_tensor(profile["velocity_scale"])
        acc = profile["velocity_gain"] * (desired_velocity - env.pursuer.vel)
    else:
        acc = action * profile["acceleration_limit"]
    return acceleration_to_ctbr(
        env, limit_norm(acc, profile["acceleration_limit"]), profile
    )


def reference_action(env, profile):
    """Training data / explicit baseline only. Never called by the learned action decoder."""
    r = env.target.pos - env.pursuer.pos
    if profile["action_mode"] == "velocity":
        desired = env.target.vel + limit_norm(
            profile["position_teacher_gain"] * r, profile["closing_speed_max"]
        )
        return (desired / r.new_tensor(profile["velocity_scale"])).clamp(-0.99, 0.99)
    if profile["action_mode"] == "acceleration":
        acc = limit_norm(
            0.2 * r + 0.75 * (env.target.vel - env.pursuer.vel),
            profile["acceleration_limit"],
        )
        return (acc / profile["acceleration_limit"]).clamp(-0.99, 0.99)
    # Preserve the verified original PD formula; epsilon only makes atanh finite.
    from pursuit_rl.simulation.pd_teacher import pd_teacher

    return pd_teacher(env).clamp(-0.999999, 0.999999)
