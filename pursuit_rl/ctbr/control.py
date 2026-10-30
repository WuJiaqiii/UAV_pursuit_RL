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


def teacher(env):
    """Nominal geometric PD + currently observed acceleration; no future target state."""
    u, t, p = env.plant.pursuer, env.plant.target, env.config["physics"]
    acc = limit(2.0 * (t.pos - u.pos) + 2.5 * (t.vel - u.vel) + t.acc, 8.0)
    force = acc + p["drag"] * u.vel * torch.linalg.vector_norm(
        u.vel, dim=1, keepdim=True
    )
    force = force.float()
    force[:, 2] -= p["gravity"]
    upward = (-force[:, 2]).clamp_min(0.2 * p["gravity"])
    horizontal = torch.linalg.vector_norm(force[:, :2], dim=1)
    force[:, :2] *= torch.clamp(
        upward * math.tan(math.radians(35.0)) / horizontal.clamp_min(1e-8), max=1
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
    rates = -4.0 * torch.stack((error[:, 2, 1], error[:, 0, 2], error[:, 1, 0]), 1)
    thrust = (force * (-u.rot[:, :, 2])).sum(1).clamp(p["thrust_min"], p["thrust_max"])
    limits = force.new_tensor([p["rate_xy"], p["rate_xy"], p["rate_yaw"]])
    return torch.cat((thrust_to_action(thrust, p)[:, None], rates / limits), 1).clamp(
        -0.999999, 0.999999
    )
