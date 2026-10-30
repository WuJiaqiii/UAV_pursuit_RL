"""Kinematic attitude/body-rate surrogate with first-order closed-loop response.

This is NOT a rotor/motor/inertia simulator. Thrust is mass-normalized to the
nominal vehicle. The SO(3) integration and body-response equations are retained
from the previous Torch backend; the new CTBR mapping uses both endpoints.
"""

from types import SimpleNamespace
import torch


def norm(x, keepdim=False):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=keepdim)


def exp_so3(w):
    angle = norm(w)
    z = angle.square()
    a = torch.where(
        angle < 1e-4, 1 - z / 6 + z * z / 120, torch.sin(angle) / angle.clamp_min(1e-12)
    )
    b = torch.where(
        angle < 1e-4,
        0.5 - z / 24 + z * z / 720,
        (1 - torch.cos(angle)) / z.clamp_min(1e-12),
    )
    x, y, q = w.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero, -q, y, q, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(k)
    return eye + a[:, None, None] * k + b[:, None, None] * k.bmm(k)


class BodyBatch:
    def __init__(self, p, n, device):
        self.cfg = p
        self.n = n
        self.device = device
        self.pursuer = SimpleNamespace()
        self.target = SimpleNamespace()
        for obj in (self.pursuer, self.target):
            for name in ("pos", "vel", "acc"):
                setattr(
                    obj,
                    name,
                    torch.zeros(
                        n,
                        3,
                        device=device,
                        dtype=torch.float64 if obj is self.target else torch.float32,
                    ),
                )
        u = self.pursuer
        u.rot = torch.eye(3, device=device).repeat(n, 1, 1)
        u.rate = torch.zeros(n, 3, device=device)
        u.thrust = torch.full((n,), p["hover_thrust"], device=device)
        for name in ("mass_ratio", "drag", "tau_rate", "tau_thrust"):
            setattr(u, name, torch.ones(n, device=device))
        self.rate_limits = torch.tensor(
            [p["rate_xy"], p["rate_xy"], p["rate_yaw"]], device=device
        )

    @torch.no_grad()
    def step(self, action):
        from pursuit_rl.ctbr.control import thrust_from_action

        u, p = self.pursuer, self.cfg
        h = p["dt"]
        command = thrust_from_action(action[:, 0], p)
        rates = action[:, 1:] * self.rate_limits
        u.rate = u.rate + (rates - u.rate) * (1 - torch.exp(-h / u.tau_rate))[:, None]
        u.thrust = u.thrust + (command - u.thrust) * (1 - torch.exp(-h / u.tau_thrust))
        u.rot = u.rot.bmm(exp_so3(u.rate * h))
        acceleration = -u.rot[:, :, 2] * (u.thrust / u.mass_ratio)[:, None]
        acceleration[:, 2] += p["gravity"]
        acceleration -= u.drag[:, None] * u.vel * norm(u.vel, True)
        u.acc = acceleration
        u.vel = u.vel + acceleration * h
        u.pos = u.pos + u.vel * h
