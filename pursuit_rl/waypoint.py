"""Single-vehicle CTBR simulation with a stationary virtual waypoint."""

from enum import IntEnum
import math

import torch

from pursuit_rl.ctbr.config import DEFAULT
from pursuit_rl.ctbr.control import thrust_to_action
from pursuit_rl.simulation.rigid_body import BodyBatch, norm
from pursuit_rl.simulation.torch_device import resolve_torch_device
from pursuit_rl.waypoint_plan import WaypointConfig as WaypointConfig


SCHEMA = "virtual-waypoint-ctbr-v1"
OBS_DIM = 20


class Outcome(IntEnum):
    ACTIVE = 0
    GOAL_REACHED = 1
    OVERSPEED = 2
    OUTSIDE = 3
    TILT = 4
    NONFINITE = 5
    TIMEOUT = 7


class WaypointEnv:
    """Vectorized, finite-horizon episodes with one body and no target dynamics."""

    def __init__(self, config, n, seed, device="cpu"):
        if n < 1:
            raise ValueError("At least one environment is required")
        self.config = config
        self.n = n
        self.device = resolve_torch_device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.physics = {**DEFAULT["physics"], "workspace_xy": 30.0,
                        "min_altitude": 3.0, "max_altitude": 18.0,
                        "max_tilt_deg": 45.0, "max_speed": config.max_speed_mps}
        self.dt = self.physics["dt"]
        self.deadline = math.ceil(config.timeout_s / self.dt)
        self.plant = BodyBatch(self.physics, n, self.device)
        self.body = self.plant.pursuer
        self.waypoint = torch.zeros_like(self.body.pos)
        self.ticks = torch.zeros(n, device=self.device, dtype=torch.long)
        self.outcome = torch.zeros_like(self.ticks)
        self.minimum_distance = torch.zeros(n, device=self.device)
        self.peak_speed = torch.zeros_like(self.minimum_distance)
        self.progress_scale = torch.ones_like(self.minimum_distance)
        self.reset()

    def reset(self, mask=None):
        slots = torch.arange(self.n, device=self.device) if mask is None else mask.nonzero().flatten()
        if not len(slots):
            return self.observe()
        sample = torch.rand((len(slots), 3), generator=self.rng, device=self.device)
        low, high = self.config.horizontal_distance_m
        distance = low + (high - low) * sample[:, 0]
        yaw = sample[:, 1] * (2 * math.pi)
        u, p = self.body, self.physics
        u.pos[slots] = u.pos.new_tensor([0.0, 0.0, -10.0])
        offsets = torch.stack((distance * yaw.cos(), distance * yaw.sin(), sample[:, 2] - 0.5), 1)
        self.waypoint[slots] = u.pos[slots] + offsets
        for name in ("vel", "acc", "rate"):
            getattr(u, name)[slots] = 0
        u.rot[slots] = torch.eye(3, device=self.device)
        u.thrust[slots] = p["hover_thrust"]
        u.mass_ratio[slots] = 1.0
        for name in ("drag", "tau_rate", "tau_thrust"):
            getattr(u, name)[slots] = p[name]
        self.ticks[slots] = 0
        self.outcome[slots] = Outcome.ACTIVE
        self.minimum_distance[slots] = norm(offsets)
        self.peak_speed[slots] = 0
        self.progress_scale[slots] = norm(offsets).clamp_min(1.0)
        return self.observe()

    def observe(self):
        u = self.body
        observation = torch.cat((
            (self.waypoint - u.pos) / 5.0,
            u.vel / self.config.max_speed_mps,
            u.rot.flatten(1),
            u.rate / self.plant.rate_limits,
            thrust_to_action(u.thrust, self.physics)[:, None],
            (1.0 - self.ticks / self.deadline)[:, None],
        ), dim=1)
        return torch.where((self.outcome != Outcome.NONFINITE)[:, None], observation, 0.0)

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.n, 4):
            raise ValueError("CTBR actions must have shape (num_envs, 4)")
        action = action.clamp(-1.0, 1.0)
        u, c, p = self.body, self.config, self.physics
        waiting = self.outcome == Outcome.ACTIVE
        previous_ticks = self.ticks.clone()
        previous_distance = norm(self.waypoint - u.pos)
        cos_tilt_limit = math.cos(math.radians(p["max_tilt_deg"]))
        for _ in range(c.decision_ticks):
            active = self.outcome == Outcome.ACTIVE
            before = u.pos
            self.plant.step(action, active=active)
            self.ticks += active.long()

            delta = u.pos - before
            fraction = (((self.waypoint - before) * delta).sum(1)
                        / delta.square().sum(1).clamp_min(1e-12)).clamp(0.0, 1.0)
            nearest = norm(self.waypoint - (before + delta * fraction[:, None]))
            speed = norm(u.vel)
            self.minimum_distance = torch.where(
                active & torch.isfinite(nearest), torch.minimum(nearest, self.minimum_distance),
                self.minimum_distance,
            )
            self.peak_speed = torch.where(
                active & torch.isfinite(speed), torch.maximum(speed, self.peak_speed), self.peak_speed,
            )
            outside = (u.pos[:, :2].abs() > p["workspace_xy"]).any(1)
            outside |= (-u.pos[:, 2] < p["min_altitude"]) | (-u.pos[:, 2] > p["max_altitude"])
            nonfinite = ~(torch.isfinite(nearest) & torch.isfinite(speed)
                          & torch.isfinite(u.rot).all(dim=(1, 2)))
            outcome = torch.zeros_like(self.outcome)
            # Later matches take priority, including a fault on the arrival tick.
            for condition, code in (
                (self.ticks >= self.deadline, Outcome.TIMEOUT),
                (nearest <= c.radius_m, Outcome.GOAL_REACHED),
                (u.rot[:, 2, 2] < cos_tilt_limit, Outcome.TILT),
                (outside, Outcome.OUTSIDE),
                (speed > c.max_speed_mps, Outcome.OVERSPEED),
                (nonfinite, Outcome.NONFINITE),
            ):
                outcome = torch.where(condition, int(code), outcome)
            self.outcome = torch.where(active, outcome, self.outcome)

        terminal = waiting & (self.outcome != Outcome.ACTIVE)
        reached = terminal & (self.outcome == Outcome.GOAL_REACHED)
        elapsed = (self.ticks - previous_ticks) * self.dt
        delta = (previous_distance - norm(self.waypoint - u.pos)) / self.progress_scale
        delta = torch.where(waiting & torch.isfinite(delta), delta, 0.0)
        reward = delta - 0.02 * elapsed + 2.0 * reached.float() - (terminal & ~reached).float()
        return self.observe(), reward, terminal, elapsed


@torch.no_grad()
def waypoint_pd(env):
    """Position/velocity PD for a stationary virtual waypoint in the simulator."""
    u, p = env.body, env.physics
    velocity = 1.6 * (env.waypoint - u.pos)
    velocity *= (env.config.reference_speed_mps / norm(velocity, True).clamp_min(1e-6)).clamp(max=1)
    force = 2.8 * (velocity - u.vel) + p["drag"] * u.vel * norm(u.vel, True)
    force[:, 2] -= p["gravity"]
    upward = (-force[:, 2]).clamp(0.5 * p["gravity"], 1.5 * p["gravity"])
    horizontal = force[:, :2]
    tilt_bound = upward[:, None] * math.tan(math.radians(30.0))
    horizontal *= (tilt_bound / norm(horizontal, True).clamp_min(1e-6)).clamp(max=1)
    force[:, 2] = -upward

    b3 = -force / norm(force, True)
    forward = torch.nn.functional.normalize(
        torch.cat((u.rot[:, :2, 0], torch.zeros_like(b3[:, :1])), dim=1), dim=1,
    )
    b2 = torch.nn.functional.normalize(torch.linalg.cross(b3, forward, dim=1), dim=1)
    desired = torch.stack((torch.linalg.cross(b2, b3, dim=1), b2, b3), dim=2)
    error = 0.5 * (desired.transpose(1, 2).bmm(u.rot) - u.rot.transpose(1, 2).bmm(desired))
    rates = -4.0 * torch.stack((error[:, 2, 1], error[:, 0, 2], error[:, 1, 0]), dim=1)
    thrust = (force * -u.rot[:, :, 2]).sum(1)
    return torch.cat((thrust_to_action(thrust, p)[:, None], rates / env.plant.rate_limits), dim=1).clamp(-1, 1)
