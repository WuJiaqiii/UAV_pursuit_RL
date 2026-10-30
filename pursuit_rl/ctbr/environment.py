"""Sustained flight and pursuit MDP, fresh CTBR every millisecond.

Only physical failure or the finite deadline ends an episode. Acquisition is a
recorded event, never an early terminal. No target boundary-repulsion force.
"""

import math
import torch
import torch.nn.functional as F
from pursuit_rl.simulation.rigid_body import BodyBatch, exp_so3, norm
from pursuit_rl.simulation.torch_device import resolve_torch_device
from .config import KINDS, validate
from .control import limit

PRIVILEGED_DIM = 44


class CTBRBatch:
    def __init__(self, config, n=None, seed=None):
        self.config = validate(config)
        self.device = resolve_torch_device(config["device"])
        self.n = config["num_envs"] if n is None else n
        if self.n < 1:
            raise ValueError("Empty environment batch")
        self.gen = torch.Generator(device=self.device).manual_seed(
            config["seed"] if seed is None else seed
        )
        self.plant = BodyBatch(config["physics"], self.n, self.device)
        self.dt = config["physics"]["dt"]
        tr = config["training"]
        task = config["task"]
        self.gamma = math.exp(-self.dt / tr["discount_time_s"])
        self.gae_lambda = math.exp(-self.dt / tr["advantage_time_s"]) / self.gamma
        self.actor_dim = 32 + 4 * config["policy"]["history_steps"] + 2
        self.critic_dim = self.actor_dim + PRIVILEGED_DIM
        self.max_steps = round(task["duration"] / self.dt)
        self.tail_steps = round(task["tail_seconds"] / self.dt)
        self.raw_history = torch.zeros(
            self.n, config["policy"]["history_steps"], 4, device=self.device
        )
        self.finished = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.center = torch.zeros(self.n, 3, device=self.device)
        self.command_velocity = torch.zeros(
            self.n, 3, device=self.device, dtype=torch.float64
        )
        self.ou = torch.zeros_like(self.center)
        self.desired_acc = torch.zeros_like(self.center)
        self.pattern = torch.zeros(self.n, dtype=torch.long, device=self.device)
        for name in (
            "phase",
            "timer",
            "speed_setting",
            "circle_radius",
            "episode_return",
            "sum_p2",
            "sum_v2",
            "tail_p2",
            "tail_v2",
            "tail_count",
            "tail_good",
            "maximum_target_speed",
            "maximum_target_accel",
            "maximum_target_jerk",
            "minimum_distance",
            "first_acquisition",
            "longest_dwell",
            "current_dwell",
            "sat_count",
            "ticks",
        ):
            setattr(self, name, torch.zeros(self.n, device=self.device))
        self.initial_rel_pos = torch.zeros_like(self.center)
        self.initial_target_vel = torch.zeros_like(self.center)
        self.final_obs = None
        self.final_priv = None
        self.reset()

    def uniform(self, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(
            shape, device=self.device, generator=self.gen
        )

    def randn(self, *shape):
        return torch.randn(shape, device=self.device, generator=self.gen)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        n = self.n
        p = self.config["physics"]
        c = self.config["task"]
        u, t = self.plant.pursuer, self.plant.target
        self.steps[mask] = 0
        self.finished[mask] = False
        self.raw_history[mask] = 0
        for name in (
            "phase",
            "timer",
            "episode_return",
            "sum_p2",
            "sum_v2",
            "tail_p2",
            "tail_v2",
            "tail_count",
            "tail_good",
            "maximum_target_speed",
            "maximum_target_accel",
            "maximum_target_jerk",
            "longest_dwell",
            "current_dwell",
            "sat_count",
            "ticks",
        ):
            getattr(self, name)[mask] = 0
        self.first_acquisition[mask] = -1.0
        self.minimum_distance[mask] = float("inf")
        self.ou[mask] = 0
        self.desired_acc[mask] = 0
        self.pattern[mask] = torch.randint(
            5, (n,), device=self.device, generator=self.gen
        )[mask]
        center = torch.zeros_like(self.center)
        center[:, :2] = self.uniform(-20, 20, n, 2)
        center[:, 2] = -0.5 * (p["min_altitude"] + p["max_altitude"]) + self.uniform(
            -5, 5, n
        )
        self.center[mask] = center[mask]
        self.phase[mask] = self.uniform(-math.pi, math.pi, n)[mask]
        self.timer[mask] = self.uniform(*c["switch_interval"], n)[mask]
        self.speed_setting[mask] = self.uniform(*c["speed"], n)[mask]
        self.circle_radius[mask] = (
            self.speed_setting.square() / max(c["max_accel"], 0.1)
        ).clamp_min(2.0)[mask]
        t.pos[mask] = self.center[mask].double()
        t.vel[mask] = 0
        t.acc[mask] = 0
        heading = self.uniform(-math.pi, math.pi, n)
        velocity = torch.stack(
            (
                torch.cos(heading) * self.speed_setting,
                torch.sin(heading) * self.speed_setting,
                torch.zeros(n, device=self.device),
            ),
            1,
        )
        if c["kind"] not in ("hover", "goto"):
            t.vel[mask] = velocity[mask].double()
        if c["kind"] == "circle":
            pos, vel, acc = self.circle_state()
            t.pos[mask] = pos[mask].double()
            t.vel[mask] = vel[mask].double()
            t.acc[mask] = acc[mask].double()
        direction = self.randn(n, 3)
        direction[:, 2] *= 0.15
        direction /= norm(direction, True).clamp_min(1e-8)
        distance = self.uniform(*c["distance"], n)
        u.pos[mask] = (t.pos - direction * distance[:, None])[mask].float()
        velocity = self.randn(n, 3) * c["initial_velocity_noise"]
        if c["match_initial_velocity"]:
            velocity += t.vel
        u.vel[mask] = velocity[mask]
        u.acc[mask] = 0
        u.rate[mask] = 0
        angles = self.uniform(-1, 1, n, 3) * math.radians(c["initial_tilt_deg"])
        angles[:, 2] = self.uniform(-math.pi, math.pi, n)
        # yaw is arbitrary; bounded tilt perturbations are composed separately.
        yaw = torch.zeros_like(angles)
        yaw[:, 2] = angles[:, 2]
        angles[:, 2] = 0
        angles = limit(angles, math.radians(c["initial_tilt_deg"]))
        rotations = exp_so3(yaw).bmm(exp_so3(angles))
        u.rot[mask] = rotations[mask]
        u.thrust[mask] = p["hover_thrust"]
        scale = c["randomization"]
        for name, base, low, high in [
            ("mass_ratio", 1.0, 0.95, 1.05),
            ("drag", p["drag"], 0.8, 1.2),
            ("tau_rate", p["tau_rate"], 0.85, 1.15),
            ("tau_thrust", p["tau_thrust"], 0.85, 1.15),
        ]:
            getattr(u, name)[mask] = (
                base * (1 + scale * (self.uniform(low, high, n) - 1))
            )[mask]
        self.command_velocity[mask] = t.vel[mask]
        self.initial_rel_pos[mask] = (t.pos - u.pos)[mask].float()
        self.initial_target_vel[mask] = t.vel[mask].float()
        self.minimum_distance[mask] = norm(t.pos - u.pos)[mask].float()
        return self.observe()

    def circle_state(self):
        r = self.circle_radius
        phase = self.phase
        w = self.speed_setting / r
        offset = torch.stack(
            (r * torch.cos(phase), r * torch.sin(phase), torch.zeros_like(r)), 1
        )
        velocity = torch.stack(
            (-r * w * torch.sin(phase), r * w * torch.cos(phase), torch.zeros_like(r)),
            1,
        )
        accel = torch.stack(
            (
                -r * w * w * torch.cos(phase),
                -r * w * w * torch.sin(phase),
                torch.zeros_like(r),
            ),
            1,
        )
        return self.center + offset, velocity, accel

    def advance_target(self):
        c = self.config["task"]
        t = self.plant.target
        n = self.n
        dt = self.dt
        if c["kind"] in ("hover", "goto"):
            return
        if c["kind"] == "circle":
            self.phase += dt * self.speed_setting / self.circle_radius
            t.pos, t.vel, t.acc = [x.double() for x in self.circle_state()]
            return
        if c["kind"] == "cruise":
            t.pos = t.pos + t.vel * dt
            return
        self.timer -= dt
        changed = self.timer <= 0
        self.timer = torch.where(
            changed, self.uniform(*c["switch_interval"], n), self.timer
        )
        if c["kind"] == "maneuver":
            # Keep the chosen Singer/turn/switchback family for this episode.
            self.pattern = torch.where(
                self.steps == 0,
                1 + torch.randint(3, (n,), device=self.device, generator=self.gen),
                self.pattern,
            )
        else:
            choice = torch.randint(5, (n,), device=self.device, generator=self.gen)
            self.pattern = torch.where(changed, choice, self.pattern)
        fresh = self.randn(n, 3)
        fresh[:, 2] *= 0.15
        fresh = limit(fresh, 1.0) * c["max_accel"]
        self.desired_acc = torch.where(changed[:, None], fresh, self.desired_acc)
        decay = math.exp(-dt / c["ou_time"])
        self.ou = (
            decay * self.ou
            + math.sqrt(1 - decay * decay) * self.randn(n, 3) * c["max_accel"]
        )
        self.ou[:, 2] *= 0.995
        speed = norm(t.vel).clamp_min(0.1)
        lateral = torch.stack((-t.vel[:, 1], t.vel[:, 0], torch.zeros_like(speed)), 1)
        lateral = lateral / norm(lateral, True).clamp_min(1e-8) * c["max_accel"]
        sign = torch.where(self.desired_acc[:, 0] >= 0, 1.0, -1.0)[:, None]
        command = torch.where(
            (self.pattern == 1)[:, None], self.ou, torch.zeros_like(t.vel)
        )
        command = torch.where((self.pattern == 2)[:, None], lateral * sign, command)
        command = torch.where((self.pattern == 3)[:, None], self.desired_acc, command)
        command = torch.where(
            (self.pattern == 4)[:, None],
            -t.vel / speed[:, None] * c["max_accel"],
            command,
        )
        command = limit(command, c["max_accel"])
        # Projection affects a latent velocity reference, never actual velocity.
        # Each projection is nonexpansive and fixes the previous reference, so
        # ||delta reference|| <= A*dt. A convex exponential follower keeps the
        # actual speed within the same convex set. Its acceleration is a convex
        # average of reference derivatives, and ||jerk|| <= 2*A/tau <= J.
        reference = limit(self.command_velocity + command * dt, c["speed"][1])
        reference[:, 2] = reference[:, 2].clamp(-0.75, 0.75)
        tau = max(2 * c["max_accel"] / c["jerk_limit"], dt)
        decay = math.exp(-dt / tau)
        velocity = decay * t.vel + (1 - decay) * reference
        acceleration = (velocity - t.vel) / dt
        self.command_velocity = reference
        t.vel = velocity
        t.pos = t.pos + velocity * dt
        t.acc = acceleration

    def observe(self):
        u, t = self.plant.pursuer, self.plant.target
        p = self.config["physics"]
        c = self.config["task"]
        margins = torch.stack(
            (
                p["workspace_xy"] - u.pos[:, 0],
                p["workspace_xy"] + u.pos[:, 0],
                p["workspace_xy"] - u.pos[:, 1],
                p["workspace_xy"] + u.pos[:, 1],
                -u.pos[:, 2] - p["min_altitude"],
                p["max_altitude"] + u.pos[:, 2],
            ),
            1,
        )
        scales = u.pos.new_tensor(
            [p["workspace_xy"]] * 4 + [p["max_altitude"] - p["min_altitude"]] * 2
        )
        remaining = (self.max_steps - self.steps).float() * self.dt / 60.0
        goals = u.pos.new_tensor(
            [c["radius"] / 5.0, c["velocity_tolerance"] / 5.0]
        ).expand(self.n, 2)
        obs = torch.cat(
            (
                (t.pos - u.pos) / 30.0,
                (t.vel - u.vel) / 15.0,
                t.acc / 7.0,
                u.vel / 15.0,
                u.rot.flatten(1),
                u.rate / max(p["rate_xy"], p["rate_yaw"]),
                (u.thrust / p["hover_thrust"])[:, None],
                margins / scales,
                remaining[:, None],
                self.raw_history.flatten(1),
                goals,
            ),
            1,
        )
        scalars = torch.stack(
            (
                self.speed_setting / 15.0,
                self.circle_radius / 100.0,
                torch.full_like(remaining, c["speed"][1] / 15.0),
                torch.full_like(remaining, c["max_accel"] / 7.0),
                torch.full_like(remaining, c["ou_time"]),
                torch.full_like(remaining, c["duration"] / 60.0),
                torch.full_like(remaining, c["tail_seconds"] / 60.0),
                torch.full_like(remaining, 2 * c["max_accel"] / c["jerk_limit"]),
            ),
            1,
        )
        obs = obs.float()
        count = self.tail_count.clamp_min(1)
        history = torch.stack(
            (
                (self.first_acquisition >= 0).float(),
                self.tail_count / self.tail_steps,
                self.tail_good / count,
                self.tail_v2 / count / 225.0,
                self.tail_p2 / count / 900.0,
            ),
            1,
        )
        kind = F.one_hot(
            torch.full((self.n,), KINDS.index(c["kind"]), device=self.device),
            len(KINDS),
        )
        priv = torch.cat(
            (
                (u.drag / max(p["drag"], 1e-8))[:, None],
                (u.tau_rate / p["tau_rate"])[:, None],
                (u.tau_thrust / p["tau_thrust"])[:, None],
                u.mass_ratio[:, None],
                F.one_hot(self.pattern, 5),
                self.ou / 7.0,
                self.desired_acc / 7.0,
                self.command_velocity / 15.0,
                self.timer[:, None],
                torch.stack((torch.sin(self.phase), torch.cos(self.phase)), 1),
                (self.center - u.pos) / 30.0,
                scalars,
                history,
                kind,
            ),
            1,
        ).float()
        critic = torch.cat((obs, priv), 1)
        assert obs.shape == (self.n, self.actor_dim) and critic.shape == (
            self.n,
            self.critic_dim,
        )
        if self.final_obs is not None:
            obs = torch.where(self.finished[:, None], self.final_obs, obs)
            critic = torch.where(self.finished[:, None], self.final_priv, critic)
        return obs, critic

    @torch.no_grad()
    def step(self, action, raw=None):
        if (
            action.shape != (self.n, 4)
            or not bool(torch.isfinite(action).all())
            or bool((action.abs() > 1.000001).any())
        ):
            raise ValueError("Expected finite normalized CTBR [N,4]")
        if raw is None:
            raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        if (
            raw.shape != action.shape
            or not bool(torch.isfinite(raw).all())
            or not torch.allclose(raw.tanh(), action, atol=1e-6, rtol=1e-5)
        ):
            raise ValueError("Raw action and executed action disagree")
        active = ~self.finished
        if self.final_obs is None:
            self.final_obs, self.final_priv = self.observe()
        p, c, r = self.config["physics"], self.config["task"], self.config["reward"]
        u, t = self.plant.pursuer, self.plant.target
        previous = norm(t.pos - u.pos).float()
        old_acc = t.acc.clone()
        old_action = self.raw_history[:, 0].tanh()
        self.raw_history = torch.cat((raw[:, None], self.raw_history[:, :-1]), 1)
        self.plant.step(action)
        self.advance_target()
        self.steps += 1
        distance = norm(t.pos - u.pos).float()
        v_error = norm(t.vel - u.vel).float()
        tilt = torch.acos(u.rot[:, 2, 2].clamp(-1, 1))
        speed = norm(u.vel)
        outside = (
            (u.pos[:, :2].abs() > p["workspace_xy"]).any(1)
            | (-u.pos[:, 2] < p["min_altitude"])
            | (-u.pos[:, 2] > p["max_altitude"])
        )
        failure = (
            outside
            | (tilt > math.radians(p["max_tilt_deg"]))
            | (speed > p["max_speed"])
        )
        deadline = self.steps >= self.max_steps
        valid_target = (
            (t.pos[:, :2].abs() < p["workspace_xy"] - 10).all(1)
            & (-t.pos[:, 2] > p["min_altitude"] + 5)
            & (-t.pos[:, 2] < p["max_altitude"] - 5)
        )
        # Invalid references are explicit failures, never quietly counted as a
        # controller success; well-configured curriculum bounds avoid this case.
        invalid = ~valid_target
        within = distance <= c["radius"]
        acquired = (self.first_acquisition < 0) & within & active
        self.first_acquisition = torch.where(
            acquired, self.steps.float() * self.dt, self.first_acquisition
        )
        self.current_dwell = torch.where(
            within & active, self.current_dwell + self.dt, torch.zeros_like(distance)
        )
        self.longest_dwell = torch.maximum(self.longest_dwell, self.current_dwell)
        self.minimum_distance = torch.where(
            active,
            torch.minimum(self.minimum_distance, distance),
            self.minimum_distance,
        )
        self.ticks += active.float()
        self.sum_p2 += torch.where(active, distance.square(), 0.0)
        self.sum_v2 += torch.where(active, v_error.square(), 0.0)
        tail = active & (self.steps > self.max_steps - self.tail_steps)
        self.tail_count += tail.float()
        self.tail_good += (tail & within).float()
        self.tail_p2 += torch.where(tail, distance.square(), 0.0)
        self.tail_v2 += torch.where(tail, v_error.square(), 0.0)
        self.sat_count += (active & (action.abs() > 0.98).any(1)).float()
        for name, value in [
            ("maximum_target_speed", norm(t.vel)),
            ("maximum_target_accel", norm(t.acc)),
            ("maximum_target_jerk", norm(t.acc - old_acc) / self.dt),
        ]:
            setattr(
                self,
                name,
                torch.where(
                    active,
                    torch.maximum(getattr(self, name), value),
                    getattr(self, name),
                ),
            )
        count = self.tail_count.clamp_min(1)
        success = (
            deadline
            & ~failure
            & ~invalid
            & (self.tail_count >= self.tail_steps)
            & (self.tail_good / count >= c["minimum_tail_fraction"])
            & (torch.sqrt(self.tail_v2 / count) <= c["velocity_tolerance"])
        )
        done = active & (failure | deadline | invalid)
        components = dict(
            survival=torch.full_like(distance, self.dt * r["survival"]),
            position=self.dt
            * r["position"]
            * torch.exp(-distance / r["position_length"]),
            relative_velocity=self.dt
            * r["relative_velocity"]
            * torch.exp(-v_error / r["velocity_length"]),
            tilt=-self.dt * r["tilt"] * tilt.square(),
            rate=-self.dt
            * r["rate"]
            * (u.rate / self.plant.rate_limits).square().sum(1),
            progress=r["progress"] * (previous - distance),
            action_derivative=-r["action_derivative"]
            * (action - old_action).square().sum(1)
            / self.dt,
            acquisition=r["acquisition_bonus"] * acquired.float(),
            completion=r["completion_bonus"] * success.float(),
            failure=-r["failure"] * (failure | invalid).float(),
        )
        self.reward_components = {
            key: torch.where(active, value, 0.0) * r["scale"]
            for key, value in components.items()
        }
        reward = sum(components.values())
        reward = torch.where(active, reward, 0.0)
        self.episode_return += reward
        obs, priv = self.observe()
        self.final_obs = torch.where(done[:, None], obs, self.final_obs)
        self.final_priv = torch.where(done[:, None], priv, self.final_priv)
        self.finished |= done
        records = []
        if bool(done.any()):
            ids = done.nonzero(as_tuple=True)[0].cpu().tolist()
            fields = (
                torch.stack(
                    (
                        success.float(),
                        outside.float(),
                        (speed > p["max_speed"]).float(),
                        invalid.float(),
                        self.steps.float() * self.dt,
                        self.episode_return,
                        self.minimum_distance,
                        torch.sqrt(self.sum_p2 / self.ticks.clamp_min(1)),
                        torch.sqrt(self.sum_v2 / self.ticks.clamp_min(1)),
                        torch.sqrt(self.tail_p2 / count),
                        torch.sqrt(self.tail_v2 / count),
                        self.tail_good / count,
                        self.first_acquisition,
                        self.longest_dwell,
                        self.maximum_target_speed,
                        self.maximum_target_accel,
                        self.maximum_target_jerk,
                        self.sat_count / self.ticks.clamp_min(1),
                    ),
                    1,
                )[done]
                .cpu()
                .tolist()
            )
            for slot, v in zip(ids, fields):
                reason = (
                    "success"
                    if v[0]
                    else (
                        "invalid_target"
                        if v[3]
                        else (
                            "outside"
                            if v[1]
                            else (
                                "speed"
                                if v[2]
                                else "tilt" if failure[slot] else "tracking_failed"
                            )
                        )
                    )
                )
                records.append(
                    dict(
                        slot=slot,
                        success=bool(v[0]),
                        reason=reason,
                        time=v[4],
                        return_unscaled=v[5],
                        minimum_distance=v[6],
                        position_rmse=v[7],
                        relative_velocity_rmse=v[8],
                        tail_position_rmse=v[9],
                        tail_velocity_rmse=v[10],
                        tail_within_fraction=v[11],
                        first_acquisition_time=v[12] if v[12] >= 0 else None,
                        longest_dwell_s=v[13],
                        target_max_speed=v[14],
                        target_max_accel=v[15],
                        target_max_jerk=v[16],
                        action_saturation_fraction=v[17],
                    )
                )
        obs, priv = self.observe()
        return (obs, priv), reward * r["scale"], done, torch.zeros_like(done), records
