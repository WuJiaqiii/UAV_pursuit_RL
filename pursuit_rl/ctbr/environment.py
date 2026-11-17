"""Batched first safe entry: 1 kHz NED dynamics and 100 Hz direct CTBR actions."""

import math
import torch
import torch.nn.functional as F
from pursuit_rl.simulation.rigid_body import BodyBatch, exp_so3, norm
from pursuit_rl.simulation.torch_device import resolve_torch_device
from .config import KINDS, validate, decision_dt, observation_dims
from .control import limit

PRIVILEGED_DIM = 32


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
        self.decision_steps = config["policy"]["decision_steps"]
        self.decision_dt = decision_dt(config)
        tr = config["training"]
        task = config["task"]
        self.physics_gamma = math.exp(-self.dt / tr["discount_time_s"])
        self.gamma = math.exp(-self.decision_dt / tr["discount_time_s"])
        self.gae_lambda = (
            math.exp(-self.decision_dt / tr["advantage_time_s"]) / self.gamma
        )
        self.base_actor_dim, self.actor_dim = observation_dims(config)
        self.critic_dim = self.actor_dim + PRIVILEGED_DIM
        self.max_steps = round(task["duration"] / self.dt)
        self.max_decision_steps = math.ceil(self.max_steps / self.decision_steps)
        self.raw_history = torch.zeros(
            self.n, config["policy"]["history_steps"], 4, device=self.device
        )
        self.target_history = torch.zeros(
            self.n, config["prediction"]["history_steps"], 10, device=self.device
        )
        self.last_physics_steps = 0
        self.reward_components = {}
        self.reward_components_unscaled = {}
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
            "timer", "speed_setting", "episode_return_scaled", "episode_return_unscaled", "sum_p2", "sum_v2",
            "maximum_target_speed", "maximum_target_accel", "maximum_target_jerk",
            "minimum_distance", "sat_count", "ticks",
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
            "timer", "episode_return_scaled", "episode_return_unscaled", "sum_p2", "sum_v2",
            "maximum_target_speed", "maximum_target_accel", "maximum_target_jerk",
            "sat_count", "ticks",
        ):
            getattr(self, name)[mask] = 0
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
        self.timer[mask] = self.uniform(*c["switch_interval"], n)[mask]
        self.speed_setting[mask] = self.uniform(*c["speed"], n)[mask]
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
        if c["kind"] != "hover":
            t.vel[mask] = velocity[mask].double()
        direction = self.randn(n, 3)
        direction[:, 2] *= 0.15
        direction /= norm(direction, True).clamp_min(1e-8)
        distance = self.uniform(*c["distance"], n)
        encounter = c.get("encounter", "random")
        target_direction = t.vel.float()
        target_direction[:, 2] = 0.0
        speed = norm(target_direction, True)
        target_direction = torch.where(
            speed > 1e-5, target_direction / speed.clamp_min(1e-5), direction
        )
        if encounter == "tail":
            direction = target_direction
        elif encounter == "head_on":
            direction = -target_direction
        u.pos[mask] = (t.pos - direction * distance[:, None])[mask].float()
        velocity = self.randn(n, 3) * c["initial_velocity_noise"]
        init_heading = c["initial_pursuer_heading"]
        if init_heading == "same":
            heading_direction = target_direction
        elif init_heading == "opposite":
            heading_direction = -target_direction
        else:
            angle = self.uniform(-math.pi, math.pi, n)
            heading_direction = torch.stack(
                (torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)), 1
            )
        velocity += heading_direction * self.uniform(
            *c["initial_pursuer_speed"], n
        )[:, None]
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
        if c.get("trimmed_initialization", False):
            # Equilibrium for level-altitude cruise with quadratic drag:
            # F_spec = drag*v*|v| - gravity*NED_vertical.
            cruise_force = u.drag[:, None] * u.vel * norm(u.vel, True)
            cruise_force[:, 2] -= p["gravity"]
            thrust = norm(cruise_force) * u.mass_ratio
            max_load = p.get("max_load_factor_g")
            upper = torch.full_like(thrust, p["thrust_max"])
            if max_load is not None:
                upper = torch.minimum(upper, max_load * p["gravity"] * u.mass_ratio)
            feasible = thrust <= upper + 1e-4
            if not bool(feasible[mask].all()):
                raise ValueError("Trimmed cruise exceeds available load/thrust envelope")
            b3 = -cruise_force / norm(cruise_force, True).clamp_min(1e-8)
            travel = u.vel.clone()
            travel[:, 2] = 0.0
            heading = travel / norm(travel, True).clamp_min(1e-8)
            stationary_heading = torch.stack((
                torch.cos(yaw[:, 2]), torch.sin(yaw[:, 2]), torch.zeros(n, device=self.device)
            ), 1)
            heading = torch.where(
                norm(travel, True) > 1e-5, heading, stationary_heading
            )
            b2 = torch.linalg.cross(b3, heading, dim=1)
            b2 = b2 / norm(b2, True).clamp_min(1e-8)
            b1 = torch.linalg.cross(b2, b3, dim=1)
            trimmed_rot = torch.stack((b1, b2, b3), 2)
            u.rot[mask] = trimmed_rot[mask]
            u.thrust[mask] = thrust[mask]
        self.command_velocity[mask] = t.vel[mask]
        self.initial_rel_pos[mask] = (t.pos - u.pos)[mask].float()
        self.initial_target_vel[mask] = t.vel[mask].float()
        self.minimum_distance[mask] = norm(t.pos - u.pos)[mask].float()
        self.target_history[mask] = 0
        self.append_target_history(mask)
        return self.observe()

    def append_target_history(self, mask):
        t = self.plant.target
        row = torch.cat(
            (
                t.pos.float(),
                t.vel.float(),
                t.acc.float(),
                torch.ones(self.n, 1, device=self.device),
            ),
            1,
        )
        updated = torch.cat((self.target_history[:, 1:], row[:, None]), 1)
        self.target_history = torch.where(
            mask[:, None, None], updated, self.target_history
        )

    def advance_target(self, active=None):
        if active is None:
            active = ~self.finished
        c = self.config["task"]
        t = self.plant.target
        n = self.n
        dt = self.dt
        if c["kind"] == "hover":
            return
        if c["kind"] == "cruise":
            t.pos = t.pos + active[:, None] * t.vel * dt
            return
        self.timer -= dt * active.float()
        changed = (self.timer <= 0) & active
        self.timer = torch.where(
            changed, self.uniform(*c["switch_interval"], n), self.timer
        )
        # Random target families switch among OU, lateral turns and acceleration.
        choice = torch.randint(5, (n,), device=self.device, generator=self.gen)
        self.pattern = torch.where(changed, choice, self.pattern)
        fresh = self.randn(n, 3)
        fresh[:, 2] *= 0.15
        fresh = limit(fresh, 1.0) * c["max_accel"]
        self.desired_acc = torch.where(changed[:, None], fresh, self.desired_acc)
        decay = math.exp(-dt / c["ou_time"])
        new_ou = (
            decay * self.ou
            + math.sqrt(1 - decay * decay) * self.randn(n, 3) * c["max_accel"]
        )
        new_ou[:, 2] *= 0.995
        self.ou = torch.where(active[:, None], new_ou, self.ou)
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
        # Keep randomized high-speed target paths within the specified speed
        # band. The lower boundary is a kinematic reference projection, while
        # actual velocity follows smoothly through the response-time filter.
        minimum = c["speed"][0]
        if minimum > 0:
            magnitude = norm(reference, True).clamp_min(1e-6)
            reference = reference * (minimum / magnitude).clamp_min(1.0)
        reference[:, 2] = reference[:, 2].clamp(-0.75, 0.75)
        tau = c["response_time_s"]
        decay = math.exp(-dt / tau)
        velocity = decay * t.vel + (1 - decay) * reference
        acceleration = (velocity - t.vel) / dt
        self.command_velocity = torch.where(
            active[:, None], reference, self.command_velocity
        )
        t.vel = torch.where(active[:, None], velocity, t.vel)
        t.pos = t.pos + active[:, None] * t.vel * dt
        t.acc = torch.where(active[:, None], acceleration, t.acc)

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
            [c["radius"] / 5.0, c["entry_relative_speed_max_mps"] / 15.0]
        ).expand(self.n, 2)
        obs = torch.cat(
            (
                (t.pos - u.pos) / 100.0,
                (t.vel - u.vel) / 50.0,
                t.acc / 7.0,
                u.vel / 50.0,
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
                self.speed_setting / 50.0,
                torch.full_like(remaining, c["speed"][1] / 50.0),
                torch.full_like(remaining, c["max_accel"] / 7.0),
                torch.full_like(remaining, c["ou_time"]),
                torch.full_like(remaining, c["duration"] / 60.0),
                torch.full_like(remaining, c["response_time_s"]),
                torch.full_like(remaining, c["entry_relative_speed_max_mps"] / 15.0),
            ), 1,
        )
        obs = obs.float()
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
                self.command_velocity.float() / 50.0,
                self.timer[:, None],
                (self.center - u.pos) / 30.0,
                scalars,
                kind,
            ), 1,
        ).float()
        if self.config["prediction"]["enabled"]:
            obs = torch.cat((obs, self.target_history.flatten(1)), 1)
        critic = torch.cat((obs, priv), 1)
        assert obs.shape == (self.n, self.actor_dim) and critic.shape == (
            self.n,
            self.critic_dim,
        )
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
        started = ~self.finished
        old_action = self.raw_history[:, 0].tanh()
        updated = torch.cat((raw[:, None], self.raw_history[:, :-1]), 1)
        self.raw_history = torch.where(
            started[:, None, None], updated, self.raw_history
        )
        derivative_cost = (action - old_action).square().sum(1) / self.decision_dt
        reward = torch.zeros(self.n, device=self.device)
        terminated = torch.zeros_like(self.finished)
        components = {}
        components_unscaled = {}
        records = []
        counts = torch.zeros(self.n, dtype=torch.long, device=self.device)
        for tick in range(self.decision_steps):
            counts += (~self.finished).long()
            step_reward, done, completed = self._physics_step(
                action,
                derivative_cost if tick == 0 else torch.zeros_like(derivative_cost),
            )
            discount = self.physics_gamma**tick
            reward += discount * step_reward
            for key, value in self.reward_components.items():
                components[key] = components.get(key, 0.0) + discount * value
            for key, value in self.reward_components_unscaled.items():
                components_unscaled[key] = (
                    components_unscaled.get(key, 0.0) + discount * value
                )
            terminated |= done
            records.extend(completed)
        self.last_physics_steps = int(counts.sum())
        self.last_physics_steps_per_env = counts
        self.reward_components = components
        self.reward_components_unscaled = components_unscaled
        self.append_target_history(started)
        obs, priv = self.observe()
        self.final_obs, self.final_priv = obs, priv
        return (obs, priv), reward, terminated, torch.zeros_like(terminated), records

    def _physics_step(self, action, derivative_cost):
        active = ~self.finished
        p, c, r = self.config["physics"], self.config["task"], self.config["reward"]
        u, t = self.plant.pursuer, self.plant.target
        old_relative = (t.pos - u.pos).float().clone()
        previous = norm(old_relative).float()
        old_acc = t.acc.clone()
        self.plant.step(action, active)
        self.advance_target(active)
        self.steps += active.long()
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
        # First crossing of the 3D sphere; physical/target faults take priority.
        new_relative = (t.pos - u.pos).float()
        relative_delta = new_relative - old_relative
        alpha = (
            -(old_relative * relative_delta).sum(1)
            / relative_delta.square().sum(1).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        segment_nearest = norm(old_relative + alpha[:, None] * relative_delta)
        entered = (distance <= c["radius"]) | (segment_nearest <= c["radius"])
        speed_ok = v_error <= c["entry_relative_speed_max_mps"]
        success = active & entered & speed_ok & ~failure & ~invalid
        unsafe_entry = active & entered & ~speed_ok & ~failure & ~invalid
        done = active & (failure | invalid | deadline | success | unsafe_entry)
        penalize = failure | invalid | unsafe_entry | (deadline & ~success)
        self.minimum_distance = torch.where(
            active,
            torch.minimum(self.minimum_distance, segment_nearest),
            self.minimum_distance,
        )
        self.ticks += active.float()
        self.sum_p2 += torch.where(active, distance.square(), 0.0)
        self.sum_v2 += torch.where(active, v_error.square(), 0.0)
        self.sat_count += (active & (action.abs() > 0.98).any(1)).float()
        for name, value in (
            ("maximum_target_speed", norm(t.vel)),
            ("maximum_target_accel", norm(t.acc)),
            ("maximum_target_jerk", norm(t.acc - old_acc) / self.dt),
        ):
            setattr(
                self, name,
                torch.where(
                    active,
                    torch.maximum(getattr(self, name), value),
                    getattr(self, name),
                ),
            )
        length = c["entry_velocity_distance_m"]
        progress_reward = r["progress"] * (
            previous / (previous + length) - distance / (distance + length)
        )
        closing_speed = (
            -((t.vel - u.vel).float() * new_relative).sum(1)
            / distance.clamp_min(1e-6)
        ).clamp_min(0.0)
        allowable_closing = torch.sqrt(
            2.0 * c["entry_brake_accel_mps2"]
            * (distance - c["radius"]).clamp_min(0.0)
        )
        scale_v = c["entry_velocity_scale_mps"]
        braking = ((closing_speed - allowable_closing).clamp_min(0.0) / scale_v).square()
        velocity_cost = (v_error / scale_v).square() * torch.exp(-distance / length)
        components = {
            "tilt": -self.dt * r["tilt"] * tilt.square(),
            "rate": -self.dt * r["rate"]
                    * (u.rate / self.plant.rate_limits).square().sum(1),
            "progress": progress_reward,
            "action_derivative": -r["action_derivative"] * derivative_cost,
            "braking": -self.dt * r["braking"] * braking,
            "entry_velocity": -self.dt * r["entry_velocity"] * velocity_cost,
            "time": -self.dt * r["time"] * torch.ones_like(distance),
            "completion": r["completion_bonus"] * success.float(),
            "failure": -r["failure"] * penalize.float(),
        }
        # Keep raw and scaled terms explicit. PPO, rollout diagnostics and the
        # decision-step component sum all use the same scaled reward tensor.
        self.reward_components_unscaled = {
            key: torch.where(active, value, 0.0)
            for key, value in components.items()
        }
        self.reward_components = {
            key: value * r["scale"]
            for key, value in self.reward_components_unscaled.items()
        }
        reward_unscaled = sum(self.reward_components_unscaled.values())
        reward = sum(self.reward_components.values())
        self.episode_return_unscaled += reward_unscaled
        self.episode_return_scaled += reward
        self.finished |= done
        records = []
        if bool(done.any()):
            ids = done.nonzero(as_tuple=True)[0].cpu().tolist()
            for slot in ids:
                if invalid[slot]:
                    reason = "invalid_target"
                elif outside[slot]:
                    reason = "outside"
                elif speed[slot] > p["max_speed"]:
                    reason = "speed"
                elif failure[slot]:
                    reason = "tilt"
                elif unsafe_entry[slot]:
                    reason = "unsafe_entry"
                elif success[slot]:
                    reason = "success"
                else:
                    reason = "timeout"
                crossed = bool(success[slot] or unsafe_entry[slot])
                records.append(
                    dict(
                        slot=slot,
                        task=c["name"],
                        success=bool(success[slot]),
                        reason=reason,
                        time=float(self.steps[slot]) * self.dt,
                        return_unscaled=float(self.episode_return_unscaled[slot]),
                        return_scaled=float(self.episode_return_scaled[slot]),
                        minimum_distance=float(self.minimum_distance[slot]),
                        position_rmse=float(
                            torch.sqrt(self.sum_p2[slot] / self.ticks[slot].clamp_min(1))
                        ),
                        relative_velocity_rmse=float(
                            torch.sqrt(self.sum_v2[slot] / self.ticks[slot].clamp_min(1))
                        ),
                        entry_time_s=float(self.steps[slot]) * self.dt if success[slot] else None,
                        entry_relative_speed_mps=float(v_error[slot]) if crossed else None,
                        unsafe_entry=bool(unsafe_entry[slot]),
                        target_max_speed=float(self.maximum_target_speed[slot]),
                        target_max_accel=float(self.maximum_target_accel[slot]),
                        target_max_jerk=float(self.maximum_target_jerk[slot]),
                        action_saturation_fraction=float(
                            self.sat_count[slot] / self.ticks[slot].clamp_min(1)
                        ),
                    )
                )
        return reward, done, records
