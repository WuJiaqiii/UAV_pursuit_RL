"""Torch control-rate wrapper with true finite-horizon terminals and versioned observations."""

import math
import torch
import torch.nn.functional as F
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.simulation.rescue_control import action_to_ctbr


class RescueBatch:
    def __init__(self, cfg, profile, n=None, seed=None, stage=None, device=None):
        self.cfg = cfg
        self.profile = profile
        self.env = TorchPursuitBatch(
            cfg,
            n=n,
            seed=seed,
            stage=profile["stage"] if stage is None else stage,
            randomize=profile["randomize"],
            device=device or cfg["device"],
        )
        self.n = self.env.n
        self.device = self.env.device
        self.action_dim = 4 if profile["action_mode"] == "ctbr" else 3
        self.memory_start = 32
        self.actor_dim = self.memory_start + self.action_dim
        self.critic_dim = self.actor_dim + 8
        self.interval = profile["decision_interval_steps"]
        self.dt = cfg["physics"]["dt"]
        self.decision_dt = self.interval * self.dt
        self.gamma_tick = math.exp(-self.dt / profile["discount_time_s"])
        self.gamma = math.exp(-self.decision_dt / profile["discount_time_s"])
        self.gae_lambda = (
            math.exp(-self.decision_dt / profile["advantage_time_s"]) / self.gamma
        )
        self.previous_raw = torch.zeros(self.n, self.action_dim, device=self.device)
        self.finished = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.final_obs = None
        self.final_priv = None
        self.episode_return = torch.zeros(self.n, device=self.device)
        self.target_boundary_steps = torch.zeros(self.n, device=self.device)
        self.record_task_metrics = False
        self.last_records = None
        self.active_physics_steps = torch.zeros(
            (), dtype=torch.long, device=self.device
        )

    def observe(self):
        u, t, p = self.env.pursuer, self.env.target, self.cfg["physics"]
        pos = u.pos
        alt = -pos[:, 2]
        margins = torch.stack(
            (
                p["workspace_xy"] - pos[:, 0],
                p["workspace_xy"] + pos[:, 0],
                p["workspace_xy"] - pos[:, 1],
                p["workspace_xy"] + pos[:, 1],
                alt - p["min_altitude"],
                p["max_altitude"] - alt,
            ),
            1,
        )
        scales = pos.new_tensor(
            [p["workspace_xy"]] * 4 + [p["max_altitude"] - p["min_altitude"]] * 2
        )
        obs = torch.cat(
            (
                (t.pos - u.pos) / self.profile["position_scale"],
                (t.vel - u.vel) / self.profile["velocity_obs_scale"],
                t.acc / self.cfg["observation"]["target_accel_scale"],
                u.vel / self.profile["velocity_obs_scale"],
                u.rot.reshape(self.n, 9),
                u.rate / max(p["rate_xy"], p["rate_yaw"]),
                (u.thrust / p["hover_thrust"])[:, None],
                margins / scales,
                (
                    (self.env.stage_config["time"] - self.env.elapsed)
                    / self.env.stage_config["time"]
                )[:, None],
                self.previous_raw,
            ),
            1,
        ).to(torch.float32)
        privileged = torch.cat(
            (
                (u.drag / max(p["drag"], 1e-8))[:, None],
                (u.tau_rate / p["tau_rate"])[:, None],
                (u.tau_thrust / p["tau_thrust"])[:, None],
                u.mass_ratio[:, None],
                F.one_hot(t.mode, 4),
            ),
            1,
        ).to(torch.float32)
        assert obs.shape == (self.n, self.actor_dim)
        critic = torch.cat((obs, privileged), 1)
        if self.final_obs is not None:
            obs = torch.where(self.finished[:, None], self.final_obs, obs)
            critic = torch.where(self.finished[:, None], self.final_priv, critic)
        return obs, critic

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.env.reset(mask)
        self.previous_raw[mask] = 0.0
        self.finished[mask] = False
        self.episode_return[mask] = 0.0
        self.target_boundary_steps[mask] = 0.0
        return self.observe()

    @torch.no_grad()
    def step(self, action, raw=None):
        if (
            action.shape != (self.n, self.action_dim)
            or not torch.isfinite(action).all()
        ):
            raise ValueError("Expected finite rescue policy actions")
        if raw is None:
            raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        if raw.shape != action.shape or not torch.isfinite(raw).all():
            raise ValueError("Invalid raw action memory")
        if not torch.allclose(torch.tanh(raw), action, atol=1e-6, rtol=1e-5):
            raise ValueError("Action and raw memory disagree")
        self.previous_raw = torch.where(
            (~self.finished)[:, None], raw, self.previous_raw
        )
        reward = torch.zeros(self.n, device=self.device)
        term = torch.zeros_like(self.finished)
        records = []
        if self.final_obs is None:
            self.final_obs, self.final_priv = self.observe()
        fields = torch.zeros(
            self.n, 13 if self.record_task_metrics else 8, device=self.device
        )
        # Completed slots may be padded to the common decision boundary; rewards
        # and final observations freeze at the FIRST event, before any reset.
        for tick in range(self.interval):
            active = ~self.finished
            if self.record_task_metrics:
                target = self.env.target
                settings = self.cfg["target"]
                boundary = (
                    target.pos[:, :2].abs() > 0.8 * settings["target_workspace_xy"]
                ).any(1)
                boundary |= (
                    target.pos[:, 2] < -settings["target_max_altitude"] + 20
                ) | (target.pos[:, 2] > -settings["target_min_altitude"] - 20)
                self.target_boundary_steps += (
                    boundary & active & (target.max_accel > 0)
                ).float()
            inner = action_to_ctbr(self.env, action, self.profile)
            _, r, base_term, base_trunc, _ = self.env.step(
                inner, observe=False, finite_horizon=True
            )
            event = self.env.last_event
            done = (base_term | base_trunc) & active
            raw_reward = r - self.profile["timeout_penalty"] * (base_trunc & ~base_term)
            self.episode_return += torch.where(active, raw_reward, 0.0)
            reward += (
                (self.gamma_tick**tick)
                * torch.where(active, raw_reward, 0.0)
                * self.profile["reward_scale"]
            )
            self.active_physics_steps += active.sum()
            obs, priv = self.observe()
            self.final_obs = torch.where(done[:, None], obs, self.final_obs)
            self.final_priv = torch.where(done[:, None], priv, self.final_priv)
            event_fields = torch.stack(
                (
                    event["hit"].float(),
                    (event["speed"] > self.cfg["physics"]["max_speed"]).float(),
                    event["outside"].float(),
                    event["failed"].float(),
                    torch.where(
                        event["hit"],
                        self.env.elapsed - self.dt + event["frac"] * self.dt,
                        self.env.elapsed,
                    ),
                    self.episode_return,
                    self.env.min_distance,
                    done.float(),
                ),
                1,
            )
            if self.record_task_metrics:
                extra = torch.stack(
                    (
                        self.env.target.max_observed_speed,
                        self.env.target.max_observed_acc,
                        event["speed"],
                        event["tilt"] * 180 / math.pi,
                        self.target_boundary_steps,
                    ),
                    1,
                )
                event_fields = torch.cat((event_fields, extra), 1)
            fields = torch.where(done[:, None], event_fields, fields)
            term |= done
            self.finished |= done
        obs, priv = self.observe()
        if bool(term.any()):
            host = fields[term].cpu().tolist()
            slots = term.nonzero(as_tuple=True)[0].cpu().tolist()
            for i, e in zip(slots, host):
                reason = (
                    "success"
                    if e[0]
                    else (
                        "speed"
                        if e[1]
                        else "outside" if e[2] else "tilt" if e[3] else "timeout"
                    )
                )
                records.append(
                    {
                        "slot": i,
                        "success": bool(e[0]),
                        "reason": reason,
                        "time": e[4],
                        "return_unscaled": e[5],
                        "minimum_distance": e[6],
                    }
                )
        if self.record_task_metrics:
            for record, values in zip(records, host if records else []):
                record.update(
                    target_max_speed_observed=values[8],
                    target_max_accel_observed=values[9],
                    pursuer_speed_at_end=values[10],
                    pursuer_tilt_at_end_deg=values[11],
                    target_boundary_active_steps=int(values[12]),
                )
        self.last_records = records
        return (obs, priv), reward, term, torch.zeros_like(term), records
