"""64-way vectorized pursuit MDP with asymmetric actor/critic observations.

The target is a kinematic random maneuverer, not a real aircraft. All
states are computed from the same discrete simulation clock. Environments
are reset explicitly so the trainer can bootstrap truncated rollouts.
"""

import numpy as np
from pursuit_rl.simulation.dynamics import QuadrotorBatch
from pursuit_rl.simulation.targets import TargetBatch
from pursuit_rl.simulation.geometry import cpa
from pursuit_rl.simulation.observations import (
    ACTOR_DIM,
    CRITIC_DIM,
    observation_dimensions,
    encode_observations,
)
from pursuit_rl.simulation.rewards import compute_reward
from pursuit_rl.simulation.events import terminal_events, episode_records


class PursuitBatch:
    def __init__(self, cfg, n=None, seed=None, stage=None, randomize=True):
        self.cfg = cfg
        self.n = int(cfg["training"]["num_envs"] if n is None else n)
        if self.n <= 0:
            raise ValueError("n must be positive")
        self.rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
        self.pursuer = QuadrotorBatch(self.n, cfg["physics"], self.rng)
        self.target = TargetBatch(self.n, cfg["target"], self.rng)
        self.stage = int(cfg["training"]["initial_stage"] if stage is None else stage)
        if not 0 <= self.stage < len(cfg["curriculum"]):
            raise ValueError("Invalid stage")
        self.randomize = randomize
        self.elapsed = np.zeros(self.n)
        self.steps = np.zeros(self.n, dtype=int)
        self.initial_distance = np.zeros(self.n)
        self.min_distance = np.full(self.n, np.inf)
        self.last_cpa_potential = np.zeros(self.n)
        self.last_precision_potential = np.zeros(self.n)
        self.initial_closing = np.zeros(self.n)
        self.initial_cpa = np.zeros(self.n)
        self.initial_rel_pos = np.zeros((self.n, 3))
        self.initial_target_vel = np.zeros((self.n, 3))
        self.episode_return = np.zeros(self.n)
        self.lateral_accel_cost_full = np.zeros(self.n)
        self.control_variation_full = np.zeros(self.n)
        self.high_envelope_count_full = np.zeros(self.n, dtype=np.int64)
        self.reward_components = {}
        self.reset()

    @property
    def stage_config(self):
        return self.cfg["curriculum"][self.stage]

    def set_stage(self, stage):
        if not 0 <= int(stage) < len(self.cfg["curriculum"]):
            raise ValueError("Invalid stage")
        self.stage = int(stage)
        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n,):
            raise ValueError("reset mask must have shape (N,)")
        ids = np.flatnonzero(mask)
        if not len(ids):
            return self.observe()
        self.target.reset(mask, self.stage_config)
        k = len(ids)
        heading = self.rng.uniform(-np.pi, np.pi, size=k)
        elevation = self.rng.uniform(-0.18, 0.18, size=k)
        direction = np.stack(
            [
                np.cos(heading) * np.cos(elevation),
                np.sin(heading) * np.cos(elevation),
                np.sin(elevation),
            ],
            axis=1,
        )
        distance = self.rng.uniform(*self.stage_config["distance"], size=k)
        initial = self.target.pos[ids] - direction * distance[:, None]
        # Preserve the sampling protocol; only the initial altitude is clipped.
        initial[:, 2] = np.clip(
            initial[:, 2],
            -self.cfg["physics"]["max_altitude"] + 20,
            -self.cfg["physics"]["min_altitude"] - 20,
        )
        geometry = self.stage_config.get("initial_geometry", "random")
        if geometry != "random":
            if geometry == "overtaking":
                planar = direction[:, :2]
            elif geometry == "head_on":
                planar = -direction[:, :2]
            else:
                planar = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
            planar = planar / np.maximum(
                np.linalg.norm(planar, axis=1, keepdims=True), 1e-9
            )
            speed = self.target.speed_setting[ids]
            self.target.vel[ids, :2] = planar * speed[:, None]
            self.target.vel[ids, 2] = 0.0
            self.target.heading[ids] = np.arctan2(planar[:, 1], planar[:, 0])
        self.pursuer.reset(mask, initial_positions=initial, randomize=self.randomize)
        self.elapsed[ids] = 0.0
        self.steps[ids] = 0
        self.episode_return[ids] = 0.0
        self.lateral_accel_cost_full[ids] = 0.0
        self.control_variation_full[ids] = 0.0
        self.high_envelope_count_full[ids] = 0
        self.initial_distance[ids] = np.linalg.norm(
            self.target.pos[ids] - self.pursuer.pos[ids], axis=1
        )
        self.min_distance[ids] = self.initial_distance[ids]
        rel = self.target.pos[ids] - self.pursuer.pos[ids]
        relvel = self.target.vel[ids] - self.pursuer.vel[ids]
        dc, _ = cpa(rel, relvel, self.cfg["reward"]["cpa_horizon"])
        self.last_cpa_potential[ids] = -dc / self.cfg["reward"]["distance_scale"]
        self.last_precision_potential[ids] = np.exp(
            -np.linalg.norm(rel, axis=1) / self.cfg["reward"]["precision_scale_m"]
        )
        self.initial_closing[ids] = -np.sum(rel * relvel, axis=1) / np.maximum(
            np.linalg.norm(rel, axis=1), 1e-8
        )
        self.initial_cpa[ids] = dc
        self.initial_rel_pos[ids] = rel
        self.initial_target_vel[ids] = self.target.vel[ids]
        return self.observe()

    def observe(self):
        return encode_observations(self)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n, 4) or not np.isfinite(actions).all():
            raise ValueError("Expected finite actions with shape (N, 4)")
        actions = np.clip(actions, -1.0, 1.0)
        dt = self.cfg["physics"]["dt"]
        previous_relative = self.target.pos - self.pursuer.pos
        self.previous_relative_distance = np.linalg.norm(previous_relative, axis=1)
        self.previous_pursuer_pos = self.pursuer.pos.copy()
        self.previous_target_pos = self.target.pos.copy()
        previous_vel = self.pursuer.vel.copy()
        old_action = self.pursuer.last_action.copy()
        self.pursuer.step(actions, dt)
        self.target.step(dt)
        self.elapsed += dt
        self.steps += 1
        relative = self.target.pos - self.pursuer.pos
        relative_velocity = self.target.vel - self.pursuer.vel
        distance = np.linalg.norm(relative, axis=1)
        self.min_distance = np.minimum(self.min_distance, distance)
        # Full-rate accumulators avoid aliasing in control/maneuver metrics when
        # NPZ state snapshots are saved at a lower rate than policy decisions.
        accel = self.pursuer.acceleration
        vhat = previous_vel / np.maximum(np.linalg.norm(previous_vel, axis=1, keepdims=True), 1e-8)
        alat = accel - np.sum(accel * vhat, axis=1, keepdims=True) * vhat
        self.lateral_accel_cost_full += np.sum(alat * alat, axis=1) * dt
        self.control_variation_full += np.linalg.norm(actions - old_action, axis=1) * (self.steps > 1)
        phy = self.cfg["physics"]
        tilt_util = np.arccos(np.clip(self.pursuer.rot[:, 2, 2], -1., 1.)) / np.deg2rad(phy["max_tilt_deg"])
        speed_util = np.linalg.norm(self.pursuer.vel, axis=1) / phy["max_speed"]
        load_util = self.pursuer.load_factor() / (phy["thrust_max"] / phy["gravity"])
        rates = np.abs(self.pursuer.rate) / [phy["rate_xy"], phy["rate_xy"], phy["rate_yaw"]]
        high = (tilt_util >= .9) | (speed_util >= .9) | (load_util >= .9) | (np.max(rates, axis=1) >= .9)
        self.high_envelope_count_full += high.astype(np.int64)
        hit, fraction, failed, speed, tilt, outside, terminated, truncated = (
            terminal_events(self, previous_relative, relative)
        )
        reward, self.reward_components = compute_reward(
            self,
            actions,
            old_action,
            relative,
            relative_velocity,
            distance,
            hit,
            failed,
            terminated,
        )
        self.episode_return += reward
        info = episode_records(
            self,
            hit,
            fraction,
            failed,
            speed,
            tilt,
            outside,
            terminated,
            truncated,
            distance,
        )
        return self.observe(), reward.astype(np.float32), terminated, truncated, info

    def snapshot(self):
        u = self.pursuer
        t = self.target
        rel = t.pos - u.pos
        vrel = t.vel - u.vel
        dc, tc = cpa(rel, vrel, self.cfg["reward"]["cpa_horizon"])
        snapshot = {
            "t": self.elapsed.copy(),
            "pursuer_pos": u.pos.copy(),
            "pursuer_vel": u.vel.copy(),
            "pursuer_rot": u.rot.copy(),
            "pursuer_rates": u.rate.copy(),
            "thrust": u.thrust.copy(),
            "pursuer_acc": getattr(u, "acceleration", np.zeros((self.n, 3))).copy(),
            "load_factor": u.load_factor().copy(),
            "target_pos": t.pos.copy(),
            "target_vel": t.vel.copy(),
            "target_acc": t.acc.copy(),
            "relative_distance": np.linalg.norm(rel, axis=1),
            "relative_velocity": vrel.copy(),
            "cpa_distance": dc,
            "cpa_time": tc,
            "action": u.last_action.copy(),
        }
        if hasattr(self, "commanded_action"):
            snapshot["commanded_action"] = self.commanded_action.copy()
        if hasattr(self, "navigation"):
            snapshot["observed_relative_distance"] = np.linalg.norm(
                self.navigation.target.pos - self.navigation.pursuer.pos, axis=1)
        return snapshot
