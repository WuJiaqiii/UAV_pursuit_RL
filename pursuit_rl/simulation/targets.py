"""Vectorized motion generator: straight, Singer OU, bounded Dubins, hybrid.

Targets are 3D kinematic agents. Random motion is causal, bounded by speed,
acceleration, jerk smoothness and a finite workspace. Not a second quadrotor.
"""

import numpy as np

MODES = {"straight": 0, "singer": 1, "dubins": 2, "hybrid": 3}


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-9)


class TargetBatch:
    def __init__(self, n, cfg, rng):
        self.n = n
        self.cfg = cfg
        self.rng = rng
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.acc = np.zeros((n, 3))
        self.mode = np.zeros(n, np.int64)
        self.heading = np.zeros(n)
        self.speed_setting = np.zeros(n)
        self.max_accel = np.full(n, cfg["max_accel"])
        self.phase_timer = np.zeros(n)
        self.phase_length = np.ones(n)
        self.ou_acc = np.zeros((n, 3))
        self.desired_acc = np.zeros((n, 3))
        self.change_count = np.zeros(n, dtype=np.int64)
        self.max_observed_speed = np.zeros(n)
        self.max_observed_acc = np.zeros(n)

    def reset(self, mask, stage):
        ids = np.where(mask)[0]
        k = len(ids)
        if not k:
            return
        mode_names = stage["modes"]
        self.mode[ids] = self.rng.choice([MODES[m] for m in mode_names], size=k)
        self.speed_setting[ids] = self.rng.uniform(*stage["speed"], size=k)
        self.max_accel[ids] = stage["max_accel"]
        self.heading[ids] = self.rng.uniform(-np.pi, np.pi, size=k)
        self.pos[ids] = 0.0
        self.pos[ids, :2] = self.rng.uniform(-70, 70, size=(k, 2))
        self.pos[ids, 2] = -self.rng.uniform(75, 130, size=k)
        self.vel[ids, :2] = (
            np.stack([np.cos(self.heading[ids]), np.sin(self.heading[ids])], axis=1)
            * self.speed_setting[ids, None]
        )
        self.vel[ids, 2] = (
            self.rng.uniform(-0.05, 0.05, size=k) * self.speed_setting[ids]
        )
        if self.cfg.get("strict_speed_ceiling", False):
            self.vel[ids] = _unit(self.vel[ids]) * self.speed_setting[ids, None]
        self.acc[ids] = 0.0
        self.ou_acc[ids] = 0.0
        self.desired_acc[ids] = 0.0
        self.phase_timer[ids] = 0.0
        self.phase_length[ids] = self.rng.uniform(
            *self.cfg["maneuver_interval"], size=k
        )
        self.change_count[ids] = 0
        self.max_observed_speed[ids] = np.linalg.norm(self.vel[ids], axis=1)
        self.max_observed_acc[ids] = 0.0

    def _change_modes(self, ids):
        if not len(ids):
            return
        # Target mode persists; reselect maneuver acceleration every segment.
        self.phase_timer[ids] = 0.0
        self.phase_length[ids] = self.rng.uniform(
            *self.cfg["maneuver_interval"], size=len(ids)
        )
        self.desired_acc[ids] = self.rng.normal(size=(len(ids), 3))
        self.desired_acc[ids, 2] *= 0.45
        self.desired_acc[ids] = _unit(self.desired_acc[ids]) * self.max_accel[ids, None]
        pattern = self.cfg["hybrid_pattern"]
        if pattern == "switchback":
            # Flip lateral acceleration at each change, not independently random.
            right = np.stack(
                [-self.vel[ids, 1], self.vel[ids, 0], np.zeros(len(ids))], axis=1
            )
            lateral = _unit(right)
            signs = np.where(self.change_count[ids] % 2 == 0, 1.0, -1.0)
            self.desired_acc[ids] = lateral * signs[:, None] * self.max_accel[ids, None]
        elif pattern == "pulse":
            # Alternating acceleration/braking along the current velocity.
            signs = np.where(self.change_count[ids] % 2 == 0, 1.0, -1.0)
            self.desired_acc[ids] = (
                _unit(self.vel[ids]) * signs[:, None] * self.max_accel[ids, None]
            )
        self.change_count[ids] += 1

    def step(self, dt):
        c = self.cfg
        n = self.n
        old_vel = self.vel.copy()
        self.phase_timer += dt
        new_ids = np.flatnonzero(self.phase_timer >= self.phase_length)
        self._change_modes(new_ids)
        acc_cmd = np.zeros((n, 3))
        mode = self.mode
        singer = mode == MODES["singer"]
        dubins = mode == MODES["dubins"]
        hybrid = mode == MODES["hybrid"]
        # Exact OU transition, stationary std sigma; physically clipped below.
        if np.any(singer):
            alpha = c["singer_alpha"]
            decay = np.exp(-alpha * dt)
            ids = np.where(singer)[0]
            self.ou_acc[ids] = decay * self.ou_acc[ids] + c["singer_sigma"] * np.sqrt(
                1 - decay**2
            ) * self.rng.normal(size=(len(ids), 3))
            # Legacy damping 0.985 per 50 ms; preserve its physical decay
            # when the simulator changes from 20 Hz to 1 kHz.
            self.ou_acc[ids, 2] *= 0.985 ** (dt / 0.05)
            acc_cmd[ids] = self.ou_acc[ids]
        if np.any(dubins):
            ids = np.where(dubins)[0]
            # Constant-ish speed with bounded curvature, hold turning rate by segment.
            turn_dir = np.sign(self.desired_acc[ids, 0])
            turn_dir = np.where(turn_dir == 0, 1.0, turn_dir)
            wmax = np.minimum(
                c["turn_rate"],
                self.max_accel[ids] / np.maximum(self.speed_setting[ids], 1.0),
            )
            rate = turn_dir * wmax
            self.heading[ids] += rate * dt
            desired = np.zeros((len(ids), 3))
            desired[:, 0] = self.speed_setting[ids] * np.cos(self.heading[ids])
            desired[:, 1] = self.speed_setting[ids] * np.sin(self.heading[ids])
            acc_cmd[ids] = (desired - self.vel[ids]) / dt
        if np.any(hybrid):
            ids = np.where(hybrid)[0]
            # Smoothly turn toward a randomly sampled acceleration direction.
            self.ou_acc[ids] += (self.desired_acc[ids] - self.ou_acc[ids]) * min(
                1.0, dt / c["hybrid_smoothing_s"]
            )
            acc_cmd[ids] = self.ou_acc[ids]
        # Bias acceleration gently inward near target workspace boundaries.
        xy_lim = c["target_workspace_xy"]
        lo = -c["target_max_altitude"]
        hi = -c["target_min_altitude"]
        for j in range(2):
            dist = np.maximum(np.abs(self.pos[:, j]) - 0.80 * xy_lim, 0.0)
            acc_cmd[:, j] += -np.sign(self.pos[:, j]) * np.minimum(
                dist * 0.08, c["max_accel"]
            )
        acc_cmd[:, 2] += np.where(self.pos[:, 2] < lo + 20.0, c["max_accel"], 0.0)
        acc_cmd[:, 2] -= np.where(self.pos[:, 2] > hi - 20.0, c["max_accel"], 0.0)
        # Speed governor: remove outward acceleration, never teleport velocity.
        vnorm = np.linalg.norm(self.vel, axis=1)
        moving_out = np.sum(acc_cmd * self.vel, axis=1) > 0
        too_fast = (
            vnorm >= np.minimum(self.speed_setting + 1.0, c["max_speed"])
        ) & moving_out
        projection = np.sum(acc_cmd * self.vel, axis=1) / np.maximum(
            vnorm * vnorm, 1e-9
        )
        acc_cmd[too_fast] -= projection[too_fast, None] * self.vel[too_fast]
        acc_norm = np.linalg.norm(acc_cmd, axis=1)
        acc_cmd *= np.minimum(1.0, self.max_accel / np.maximum(acc_norm, 1e-9))[:, None]
        # Semimplicit integration, actual acceleration exactly bounded by command.
        # Enforce the declared maximum target speed by limiting the step's
        # acceleration, rather than clipping the resulting velocity. If the
        # unconstrained next velocity crosses the speed sphere, solve for the
        # largest feasible acceleration fraction in [0, 1].
        max_speed = float(c["max_speed"])
        proposed = self.vel + acc_cmd * dt
        above = (
            (np.linalg.norm(proposed, axis=1) > max_speed + 1e-10)
            if c.get("strict_speed_ceiling", False)
            else np.zeros(n, dtype=bool)
        )
        if np.any(above):
            ids = np.flatnonzero(above)
            v = self.vel[ids]
            delta = acc_cmd[ids] * dt
            aa = np.sum(delta * delta, axis=1)
            bb = 2 * np.sum(v * delta, axis=1)
            cc = np.sum(v * v, axis=1) - max_speed**2
            discr = np.maximum(bb * bb - 4 * aa * cc, 0.0)
            alpha = np.where(
                aa > 1e-16, (-bb + np.sqrt(discr)) / np.maximum(2 * aa, 1e-16), 0.0
            )
            acc_cmd[ids] *= np.clip(alpha, 0.0, 1.0)[:, None]
        self.vel += acc_cmd * dt
        self.pos += self.vel * dt
        self.acc = (self.vel - old_vel) / dt
        self.max_observed_speed = np.maximum(
            self.max_observed_speed, np.linalg.norm(self.vel, axis=1)
        )
        self.max_observed_acc = np.maximum(
            self.max_observed_acc, np.linalg.norm(self.acc, axis=1)
        )
        return self.pos, self.vel, self.acc
