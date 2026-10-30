"""Batch 6DoF quadrotor with commanded thrust/body-rate response.

NED world (positive Z down), FRD body, thrust acts in -body-Z.
This is a model-based research surrogate, NOT identified flight hardware.
"""

import numpy as np
from pursuit_rl.simulation.geometry import exp_so3


class QuadrotorBatch:
    def __init__(self, n, cfg, rng):
        self.n = n
        self.cfg = cfg
        self.rng = rng
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        self.rate = np.zeros((n, 3))
        self.thrust = np.full(n, cfg["hover_thrust"])
        self.last_action = np.zeros((n, 4))
        self.drag = np.full(n, cfg["drag"])
        self.tau_rate = np.full(n, cfg["tau_rate"])
        self.tau_thrust = np.full(n, cfg["tau_thrust"])
        self.mass_ratio = np.ones(n)
        self.acceleration = np.zeros((n, 3))

    def reset(self, mask, initial_positions=None, randomize=True):
        ids = np.where(mask)[0]
        k = len(ids)
        if not k:
            return
        self.pos[ids] = 0.0 if initial_positions is None else initial_positions
        self.vel[ids] = 0.0
        self.rot[ids] = np.eye(3)
        self.rate[ids] = 0.0
        self.acceleration[ids] = 0.0
        self.thrust[ids] = self.cfg["hover_thrust"]
        self.last_action[ids] = 0.0
        if randomize:
            self.drag[ids] = self.cfg["drag"] * self.rng.uniform(0.8, 1.2, k)
            self.tau_rate[ids] = self.cfg["tau_rate"] * self.rng.uniform(0.85, 1.15, k)
            self.tau_thrust[ids] = self.cfg["tau_thrust"] * self.rng.uniform(
                0.85, 1.15, k
            )
            self.mass_ratio[ids] = self.rng.uniform(0.95, 1.05, k)
        else:
            self.drag[ids] = self.cfg["drag"]
            self.tau_rate[ids] = self.cfg["tau_rate"]
            self.tau_thrust[ids] = self.cfg["tau_thrust"]
            self.mass_ratio[ids] = 1.0

    def step(self, actions, dt):
        p = self.cfg
        old_velocity = self.vel.copy()
        acts = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        thrust_cmd = np.clip(
            p["hover_thrust"] + p["thrust_delta"] * acts[:, 0],
            p["thrust_min"],
            p["thrust_max"],
        )
        rate_cmd = acts[:, 1:] * np.array([p["rate_xy"], p["rate_xy"], p["rate_yaw"]])
        h = dt / int(p["substeps"])
        for _ in range(int(p["substeps"])):
            ar = 1 - np.exp(-h / self.tau_rate)
            at = 1 - np.exp(-h / self.tau_thrust)
            self.rate += (rate_cmd - self.rate) * ar[:, None]
            self.thrust += (thrust_cmd - self.thrust) * at
            self.rot = self.rot @ exp_so3(self.rate * h)
            a = -self.rot[:, :, 2] * (self.thrust / self.mass_ratio)[:, None]
            a[:, 2] += p["gravity"]
            a -= (
                self.drag[:, None]
                * self.vel
                * np.linalg.norm(self.vel, axis=1)[:, None]
            )
            self.vel += a * h
            self.pos += self.vel * h
        self.last_action = acts.copy()
        self.acceleration = (self.vel - old_velocity) / dt
        return self.pos, self.vel

    def fails(self):
        p = self.cfg
        tilt = np.arccos(np.clip(self.rot[:, 2, 2], -1.0, 1.0))
        speed = np.linalg.norm(self.vel, axis=1)
        alt = -self.pos[:, 2]
        outside = (
            np.any(np.abs(self.pos[:, :2]) > p["workspace_xy"], axis=1)
            | (alt < p["min_altitude"])
            | (alt > p["max_altitude"])
        )
        failed = (
            outside | (tilt > np.deg2rad(p["max_tilt_deg"])) | (speed > p["max_speed"])
        )
        return failed, speed, tilt, outside

    def load_factor(self):
        """Thrust-specific load / g using filtered thrust and randomized mass."""
        return self.thrust / np.maximum(self.mass_ratio * self.cfg["gravity"], 1e-8)
