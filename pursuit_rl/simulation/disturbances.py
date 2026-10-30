"""Evaluation-only sensor/command disturbances. No training configuration changes.

A deterministic seeded measurement channel is shared by PPO and classical baselines.
Position/velocity and target acceleration are all delayed, then noise is applied.
The low-level attitude/rate stabilization retains the local onboard IMU state.
Delays are integer multiples of the environment control period.
"""
from collections import deque
from types import SimpleNamespace
import numpy as np
from pursuit_rl.simulation.geometry import body_frame, cpa

DEFAULT_DISTURBANCE = dict(position_std_m=0.0, velocity_std_mps=0.0,
                           target_accel_std_mps2=0.0, observation_delay_s=0.0,
                           command_delay_s=0.0)


def validate_disturbance(spec, dt):
    spec = {**DEFAULT_DISTURBANCE, **(spec or {})}
    unknown = set(spec) - set(DEFAULT_DISTURBANCE)
    if unknown:
        raise ValueError(f"Unknown disturbance: {sorted(unknown)}")
    for key, value in spec.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    for name in ('observation_delay_s', 'command_delay_s'):
        n = round(spec[name] / dt)
        if abs(spec[name] - n * dt) > 1e-8:
            raise ValueError(f"{name} must be an integral multiple of dt={dt}")
    return {k: float(v) for k, v in spec.items()}


class SensorChannel:
    def __init__(self, cfg, spec, seed):
        self.cfg = cfg
        self.spec = validate_disturbance(spec, cfg['physics']['dt'])
        self.lag = round(self.spec['observation_delay_s'] / cfg['physics']['dt'])
        self.states = deque(maxlen=self.lag + 1)
        self.rng = np.random.default_rng(int(seed) + 460001)

    def observe(self, env, original_actor):
        measured = {
            'p_pos': env.pursuer.pos.copy(), 'p_vel': env.pursuer.vel.copy(),
            't_pos': env.target.pos.copy(), 't_vel': env.target.vel.copy(),
            't_acc': env.target.acc.copy(),
        }
        self.states.append(measured)
        delayed = self.states[0]
        s = self.spec
        ppos = delayed['p_pos'] + self.rng.normal(0, s['position_std_m'], delayed['p_pos'].shape)
        tpos = delayed['t_pos'] + self.rng.normal(0, s['position_std_m'], delayed['t_pos'].shape)
        pvel = delayed['p_vel'] + self.rng.normal(0, s['velocity_std_mps'], delayed['p_vel'].shape)
        tvel = delayed['t_vel'] + self.rng.normal(0, s['velocity_std_mps'], delayed['t_vel'].shape)
        tacc = delayed['t_acc'] + self.rng.normal(0, s['target_accel_std_mps2'], delayed['t_acc'].shape)
        # The IMU attitude and body rates are local and unperturbed by target tracking noise.
        env.navigation = SimpleNamespace(
            pursuer=SimpleNamespace(pos=ppos, vel=pvel),
            target=SimpleNamespace(pos=tpos, vel=tvel, acc=tacc),
        )
        rel, rv = tpos - ppos, tvel - pvel
        rot = env.pursuer.rot
        actor = original_actor.copy()
        actor[:, 0:3] = body_frame(rot, rel) / self.cfg['reward']['distance_scale']
        actor[:, 3:6] = body_frame(rot, rv) / self.cfg['reward']['speed_scale']
        actor[:, 6:9] = body_frame(rot, tacc) / self.cfg['observation']['target_accel_scale']
        actor[:, 9:12] = body_frame(rot, pvel) / self.cfg['reward']['speed_scale']
        if self.cfg['observation']['use_cpa']:
            dc, tc = cpa(rel, rv, self.cfg['reward']['cpa_horizon'])
            actor[:, 28] = dc / self.cfg['reward']['distance_scale']
            actor[:, 29] = tc / self.cfg['reward']['cpa_horizon']
        return actor.astype(np.float32)


class CommandChannel:
    def __init__(self, cfg, spec, n):
        s = validate_disturbance(spec, cfg['physics']['dt'])
        self.lag = round(s['command_delay_s'] / cfg['physics']['dt'])
        self.history = deque(maxlen=self.lag + 1)
        self.n = n

    def apply(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (self.n, 4):
            raise ValueError('CTBR shape mismatch')
        self.history.append(command.copy())
        if len(self.history) <= self.lag:
            return np.zeros_like(command)  # neutral hover and no body rate
        return self.history[0].copy()
