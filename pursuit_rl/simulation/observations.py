"""Named actor/critic features in NED/FRD; arrays have leading batch axis N."""

import numpy as np
from pursuit_rl.simulation.geometry import body_frame, cpa

ACTOR_DIM = 31
CRITIC_DIM = 43
ACTOR_SLICES = {
    "relative_position": slice(0, 3),
    "relative_velocity": slice(3, 6),
    "target_acceleration": slice(6, 9),
    "pursuer_velocity": slice(9, 12),
    "attitude": slice(12, 21),
    "body_rates": slice(21, 24),
    "previous_action": slice(24, 28),
    "cpa": slice(28, 30),
    "remaining_time": slice(30, 31),
    "boundary": slice(31, 34),
}


def observation_dimensions(cfg):
    extra = 3 if cfg["observation"]["boundary_features"] else 0
    return ACTOR_DIM + extra, CRITIC_DIM + extra


def boundary_vector(pos, physics):
    limit = physics["workspace_xy"]
    low, high = -physics["max_altitude"], -physics["min_altitude"]
    xy = np.where(pos[:, :2] >= 0, limit - pos[:, :2], -limit - pos[:, :2])
    dz = np.where(pos[:, 2] > (low + high) / 2, high - pos[:, 2], low - pos[:, 2])
    return np.column_stack((xy, dz))


def encode_observations(env):
    u = env.pursuer
    t = env.target
    rel = t.pos - u.pos
    vrel = t.vel - u.vel
    d_cpa, t_cpa = cpa(rel, vrel, env.cfg["reward"]["cpa_horizon"])
    rem = (env.stage_config["time"] - env.elapsed) / env.stage_config["time"]
    # Body FRD velocity information; attitude matrix global NED => orientation-gravity info.
    cpa_d_feature = (
        d_cpa / env.cfg["reward"]["distance_scale"]
        if env.cfg.get("observation", {}).get("use_cpa", True)
        else np.zeros(env.n)
    )
    cpa_t_feature = (
        t_cpa / env.cfg["reward"]["cpa_horizon"]
        if env.cfg.get("observation", {}).get("use_cpa", True)
        else np.zeros(env.n)
    )
    actor = np.concatenate(
        [
            body_frame(u.rot, rel) / env.cfg["reward"]["distance_scale"],
            body_frame(u.rot, vrel) / env.cfg["reward"]["speed_scale"],
            body_frame(u.rot, t.acc) / env.cfg["observation"]["target_accel_scale"],
            body_frame(u.rot, u.vel) / env.cfg["reward"]["speed_scale"],
            u.rot.reshape(env.n, 9),
            u.rate[:, :]
            / max(env.cfg["physics"]["rate_xy"], env.cfg["physics"]["rate_yaw"]),
            u.last_action,
            cpa_d_feature[:, None],
            cpa_t_feature[:, None],
            rem[:, None],
            *(
                [
                    body_frame(u.rot, boundary_vector(u.pos, env.cfg["physics"]))
                    / env.cfg["reward"]["distance_scale"]
                ]
                if env.cfg.get("observation", {}).get("boundary_features", False)
                else []
            ),
        ],
        axis=1,
    ).astype(np.float32)
    # 12 privileged dimensions: true filtered thrust, filtered rates (4), drag,
    # two lag constants, mass ratio, 4 one-hot target motion modes.
    mode_onehot = np.eye(4, dtype=np.float32)[t.mode]
    privilege = np.concatenate(
        [
            (u.thrust / env.cfg["physics"]["hover_thrust"])[:, None],
            u.rate / max(env.cfg["physics"]["rate_xy"], 1.0e-6),
            (u.drag / max(env.cfg["physics"]["drag"], 1.0e-6))[:, None],
            (u.tau_rate / env.cfg["physics"]["tau_rate"])[:, None],
            (u.tau_thrust / env.cfg["physics"]["tau_thrust"])[:, None],
            u.mass_ratio[:, None],
            mode_onehot,
        ],
        axis=1,
    ).astype(np.float32)
    critic = np.concatenate([actor, privilege], axis=1)
    ad, cd = observation_dimensions(env.cfg)
    assert actor.shape == (env.n, ad), (actor.shape, ad)
    assert critic.shape == (env.n, cd), (critic.shape, cd)
    return actor, critic
