"""Terminal events and episode records; success precedes same-step failure."""

import numpy as np
from pursuit_rl.simulation.geometry import segment_first_hit


def terminal_events(env, previous_relative, relative):
    hit, fraction = segment_first_hit(
        previous_relative, relative, env.stage_config["success_radius"]
    )
    failed, speed, tilt, outside = env.pursuer.fails()
    terminated = hit | failed
    truncated = (env.elapsed >= env.stage_config["time"] - 1e-8) & ~terminated
    return hit, fraction, failed, speed, tilt, outside, terminated, truncated


def terminal_reason(hit, speed, outside, failed, max_speed):
    if hit:
        return "success"
    if speed > max_speed:
        return "speed"
    if outside:
        return "outside"
    if failed:
        return "tilt"
    return "timeout"


def episode_records(
    env, hit, frac, failed, speed, tilt, outside, terminated, truncated, dist
):
    dt = env.cfg["physics"]["dt"]
    info = []
    for i in range(env.n):
        record = {}
        if terminated[i] or truncated[i]:
            reason = terminal_reason(
                hit[i], speed[i], outside[i], failed[i], env.cfg["physics"]["max_speed"]
            )
            reach_time = float(env.elapsed[i] - dt + frac[i] * dt) if hit[i] else None
            record = {
                "success": bool(hit[i]),
                "reason": reason,
                "stage": int(env.stage),
                "time": float(env.elapsed[i]),
                "reach_time": reach_time,
                "start_distance": float(env.initial_distance[i]),
                "start_closing_speed": float(env.initial_closing[i]),
                "start_cpa_distance": float(env.initial_cpa[i]),
                "start_rel_pos": env.initial_rel_pos[i].tolist(),
                "start_target_vel": env.initial_target_vel[i].tolist(),
                "initial_geometry": env.stage_config.get("initial_geometry", "random"),
                "terminal_distance": float(dist[i]),
                "minimum_distance": float(min(env.min_distance[i], dist[i])),
                "speed": float(speed[i]),
                "tilt_deg": float(np.rad2deg(tilt[i])),
                "return": float(env.episode_return[i]),
                "control_variation_full": float(env.control_variation_full[i]),
                "lateral_accel_cost_full": float(env.lateral_accel_cost_full[i]),
                "high_envelope_any_ratio_full": float(env.high_envelope_count_full[i] / max(int(env.steps[i]), 1)),
                "capture_pursuer_pos": (env.previous_pursuer_pos[i] + frac[i] * (env.pursuer.pos[i] - env.previous_pursuer_pos[i])).tolist() if hit[i] else None,
                "capture_target_pos": (env.previous_target_pos[i] + frac[i] * (env.target.pos[i] - env.previous_target_pos[i])).tolist() if hit[i] else None,
                "target_speed_peak": float(env.target.max_observed_speed[i]),
                "target_acc_peak": float(env.target.max_observed_acc[i]),
                "target_maneuvers": int(env.target.change_count[i]),
                "target_mode": int(env.target.mode[i]),
                "is_truncated": bool(truncated[i]),
                "time_limit": float(env.stage_config["time"]),
            }
        info.append(record)
    return info
