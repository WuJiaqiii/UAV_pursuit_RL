import numpy as np
from pursuit_rl.artifacts.trace import normalize_trace


def lateral_acceleration_cost(trace, dt=None):
    return _lateral_acceleration_cost(normalize_trace(trace), dt)


def _lateral_acceleration_cost(trace, dt=None):
    """Integral of lateral acceleration squared using actual trace intervals."""
    z = trace
    v = z["pursuer_vel"]
    if len(v) < 2:
        return 0.0
    intervals = np.diff(z["t"])
    if dt is not None and not np.allclose(intervals, dt):
        raise ValueError("Supplied dt disagrees with trace timestamps")
    a = np.diff(v, axis=0) / intervals[:, None]
    vh = v[:-1] / np.maximum(np.linalg.norm(v[:-1], axis=1, keepdims=True), 1e-8)
    lat = a - np.sum(a * vh, axis=1, keepdims=True) * vh
    return float(np.sum(np.sum(lat * lat, axis=1) * intervals))


def control_variation(trace):
    return _control_variation(normalize_trace(trace))


def _control_variation(trace):
    z = trace
    if "action" not in z:
        return None
    u = z["action"]
    if u.shape != (len(z["t"]), 4) or not np.isfinite(u).all():
        raise ValueError("action must have finite shape (T, 4)")
    # Exclude the reset-to-first-command jump: t=0 has no prior command.
    commands = u[1:] if "trace_schema" in z else u
    if len(commands) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(commands, axis=0), axis=1)))


def maneuver_metrics(trace):
    return _maneuver_metrics(normalize_trace(trace))


def _maneuver_metrics(trace):
    z = trace
    cost = _lateral_acceleration_cost(z)
    duration = float(z["t"][-1] - z["t"][0])
    commands = z.get('commanded_action')
    effective = z.get('action')
    tracking = None
    if commands is not None and effective is not None:
        if commands.shape != effective.shape:
            raise ValueError('commanded_action and action shapes differ')
        tracking = float(np.mean(np.linalg.norm(commands - effective, axis=1)))
    variation = _control_variation(z)
    if 'terminal_record_json' in z:
        import json
        rec = json.loads(str(z['terminal_record_json']))
        variation = rec.get('control_variation_full', variation)
        cost = rec.get('lateral_accel_cost_full', cost)
    return {
        "control_variation_per_s": variation / duration if variation is not None and duration > 0 else None,
        "command_tracking_l2_mean": tracking,
        "lateral_accel_cost": cost,
        "control_variation": variation,
        "lateral_accel_mean_square": cost / duration if duration > 0 else None,
    }
