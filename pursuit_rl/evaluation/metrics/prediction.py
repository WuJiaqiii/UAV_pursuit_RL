import numpy as np
from pursuit_rl.simulation.geometry import cpa
from pursuit_rl.artifacts.trace import normalize_trace


def cpa_distance(rel_pos, rel_vel, horizon=2.0):
    d, t = cpa(np.asarray(rel_pos)[None], np.asarray(rel_vel)[None], horizon)
    return float(d[0]), float(t[0])


def cpa_metrics(trace, horizon=None):
    return _cpa_metrics(normalize_trace(trace), horizon)


def _cpa_metrics(trace, horizon=None):
    z = trace
    horizon = float(z.get("cpa_horizon", 2.0)) if horizon is None else horizon
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("CPA horizon must be finite and positive")
    rp = z["target_pos"] - z["pursuer_pos"]
    rv = z["target_vel"] - z["pursuer_vel"]
    d, _ = cpa(rp, rv, horizon)
    return {
        "initial_cpa": float(d[0]),
        "final_cpa": float(d[-1]),
        "mean_cpa": float(np.mean(d)),
        "cpa_improvement": float(d[0] - d[-1]),
        "cpa_integral": float(np.sum(.5 * (d[1:] + d[:-1]) * np.diff(z['t']))) if len(d) > 1 else 0.,
        "late_cpa_mean": float(np.mean(d[z['t'] >= z['t'][-1] - min(5., (z['t'][-1] - z['t'][0]) / 2)])),
        "cpa_horizon": horizon,
    }
