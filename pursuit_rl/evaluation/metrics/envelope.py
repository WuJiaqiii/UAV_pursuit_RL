import numpy as np
from pursuit_rl.artifacts.trace import normalize_trace


def envelope_metrics(trace, vmax=None, nmax=None, tiltmax_deg=None):
    return _envelope_metrics(normalize_trace(trace), vmax, nmax, tiltmax_deg)


def _envelope_metrics(trace, vmax=None, nmax=None, tiltmax_deg=None):
    z = trace
    vmax = float(z.get("max_speed", 30.0)) if vmax is None else vmax
    nmax = float(z.get("max_load_factor", 30.0 / 9.81)) if nmax is None else nmax
    tiltmax_deg = (
        float(z.get("max_tilt_deg", 65.0)) if tiltmax_deg is None else tiltmax_deg
    )
    if min(vmax, nmax, tiltmax_deg) <= 0:
        raise ValueError("Envelope limits must be positive")
    speed = np.linalg.norm(z["pursuer_vel"], axis=1)
    result = {
        "speed_util_max": float(np.max(speed) / vmax),
        "speed_util_mean": float(np.mean(speed) / vmax),
        "load_factor_max": None,
        "load_factor_rms": None,
        "high_envelope_ratio": None,
        "tilt_max_deg": None,
        "tilt_util_max": None,
    }
    result.update(
        speed_util_p95=float(np.percentile(speed / vmax, 95)),
        speed_high_ratio=float(np.mean(speed / vmax >= .90)),
        load_util_max=None, load_util_p95=None, load_high_ratio=None,
        tilt_util_p95=None, tilt_high_ratio=None,
        rate_util_max=None, rate_util_p95=None, rate_high_ratio=None,
        high_envelope_any_ratio=None,
    )
    per_axis = [speed / vmax >= .90]
    if "load_factor" in z:
        n = z["load_factor"]
        if n.shape != z["t"].shape or not np.isfinite(n).all():
            raise ValueError("load_factor must have finite shape (T,)")
        per_axis.append(n / nmax >= .90)
        result.update(
            load_factor_max=float(np.max(n)),
            load_factor_rms=float(np.sqrt(np.mean(n * n))),
            high_envelope_ratio=float(np.mean(n / nmax > 0.9)),
            load_util_max=float(np.max(n / nmax)),
            load_util_p95=float(np.percentile(n / nmax, 95)),
            load_high_ratio=float(np.mean(n / nmax >= .90)),
        )
    if "pursuer_rot" in z:
        rot = z["pursuer_rot"]
        if rot.shape != (len(z["t"]), 3, 3) or not np.isfinite(rot).all():
            raise ValueError("pursuer_rot must have finite shape (T, 3, 3)")
        tilt = np.rad2deg(np.arccos(np.clip(rot[:, 2, 2], -1.0, 1.0)))
        per_axis.append(tilt / tiltmax_deg >= .90)
        result.update(
            tilt_max_deg=float(np.max(tilt)),
            tilt_util_max=float(np.max(tilt) / tiltmax_deg),
            tilt_util_p95=float(np.percentile(tilt / tiltmax_deg, 95)),
            tilt_high_ratio=float(np.mean(tilt / tiltmax_deg >= .90)),
        )
    if 'pursuer_rates' in z:
        rate = z['pursuer_rates']
        if rate.shape != (len(z['t']), 3) or not np.isfinite(rate).all():
            raise ValueError('pursuer_rates must be finite (T,3)')
        limits = np.array([float(z.get('max_rate_xy', 1.2)),
                           float(z.get('max_rate_xy', 1.2)),
                           float(z.get('max_rate_yaw', 1.0))])
        usage = np.max(np.abs(rate) / limits, axis=1)
        per_axis.append(usage >= .90)
        result.update(rate_util_max=float(np.max(usage)),
                      rate_util_p95=float(np.percentile(usage, 95)),
                      rate_high_ratio=float(np.mean(usage >= .90)))
    result['high_envelope_any_ratio'] = float(np.mean(np.logical_or.reduce(per_axis)))
    if "terminal_record_json" in z:
        import json
        record = json.loads(str(z["terminal_record_json"]))
        if "high_envelope_any_ratio_full" in record:
            result["high_envelope_any_ratio"] = float(record["high_envelope_any_ratio_full"])
    return result
