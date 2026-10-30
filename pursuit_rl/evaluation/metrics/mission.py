"""First-hit metrics using authoritative environment events when available."""

import numpy as np
from pursuit_rl.artifacts.trace import normalize_trace
from pursuit_rl.simulation.geometry import segment_first_hit


def mission_metrics(trace, capture_radius=None):
    return _mission_metrics(normalize_trace(trace), capture_radius)


def _mission_metrics(trace, capture_radius=None):
    z = trace
    radius = float(z["capture_radius"]) if "capture_radius" in z else capture_radius
    if radius is None or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Legacy trace requires an explicit positive capture radius")
    if (
        capture_radius is not None
        and "capture_radius" in z
        and not np.isclose(radius, capture_radius)
    ):
        raise ValueError(
            "Cannot relabel an early-terminated trace with another capture radius"
        )
    d, t = z["relative_distance"], z["t"]
    minimum = float(np.min(d))
    positions = all(k in z for k in ("target_pos", "pursuer_pos"))
    if positions and len(t) > 1:
        rel = z["target_pos"] - z["pursuer_pos"]
        delta = np.diff(rel, axis=0)
        alpha = np.clip(
            -np.sum(rel[:-1] * delta, axis=1)
            / np.maximum(np.sum(delta * delta, axis=1), 1e-12),
            0.0,
            1.0,
        )
        minimum = min(
            minimum,
            float(np.min(np.linalg.norm(rel[:-1] + alpha[:, None] * delta, axis=1))),
        )
    if "terminal_success" in z:
        if "terminal_record_json" in z:
            import json
            rec = json.loads(str(z["terminal_record_json"]))
            # For downsampled traces, do not reconstruct a fictitious CPA
            # event between far-apart saved samples. The authoritative
            # minimum is accumulated at the true integration rate.
            if int(z.get("trace_stride", 1)) > 1:
                minimum = float(rec.get("minimum_distance", minimum))
                if bool(z["terminal_success"]):
                    minimum = min(minimum, float(radius))
        success = bool(z["terminal_success"])
        reason = str(z["terminal_reason"])
        if reason not in (
            "success",
            "speed",
            "tilt",
            "outside",
            "timeout",
        ) or success != (reason == "success"):
            raise ValueError("Inconsistent terminal success/reason metadata")
        reach = float(z["terminal_reach_time"]) if success else None
        if success and (
            not np.isfinite(reach) or reach < t[0] - 1e-8 or reach > t[-1] + 1e-8
        ):
            raise ValueError("Invalid terminal reach time")
        source = "environment_event"
    else:
        success = bool(d[0] <= radius)
        reach = float(t[0]) if success else None
        source = "segment_reconstruction"
        if not success and positions and len(t) > 1:
            hits, fractions = segment_first_hit(rel[:-1], rel[1:], radius)
            if hits.any():
                i = np.flatnonzero(hits)[0]
                success = True
                reach = float(t[i] + fractions[i] * (t[i + 1] - t[i]))
        elif not positions:
            hits = np.flatnonzero(d <= radius)
            success = bool(len(hits))
            reach = float(t[hits[0]]) if success else None
            source = "sampled_distance_only"
        reason = "success" if success else "unknown"
    horizon = float(z["time_limit"]) if "time_limit" in z else None
    return {
        "success": success,
        "capture_time": reach,
        "capture_radius": float(radius),
        "min_distance": minimum,
        "min_sampled_distance": float(np.min(d)),
        "reason": reason,
        "mission_source": source,
        "observed_duration": float(t[-1] - t[0]),
        "capped_time": reach if success else horizon,
    }
