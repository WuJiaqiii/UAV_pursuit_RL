"""Shared mission counts, first-entry times and binomial confidence intervals."""

import math
import numpy as np


def wilson(k, n, z=1.95996398454):
    if not n:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - rad, center + rad


def mission_summary(records):
    if not records:
        raise ValueError("Cannot summarize an empty mission set")
    n = len(records)
    successes = sum(bool(row["success"]) for row in records)
    low, high = wilson(successes, n)
    result = {
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "wilson_low": low,
        "wilson_high": high,
    }
    for reason in ("speed", "tilt", "outside", "timeout", "unknown"):
        result["failures_" + reason] = sum(
            not row["success"] and row["reason"] == reason for row in records
        )
    for source, target in [
        ("capture_time", "mean_capture_time_success"),
        ("capped_time", "mean_capped_time"),
    ]:
        values = [row[source] for row in records if row.get(source) is not None]
        result[target] = float(np.mean(values)) if values else None
        result[target + "_n"] = len(values)
    return result
