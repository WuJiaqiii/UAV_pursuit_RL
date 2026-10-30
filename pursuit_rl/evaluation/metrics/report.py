"""Canonical aggregation and report writers. Missing values remain missing."""

import csv
from pathlib import Path
import numpy as np
from pursuit_rl.evaluation.metrics.statistics import mission_summary

NUMERIC_METRICS = (
    "min_distance",
    "min_sampled_distance",
    "observed_duration",
    "speed_util_max",
    "tilt_max_deg",
    "load_factor_max",
    "load_factor_rms",
    "lateral_accel_cost",
    "lateral_accel_mean_square",
    "control_variation",
    "initial_cpa",
    "final_cpa",
    "mean_cpa",
    "cpa_improvement", "cpa_integral", "late_cpa_mean",
    "speed_util_p95", "speed_high_ratio", "load_util_max", "load_util_p95",
    "load_high_ratio", "tilt_util_p95", "tilt_high_ratio", "rate_util_max",
    "rate_util_p95", "rate_high_ratio", "high_envelope_any_ratio",
    "control_variation_per_s", "command_tracking_l2_mean",
)


def aggregate(records):
    if not records:
        raise ValueError("Cannot aggregate an empty metric set")
    n = len(records)
    result = mission_summary(records)
    for key in NUMERIC_METRICS:
        values = [r[key] for r in records if r.get(key) is not None]
        result[key + "_mean"] = float(np.mean(values)) if values else None
        result[key + "_n"] = len(values)
    for key in ("lateral_accel_cost", "control_variation"):
        values = [r[key] for r in records if r["success"] and r.get(key) is not None]
        result[key + "_success_mean"] = float(np.mean(values)) if values else None
    for horizon in (30, 60, 120):
        result[f"capture_by_{horizon}s"] = (
            sum(r["success"] and r["capture_time"] <= horizon for r in records) / n
        )
    return result


def write_summary(records, out="summary.csv"):
    if not records:
        raise ValueError("Cannot write an empty summary")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(records)


def _tex(value):
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    chars = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(chars.get(c, c) for c in str(value))


def write_latex(records, out="summary_table.tex"):
    cols = [
        "algorithm",
        "scenario",
        "episodes",
        "success_rate",
        "mean_capture_time_success",
        "mean_capped_time",
    ]
    lines = [
        r"\begin{tabular}{llrrrr}",
        " & ".join(map(_tex, cols)) + r" \\",
        r"\hline",
    ]
    for r in records:
        lines.append(" & ".join(_tex(r.get(c)) for c in cols) + r" \\")
    lines.append(r"\end{tabular}")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
