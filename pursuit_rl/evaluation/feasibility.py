"""Optimistic constant-velocity intercept geometry diagnostics.

Assumptions: target holds current velocity forever, pursuer is already able
instantaneously to fly in any direction at max_speed, ignores acceleration,
turning radius, altitude, workspace, and capture radius. The solution is a
lower bound on needed time *under this hypothetical motion scenario*, NOT a
reachability certificate for maneuvering targets.
"""

import csv
import json
from pathlib import Path
import numpy as np


def ideal_intercept_time(r, target_vel, pursuer_max_speed):
    r = np.asarray(r, dtype=float)
    v = np.asarray(target_vel, dtype=float)
    d = np.linalg.norm(r)
    if d == 0:
        return 0.0
    # Solve interception of the target center under constant velocity.
    aa = float(np.dot(v, v) - pursuer_max_speed**2)
    bb = float(2 * np.dot(r, v))
    cc = float(np.dot(r, r))
    if abs(aa) < 1e-12:
        return float(-cc / bb) if bb < 0 else float("inf")
    discr = bb * bb - 4 * aa * cc
    if discr < 0:
        return float("inf")
    roots = sorted(
        x
        for x in ((-bb - np.sqrt(discr)) / (2 * aa), (-bb + np.sqrt(discr)) / (2 * aa))
        if x >= 0
    )
    return float(roots[0]) if roots else float("inf")


def enrich_episode(e, max_speed=30.0, horizon=120.0):
    e = dict(e)
    if "start_rel_pos" not in e or "start_target_vel" not in e:
        return e
    t = ideal_intercept_time(e["start_rel_pos"], e["start_target_vel"], max_speed)
    e["ideal_cv_intercept_s"] = t if np.isfinite(t) else None
    e["ideal_cv_ratio"] = min(t / horizon, 99.0) if np.isfinite(t) else None
    e["ideal_cv_category"] = (
        "no_const_velocity_solution"
        if not np.isfinite(t)
        else (
            "near_limit"
            if t >= 0.75 * horizon
            else ("medium" if t >= 0.4 * horizon else "easy")
        )
    )
    return e


def produce(inputs, out, max_speed=None):
    records = []
    for argument in inputs:
        candidate = Path(argument)
        paths = (
            sorted(candidate.rglob("episodes.json"))
            if candidate.is_dir()
            else [candidate]
        )
        for path in paths:
            records.extend(_episode_rows(path, max_speed))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No episode entries found")
    with out.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def _episode_rows(path, max_speed):
    records = []
    x = json.loads(Path(path).read_text(encoding="utf8"))
    max_speed = (
        max_speed
        if max_speed is not None
        else x.get("evaluation_protocol", {}).get("pursuer_max_speed")
    )
    if max_speed is None or max_speed <= 0:
        raise ValueError(
            "Report requires pursuer_max_speed metadata or an explicit --max-speed"
        )
    for episode in x["episodes_detail"]:
        e = enrich_episode(
            episode,
            max_speed=max_speed,
            horizon=float(episode.get("time_limit", 120.0)),
        )
        records.append(
            {
                "file": str(path),
                "scenario": x.get("evaluation_protocol", {}).get(
                    "scenario", path.parent.name
                ),
                "success": int(e["success"]),
                "reason": e["reason"],
                "start_distance": e["start_distance"],
                "start_closing_speed": e.get("start_closing_speed"),
                "initial_geometry": e.get("initial_geometry", "random"),
                "ideal_cv_intercept_s": e.get("ideal_cv_intercept_s"),
                "ideal_cv_category": e.get("ideal_cv_category"),
                "time_limit": episode.get("time_limit", 120.0),
                "reach_time": e["reach_time"],
            }
        )
    return records
