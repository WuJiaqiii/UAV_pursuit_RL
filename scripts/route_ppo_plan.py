"""Configuration and evaluation rules for the virtual survey comparison."""

import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.survey_route_profiles import ARC, STATIC


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiments/route_ppo_compare.json"
ARMS = ("critic_only", "ppo", "ppo_bc")
MODES = ("smoke", "pilot", "confirm")
VALIDATION_ROUTES = (STATIC, ARC)
ROUTE_SOURCE_FINGERPRINT = "deaec637e145d3cc87fd94140a2356488bd2995d1a27c1ef33d75d3a59d30229"


class IncompatibleRouteInput(ValueError):
    """The checkout or checkpoint cannot run this survey comparison."""


def check_route_source(root):
    # Match the v4 checkpoint fingerprint before importing Torch or its loader.
    digest = hashlib.sha256()
    paths = [root / "main.py", *(root / "pursuit_rl").rglob("*.py"),
             *(root / "tests").rglob("*.py")]
    try:
        for path in sorted(paths):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    except OSError as error:
        raise IncompatibleRouteInput(f"Cannot read survey source in {root}: {error}") from error
    found = digest.hexdigest()
    if found != ROUTE_SOURCE_FINGERPRINT:
        raise IncompatibleRouteInput(
            f"The code in {root} is not the v4 survey implementation required by mixed_bc.pt. "
            f"Source fingerprint: {found[:12]} (expected {ROUTE_SOURCE_FINGERPRINT[:12]}). "
            "Run the command in a clean worktree of origin/refactor/route-ppo-suite; "
            "copying the experiment scripts alone does not install the required implementation."
        )
    return found


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seeds(values, label):
    if (not isinstance(values, list) or not values
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must be an array of unique nonnegative 32-bit seeds")


def positive(group, keys):
    for key in keys:
        if type(group[key]) is not int or group[key] < 1:
            raise ValueError(f"{key} must be a positive integer")


def load_plan(path, mode):
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = ("warmup", "training", "evaluation")
    unknown = data.keys() - set(groups) - set(MODES)
    if unknown:
        raise ValueError(f"Unknown plan sections: {', '.join(sorted(unknown))}")
    run = data[mode]
    unknown = run.keys() - set(groups) - {"seeds", "updates", "evaluate_every"}
    if unknown:
        raise ValueError(f"Unknown {mode} settings: {', '.join(sorted(unknown))}")
    warmup, training, ev = (data[key] | run.get(key, {}) for key in groups)
    positive(warmup, ("episodes", "critic_updates"))
    positive(training, ("envs", "rollout_steps", "batch_size", "epochs"))
    positive(run, ("updates", "evaluate_every"))
    positive(ev, ("episodes_per_seed",))
    if training["batch_size"] > training["envs"] * training["rollout_steps"]:
        raise ValueError("batch_size cannot exceed envs times rollout_steps")
    for key in ("actor_lr", "critic_lr", "bc_coefficient"):
        if type(training[key]) not in (float, int) or not 0 < training[key] < math.inf:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < ev["minimum_joint_rate"] <= 1 or not 0 <= ev["max_tail_rmse_increase_m"] < math.inf:
        raise ValueError("Invalid precision gate settings")
    seeds(run["seeds"], "training seeds")
    seeds(ev["validation_seeds"], "validation seeds")
    seeds(ev["test_seeds"], "test seeds")
    if set(ev["validation_seeds"]) & set(ev["test_seeds"]):
        raise ValueError("Validation and final test seeds must be distinct")
    return dict(mode=mode, warmup=warmup, training=training, evaluation=ev,
                seeds=run["seeds"], updates=run["updates"], evaluate_every=run["evaluate_every"])


def planned_updates(plan, preflight=False):
    return 0 if preflight else len(ARMS) * len(plan["seeds"]) * plan["updates"]


def aggregate(reports):
    def mean(field):
        values = [row[field] for row in reports]
        return statistics.mean(values) if all(value is not None for value in values) else None

    height = [row["mean_tail_axis_rmse_m"] for row in reports]
    return {
        "episodes": sum(row["episodes"] for row in reports),
        "successes": sum(row["successes"] for row in reports),
        "precision_successes": sum(round(row["joint_precision_success_rate"] * row["episodes"]) for row in reports),
        "full_flights": sum(row["full_tail_episodes"] for row in reports),
        "tail_rmse_m": mean("mean_tail_position_rmse"),
        "tail_height_rmse_m": statistics.mean(x[2] for x in height) if all(height) else None,
    }


def eligible(current, baseline, ev):
    return set(current) == set(baseline) == set(VALIDATION_ROUTES) and all(
        row["episodes"] > 0 and row["full_flights"] == row["episodes"]
        and row["precision_successes"] / row["episodes"] >= ev["minimum_joint_rate"]
        and row["tail_rmse_m"] is not None and baseline[name]["tail_rmse_m"] is not None
        and row["tail_rmse_m"] <= baseline[name]["tail_rmse_m"] + ev["max_tail_rmse_increase_m"]
        for name, row in current.items()
    )


def rank(summary):
    row = summary[ARC]
    return (row["precision_successes"], -row["tail_rmse_m"])


def check_frozen_scores(result, baseline):
    """A Critic-only update must preserve fixed-seed deterministic flight metrics."""
    for route in VALIDATION_ROUTES:
        for metric, expected in baseline[route].items():
            actual = result[route][metric]
            if metric.endswith("_m") and expected is not None and actual is not None:
                same = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-5)
            else:
                same = actual == expected
            if not same:
                raise RuntimeError(f"Frozen policy evaluation changed: {route}.{metric} {expected} -> {actual}")
