"""Paired comparisons consume explicit episode reports and verify the task protocol."""

import json
import math
from pathlib import Path
import numpy as np


def exact_mcnemar_p(rescued, lost):
    n = rescued + lost
    if not n:
        return 1.0
    k = min(rescued, lost)
    vals = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        - n * math.log(2)
        for i in range(k + 1)
    ]
    hi = max(vals)
    return min(1.0, 2 * math.exp(hi) * sum(math.exp(v - hi) for v in vals))


def _report(path):
    path = Path(path)
    if path.is_dir():
        path = path / "episodes.json"
    data = json.loads(path.read_text(encoding="utf8"))
    if (
        not isinstance(data, dict)
        or "episodes_detail" not in data
        or "evaluation_protocol" not in data
    ):
        raise ValueError(
            "Expected a current evaluation report or an algorithm/scenario folder containing episodes.json"
        )
    protocol = data["evaluation_protocol"]
    required = {"version", "seed", "batch_size", "episodes", "stage", "task_hash"}
    if not isinstance(protocol, dict) or not required.issubset(protocol):
        raise ValueError("Report has an incomplete evaluation protocol")
    return data


def compare(a, b):
    a, b = _report(a), _report(b)
    for field in ["version", "seed", "batch_size", "episodes", "stage", "task_hash"]:
        if a["evaluation_protocol"].get(field) != b["evaluation_protocol"].get(field):
            raise ValueError(f"Incomparable evaluation protocols: {field}")
    aa = {row["episode"]: row for row in a["episodes_detail"]}
    bb = {row["episode"]: row for row in b["episodes_detail"]}
    if (
        len(aa) != len(a["episodes_detail"])
        or len(bb) != len(b["episodes_detail"])
        or set(aa) != set(bb)
        or not aa
    ):
        raise ValueError("Episode IDs must be unique, nonempty and identical")
    if len(aa) != a["evaluation_protocol"]["episodes"]:
        raise ValueError("Episode count disagrees with protocol")
    for episode in aa:
        for field in [
            "seed",
            "start_rel_pos",
            "start_target_vel",
            "start_distance",
            "target_mode",
        ]:
            if (
                field not in aa[episode]
                or field not in bb[episode]
                or not np.array_equal(aa[episode][field], bb[episode][field])
            ):
                raise ValueError(f"Episode {episode} initial state mismatch: {field}")
    both = sum(bool(aa[k]["success"]) and bool(bb[k]["success"]) for k in aa)
    a_only = sum(bool(aa[k]["success"]) and not bool(bb[k]["success"]) for k in aa)
    b_only = sum(bool(bb[k]["success"]) and not bool(aa[k]["success"]) for k in aa)
    return {
        "episodes": len(aa),
        "a_only": a_only,
        "b_only": b_only,
        "both_success": both,
        "both_fail": len(aa) - both - a_only - b_only,
        "exact_mcnemar_p": exact_mcnemar_p(a_only, b_only),
    }
