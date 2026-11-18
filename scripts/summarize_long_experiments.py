"""Summarize matched first-entry campaigns without mixing training and test successes.

Only complete PPO runs with independent final evaluations contribute to the
success-rate estimates. BC-gated/numerically failed/incomplete runs remain
visible, but never enter the denominator. Cross-geometry diagnostics are
reported separately from the current-task promotion metric.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes, episodes, z=1.959963984540054):
    """Approximate 95% confidence interval for binary episode outcomes."""
    if episodes <= 0:
        return [None, None]
    p = successes / episodes
    denom = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denom
    half = (
        z * math.sqrt(
            p * (1 - p) / episodes + z * z / (4 * episodes * episodes)
        ) / denom
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == episodes else min(1.0, center + half)
    return [lower, upper]


def aggregate_evaluations(reports):
    """Return None rather than inventing missing counts from floating rates."""
    if not reports:
        return None
    counts = [
        (item.get("successes"), item.get("episodes"))
        for item in reports
    ]
    counts_available = all(
        isinstance(s, int) and isinstance(n, int) and 0 <= s <= n and n > 0
        for s, n in counts
    )
    per_seed = [item["success_rate"] for item in reports]
    mean = sum(per_seed) / len(per_seed)
    sd = math.sqrt(sum((x - mean) ** 2 for x in per_seed) / len(per_seed))
    result = {
        "evaluations": len(reports),
        "mean_evaluation_success_rate": mean,
        "sd_evaluation_success_rate": sd,
    }
    if counts_available:
        successes = sum(s for s, _ in counts)
        episodes = sum(n for _, n in counts)
        result.update(
            successes=successes, episodes=episodes,
            pooled_success_rate=successes / episodes,
            wilson_95=wilson(successes, episodes),
        )
    else:
        result.update(successes=None, episodes=None,
                      pooled_success_rate=None, wilson_95=[None, None])
    return result


def summarize(matrix):
    grouped = defaultdict(list)
    for row in matrix.get("results", []):
        grouped[row["variant"]].append(row)
    outputs = []
    for name in sorted(grouped):
        rows = grouped[name]
        valid = [
            row for row in rows
            if row.get("status") == "trained"
            and row.get("ppo_complete") is True
            and row.get("final_evaluations")
        ]
        final = [ev for row in valid for ev in row["final_evaluations"]]
        retention = [
            ev for row in valid for ev in row.get("retention_evaluations", [])
        ]
        cross = defaultdict(list)
        for row in valid:
            for ev in row.get("cross_evaluations", []):
                cross[ev["profile"]].append(ev)
        outputs.append({
            "variant": name,
            "total_runs": len(rows),
            "complete_runs": len(valid),
            "failed_runs": sum(row.get("status") == "failed" for row in rows),
            "bc_gate_failures": sum(
                row.get("status") == "bc_initialization_gate_failed" for row in rows
            ),
            "incomplete_runs": sum(
                row.get("status") == "incomplete" for row in rows
            ),
            "dry_runs": sum(row.get("status") == "dry_run" for row in rows),
            "completed_ppo_steps": sum(row.get("physics_steps", 0) for row in valid),
            "final_gate_passes": sum(
                row.get("final_gate_passed") is True for row in valid
            ),
            "current_task": aggregate_evaluations(final),
            "retained_static": aggregate_evaluations(retention),
            "cross_geometry": {
                key: aggregate_evaluations(value)
                for key, value in sorted(cross.items())
            },
        })
    # Order is deterministic, but no single-seed or incomplete experiment
    # is ranked as definitive performance evidence.
    return {
        "schema": "pursuit-first-entry-ablation-summary-v1",
        "source_matrix_schema": matrix.get("schema"),
        "stage": matrix.get("stage"),
        "source_revision": matrix.get("commit_sha"),
        "warning": (
            "Wilson intervals cover pooled episode outcomes and do not replace "
            "between-seed uncertainty; compare individual matched seeds as well. "
            "Training mixed-task success is excluded."
        ),
        "variants": outputs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text(encoding="utf8"))
    report = summarize(matrix)
    out = args.out if args.out is not None else args.matrix.parent / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )
    csv_path = out.with_suffix(".csv")
    fields = [
        "variant", "total_runs", "complete_runs", "failed_runs",
        "bc_gate_failures", "incomplete_runs", "completed_ppo_steps",
        "final_gate_passes", "current_success_rate", "current_episodes",
        "current_ci_lower", "current_ci_upper", "retention_success_rate",
    ]
    with csv_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for v in report["variants"]:
            current = v["current_task"] or {}
            retention = v["retained_static"] or {}
            ci = current.get("wilson_95", [None, None])
            writer.writerow({
                **{k: v[k] for k in fields[:8]},
                "current_success_rate": current.get("pooled_success_rate"),
                "current_episodes": current.get("episodes"),
                "current_ci_lower": ci[0],
                "current_ci_upper": ci[1],
                "retention_success_rate": retention.get("pooled_success_rate"),
            })
    print(f"Summary: {out}\nTable: {csv_path}")
    return report


if __name__ == "__main__":
    main()
