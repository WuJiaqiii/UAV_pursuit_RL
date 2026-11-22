"""Generate cross-controller, per-task final-evaluation summary CSV for v9.

Never pools different trained policies as if one model. Failures and unfinished
runs remain in campaign_results.json and are not used for success estimates.
"""
import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "stage", "variant", "distribution", "task", "method",
    "successes", "episodes", "success_rate",
    "wilson_95_lower", "wilson_95_upper",
    "delta_success_vs_pn",
    "delta_success_vs_pd_observed",
    "delta_success_vs_pd_observed_smoothed",
    "unsafe_entry_rate", "tilt_failure_rate", "mean_maximum_tilt_deg",
    "near_guard_episode_fraction", "per_seed_maximum_tilt_p95_deg",
    "mean_entry_time_s",
    "mean_entry_relative_speed_mps", "mean_minimum_distance_m",
    "mean_action_saturation_fraction", "per_seed_success_rates",
    "failures",
]


def rows(results):
    for stage, stage_info in results.get("stages", {}).items():
        for name, categories in stage_info.get("summary_by_variant", {}).items():
            for distribution, tasks in categories.items():
                for task, methods in tasks.items():
                    pn = methods.get("pn_ctbr", {}).get("success_rate")
                    pd_obs = methods.get("pd_observed", {}).get("success_rate")
                    pd_obs_smooth = methods.get("pd_observed_smoothed", {}).get("success_rate")
                    for method, report in methods.items():
                        success = report["success_rate"]
                        yield dict(
                            stage=stage, variant=name,
                            distribution=distribution, task=task, method=method,
                            successes=report["successes"],
                            episodes=report["episodes"],
                            success_rate=success,
                            wilson_95_lower=report["wilson_95"][0],
                            wilson_95_upper=report["wilson_95"][1],
                            delta_success_vs_pn=(
                                success - pn if pn is not None else None
                            ),
                            delta_success_vs_pd_observed=(
                                success - pd_obs if pd_obs is not None else None
                            ),
                            delta_success_vs_pd_observed_smoothed=(
                                success - pd_obs_smooth
                                if pd_obs_smooth is not None else None
                            ),
                            unsafe_entry_rate=report.get("unsafe_entry_rate"),
                            tilt_failure_rate=report.get("tilt_failure_rate"),
                            mean_maximum_tilt_deg=report.get("mean_maximum_tilt_deg"),
                            near_guard_episode_fraction=report.get("near_guard_episode_fraction"),
                            per_seed_maximum_tilt_p95_deg=json.dumps(
                                report.get("per_seed_maximum_tilt_p95_deg", [])
                            ),
                            mean_entry_time_s=report.get("mean_entry_time_s"),
                            mean_entry_relative_speed_mps=report.get(
                                "mean_entry_relative_speed_mps"
                            ),
                            mean_minimum_distance_m=report.get(
                                "mean_minimum_distance_m"
                            ),
                            mean_action_saturation_fraction=report.get(
                                "mean_action_saturation_fraction"
                            ),
                            per_seed_success_rates=json.dumps(
                                report.get("per_seed_success_rates", [])
                            ),
                            failures=json.dumps(report.get("failures", {})),
                        )


def to_csv(results, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows(results))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign_results", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    out = a.out or a.campaign_results.with_name("comparison_summary.csv")
    return to_csv(json.loads(a.campaign_results.read_text()), out)


if __name__ == "__main__":
    main()
