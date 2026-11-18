"""Summary reporting must exclude incomplete runs and mixed training results."""

import json
from scripts.summarize_long_experiments import summarize, wilson, main


def test_wilson_extremes_and_counts():
    assert wilson(0, 0) == [None, None]
    low, high = wilson(0, 100)
    assert low == 0 and 0 < high < .05
    low, high = wilson(100, 100)
    assert .95 < low < 1 and high == 1


def test_summary_uses_only_complete_independent_task_evaluations(tmp_path):
    matrix = {
        "schema": "pursuit-first-entry-reward-factorial-v2",
        "stage": "full",
        "results": [
            {
                "variant": "good", "status": "trained", "ppo_complete": True,
                "physics_steps": 40_000_000, "final_gate_passed": True,
                "final_evaluations": [
                    {"seed": 79042, "episodes": 20, "successes": 18,
                     "success_rate": .9},
                    {"seed": 179042, "episodes": 20, "successes": 16,
                     "success_rate": .8}
                ],
                "retention_evaluations": [
                    {"seed": 89042, "episodes": 20, "successes": 20,
                     "success_rate": 1.0}
                ],
                "cross_evaluations": [
                    {"profile": "03_low_head_on", "seed": 279042,
                     "episodes": 20, "successes": 2, "success_rate": .1}
                ],
            },
            {
                "variant": "good", "status": "failed", "ppo_complete": False,
                "final_evaluations": [
                    {"episodes": 20, "successes": 20, "success_rate": 1.0}
                ],
            },
            {
                "variant": "bad", "status": "incomplete", "ppo_complete": False,
                "physics_steps": 10_000_000
            },
        ],
    }
    result = summarize(matrix)
    good = next(x for x in result["variants"] if x["variant"] == "good")
    assert good["complete_runs"] == 1
    assert good["failed_runs"] == 1
    assert good["current_task"]["successes"] == 34
    assert good["current_task"]["episodes"] == 40
    assert good["current_task"]["pooled_success_rate"] == .85
    assert good["retained_static"]["pooled_success_rate"] == 1.
    assert good["cross_geometry"]["03_low_head_on"]["pooled_success_rate"] == .1
    bad = next(x for x in result["variants"] if x["variant"] == "bad")
    assert bad["current_task"] is None
    path = tmp_path / "ablation_matrix.json"
    path.write_text(json.dumps(matrix))
    main([str(path)])
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
