"""Independent high-speed stress profiles, controller parity and resumability."""
import json
from pathlib import Path

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import new_model, save
from pursuit_rl.ctbr.config import load
from scripts import run_highspeed_stress as stress


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(31)
    torch.set_num_threads(1)


def plan():
    return json.loads(
        Path("configs/experiments/06_v10_pd_assisted_ctbr.json").read_text()
    )


def base():
    c = load("configs/first_entry_highspeed/01_static.json", device="cpu")
    c["prediction"]["enabled"] = False
    return c


def test_legacy_medium_high_extreme_tasks_preserve_policy_contract():
    p = plan()
    assert [len(p["stress_groups"][key]) for key in ("medium", "high", "extreme")] == [3, 3, 3]
    assert set(p["profiles"]) == {
        "01_static", "02_low_tail", "03_low_head_on", "04_low_random"
    }
    assert not (set(p["profiles"]) & set(p["stress_profiles"]))
    groups = stress.stress_configs(base(), p, ["medium", "high", "extreme"], 42, "cpu")
    assert sum(len(tasks) for tasks in groups.values()) == 9
    assert all(c["task"]["radius"] == 0.5 for tasks in groups.values() for c in tasks)
    assert all(c["task"]["entry_relative_speed_max_mps"] == 15 for tasks in groups.values() for c in tasks)
    assert all(c["num_envs"] == p["num_envs"] for tasks in groups.values() for c in tasks)
    assert [c["task"]["speed"] for c in groups["medium"]] == [[15, 20]] * 3
    assert [c["task"]["speed"] for c in groups["high"]] == [[40, 45]] * 3
    assert [c["task"]["speed"] for c in groups["extreme"]] == [[45, 50]] * 3


def test_unknown_group_and_incompatible_entry_contract_are_rejected():
    p = plan()
    with pytest.raises(KeyError):
        stress.stress_configs(base(), p, ["unregistered"], 42, "cpu")
    broken = base()
    broken["task"]["radius"] = 0.65
    with pytest.raises(ValueError, match="first-entry contract"):
        stress.stress_configs(broken, p, ["medium"], 42, "cpu")


def test_stress_independent_result_and_resume(tmp_path, monkeypatch):
    c = base()
    model_path = tmp_path / "checkpoint.pt"
    save(model_path, new_model(c, optimizers=False), c,
         {"physics_steps": 0, "ppo_updates": 0}, "fake_parent")
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "campaign_manifest.json").write_text(json.dumps({
        "parent_checkpoint": str(model_path)
    }))
    campaign = {
        "stages": {
            "confirmation": {
                "variants": {
                    "dagger4_guard_pd_aux": [
                        {"seed": 42, "status": "completed",
                         "checkpoint": str(model_path)}
                    ]
                }
            }
        }
    }
    (root / "campaign_results.json").write_text(json.dumps(campaign))
    calls = []

    def mock_evaluate(model, parent, configs, episodes, seed, dest, methods=None):
        calls.append((len(configs), episodes, tuple(methods)))
        result = {}
        for config in configs:
            result[config["task"]["name"]] = {
                method: {
                    "episodes": episodes, "successes": episodes,
                    "success_rate": 1.0, "failures": {},
                    "unsafe_entry_rate": 0.0,
                }
                for method in methods
            }
        return result

    monkeypatch.setattr(stress, "evaluate_methods", mock_evaluate)
    argv = [
        "--campaign-results", str(root / "campaign_results.json"),
        "--plan", "configs/experiments/06_v10_pd_assisted_ctbr.json",
        "--groups", "medium",
        "--variants", "dagger4_guard_pd_aux",
        "--methods", "ppo_v9", "pd_observed",
        "--episodes", "2", "--device", "cpu",
    ]
    dry = stress.run(argv + ["--dry-run"])
    assert len(dry["planned"]["selected"]) == 1
    assert not (root / "highspeed_stress").exists()
    result = stress.run(argv)
    assert result["summary_by_variant"]["dagger4_guard_pd_aux"]["medium"]["05_medium_tail"]["ppo_v9"]["success_rate"] == 1.0
    csv_path = root / "highspeed_stress/stress_summary.csv"
    assert csv_path.exists()
    assert csv_path.read_text().count("05_medium_tail") == 2
    assert calls == [(3, 2, ("ppo_v9", "pd_observed"))]
    again = stress.run(argv + ["--resume-existing"])
    assert again["summary_by_variant"] == result["summary_by_variant"]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="manifest changed"):
        stress.run(argv + ["--resume-existing", "--episodes", "3"])
