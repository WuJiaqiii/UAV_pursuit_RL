import importlib.util, json, shutil, hashlib
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "curriculum_cli", ROOT / "scripts/curriculum.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_setup(tmp_path):
    parent = tmp_path / "baseline/seed_42/checkpoints/best.pt"
    parent.parent.mkdir(parents=True)
    torch.save({"progress": {"physics_steps": 20000000}}, parent)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "pursuit-curriculum-plan-v1",
                "phases": [
                    {"name": "one", "profile": "one.json"},
                    {"name": "two", "profile": "two.json"},
                ],
            }
        )
    )
    return parent, plan


def fake_runner(calls, fail_regression=False):
    def execute(args, log):
        calls.append(args)
        out = Path(args[args.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if "train" in args:
            parent = Path(args[args.index("--init-from") + 1])
            checkpoint = out / "checkpoints"
            checkpoint.mkdir(exist_ok=True)
            torch.save(
                {
                    "progress": {
                        "physics_steps": 0,
                        "parent_checkpoint": {
                            "sha256": hashlib.sha256(parent.read_bytes()).hexdigest()
                        },
                    }
                },
                checkpoint / "latest.pt",
            )
            shutil.copy2(checkpoint / "latest.pt", checkpoint / "best.pt")
            (out / "training_summary.json").write_text(
                json.dumps({"physics_steps": 0, "ppo_updates": 0})
            )
            (out / "initial_validation.json").write_text(
                json.dumps({"success_rate": 1.0})
            )
        else:
            bad = fail_regression and out.name == "baseline_regression"
            case = {
                "learned": {
                    "success_rate": 0.5 if bad else 1.0,
                    "mean_capture_time_timeout_penalty": 4.0,
                }
            }
            (out / "summary.json").write_text(json.dumps({"cases": [case]}))

    return execute


def test_two_phase_flow_uses_previous_selected_model(tmp_path, monkeypatch):
    parent, plan = make_setup(tmp_path)
    m = module()
    calls = []
    monkeypatch.setattr(m, "execute", fake_runner(calls))
    root = tmp_path / "result"
    assert (
        m.main(
            [
                "--plan",
                str(plan),
                "--baseline-root",
                str(parent.parents[2]),
                "--out",
                str(root),
                "--device",
                "cpu",
                "--physics-steps",
                "0",
            ]
        )
        == 0
    )
    train = [c for c in calls if "train" in c]
    assert len(train) == 2
    assert Path(train[0][train[0].index("--init-from") + 1]) == parent
    assert (
        Path(train[1][train[1].index("--init-from") + 1])
        == root / "one/seed_42/checkpoints/best.pt"
    )
    assert (
        len(json.loads((root / "curriculum_summary.json").read_text())["phases"]) == 2
    )


def test_regression_failure_prevents_next_phase(tmp_path, monkeypatch):
    parent, plan = make_setup(tmp_path)
    m = module()
    calls = []
    monkeypatch.setattr(m, "execute", fake_runner(calls, True))
    with pytest.raises(RuntimeError, match="Original-task regression"):
        m.main(
            [
                "--plan",
                str(plan),
                "--baseline-root",
                str(parent.parents[2]),
                "--out",
                str(tmp_path / "result"),
                "--device",
                "cpu",
                "--physics-steps",
                "0",
            ]
        )
    assert len([c for c in calls if "train" in c]) == 1


def test_missing_matching_seed_is_not_replaced_with_seed42(tmp_path, monkeypatch):
    parent, plan = make_setup(tmp_path)
    m = module()
    calls = []
    monkeypatch.setattr(m, "execute", fake_runner(calls))
    with pytest.raises(FileNotFoundError, match="matching-seed"):
        m.main(
            [
                "--plan",
                str(plan),
                "--baseline-root",
                str(parent.parents[2]),
                "--out",
                str(tmp_path / "result"),
                "--seeds",
                "43",
            ]
        )
    assert calls == []
