import copy
import json
import pytest
import torch
from pursuit_rl.ctbr.checkpoint import new_model, load as load_model, save
from pursuit_rl.ctbr.training import imitation_data, fit_actor, fit_critic, train
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.errors import ExperimentGateError


def test_imitation_resets_transients_and_removes_shortcut_features(small, monkeypatch):
    small["training"]["bc_segment_seconds"] = 0.004
    original = CTBRBatch.reset
    calls = []

    def reset(self, mask=None):
        calls.append(mask is None)
        return original(self, mask)

    monkeypatch.setattr(CTBRBatch, "reset", reset)
    x, y, steps = imitation_data(small)
    assert steps == 30 and sum(calls) == 4
    assert torch.equal(x[:, 21:], torch.zeros_like(x[:, 21:]))
    assert len(x) == 16 and y.shape == (16, 4)


def test_normalization_fold_preserves_kinematic_policy(small):
    model = new_model(small)
    x = torch.randn(128, 66)
    clean = x.clone()
    clean[:, 21:] = 0
    with torch.no_grad():
        expected = model.actor(clean).clone()
    fit_actor(model, (x, torch.zeros(128, 4)), 0, 64)
    with torch.no_grad():
        actual = model.actor(x)
    assert torch.allclose(actual, expected, atol=2e-6)
    assert torch.equal(
        model.actor[0].weight[:, 21:], torch.zeros_like(model.actor[0].weight[:, 21:])
    )


def setup_longer(small):
    small["task"].update(duration=0.5, tail_seconds=0.25)
    small["training"].update(
        physics_steps=0,
        bc_updates=1,
        dagger_updates=1,
        dagger_rounds=1,
        critic_updates=0,
        bc_samples=8,
        bc_stride=2,
    )
    return small


def zero_actor(model):
    with torch.no_grad():
        for p in model.actor.parameters():
            p.zero_()


def test_degraded_dagger_candidate_is_rejected(small, tmp_path, monkeypatch):
    c = setup_longer(small)
    calls = []

    def fit(model, *args):
        zero_actor(model)
        calls.append(1)
        if len(calls) == 2:
            with torch.no_grad():
                model.actor[-1].bias[0] = 5
        return 0.0

    monkeypatch.setattr("pursuit_rl.ctbr.training.fit_actor", fit)
    result = train(c, tmp_path / "run")
    model, _, _ = load_model(tmp_path / "run/checkpoints/initial.pt")
    assert result["rejected_dagger_rounds"] == 1 and result["accepted_bc_updates"] == 1
    assert result["best_origin"] == "bc_initial"
    assert all(torch.count_nonzero(p) == 0 for p in model.actor.parameters())
    initial = json.loads((tmp_path / "run/initial_validation.json").read_text())
    assert initial["success_rate"] == 1.0


def test_unqualified_initialization_stops_before_ppo_and_resume_cannot_bypass(
    small, tmp_path, monkeypatch
):
    c = setup_longer(small)
    c["training"].update(physics_steps=64, dagger_rounds=0)

    def fit(model, *args):
        zero_actor(model)
        with torch.no_grad():
            model.actor[-1].bias[0] = 5
        return 0.0

    monkeypatch.setattr("pursuit_rl.ctbr.training.fit_actor", fit)
    root = tmp_path / "bad"
    with pytest.raises(ExperimentGateError, match="PPO was not started"):
        train(c, root)
    details = json.loads((root / "gate_failure.json").read_text())
    assert (
        details["success_rate"] == 0
        and details["event"] == "initialization_gate_failed"
    )
    _, _, state = load_model(root / "checkpoints/latest.pt")
    assert state["progress"]["ppo_updates"] == 0
    with pytest.raises(ExperimentGateError):
        train(c, root, resume=root / "checkpoints/latest.pt")


def test_no_tail_samples_are_null_not_perfect_tracking(small):
    small["physics"]["max_speed"] = 0.001
    small["task"]["initial_velocity_noise"] = 0.2
    model = new_model(small)
    zero_actor(model)
    report, _ = evaluate(model, small, 2, 87)
    assert report["success_rate"] == 0 and report["full_tail_episodes"] == 0
    assert report["mean_tail_position_rmse"] is None
    assert report["mean_tail_velocity_rmse"] is None
    assert report["observed_tail_episodes"] == 0
    json.dumps(report, allow_nan=False)


def test_previous_v690_checkpoint_keeps_exact_actor_and_receives_training_defaults(
    small, tmp_path
):
    model = new_model(small)
    old = copy.deepcopy(small)
    for key in (
        "bc_segment_seconds",
        "initialization_episodes",
        "initialization_success_floor",
    ):
        old["training"].pop(key)
    path = tmp_path / "old.pt"
    save(path, model, old, {}, "bc_initial")
    restored, c, _ = load_model(path)
    assert c["training"]["bc_segment_seconds"] == 0.5
    assert all(
        torch.equal(p, restored.state_dict()[k]) for k, p in model.state_dict().items()
    )


def test_cli_gate_failure_exits_without_python_traceback(tmp_path, monkeypatch, capsys):
    import main

    def failed(*args, **kwargs):
        raise ExperimentGateError(
            dict(
                event="curriculum_gate_failed",
                message="expected gate",
                report="gate_current.json",
            )
        )

    monkeypatch.setattr(main, "run", failed)
    with pytest.raises(SystemExit) as exc:
        main.main(["curriculum", "--out", str(tmp_path)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "curriculum_gate_failed" in error and "Traceback" not in error


def test_critic_fits_rare_initial_return(small):
    small["training"]["critic_lr"] = 0.01
    model = new_model(small)
    x = torch.zeros(10001, 110)
    x[:, 31] = 0.1
    x[0, 31] = 0.2
    y = torch.zeros(10001)
    y[0] = 2.0
    fit = fit_critic(model, (x, y), 500, 128)
    prediction = model.values_tensor(x[:2])
    assert fit["initial_state_samples"] == 1
    assert abs(float(prediction[0]) - 2) < 0.15 and abs(float(prediction[1])) < 0.1
