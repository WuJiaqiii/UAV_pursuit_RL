import copy
import json
from pathlib import Path
import pytest
import torch

from pursuit_rl.ctbr.config import load, fingerprint, interface
from pursuit_rl.ctbr.checkpoint import (
    new_model,
    save,
    load as restore,
    migrate,
    LEGACY_SCHEMA,
)
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.training import train
from pursuit_rl.ctbr.curriculum import gate
from pursuit_rl.ctbr.experiments import run as experiment
from pursuit_rl.ctbr.evaluation import evaluate


def legacy_checkpoint(c, path):
    old = copy.deepcopy(c)
    old["policy"]["position_error_cap_m"] = 0.0
    model = new_model(old)
    save(
        path,
        model,
        old,
        dict(ppo_updates=16, cumulative_ppo_physics_steps=2099151),
        "ppo",
    )
    data = torch.load(path, weights_only=False)
    old["schema"] = "pursuit-ctbr-v3"
    old["policy"].pop("position_error_cap_m")
    old["task"].pop("maximum_tail_rmse")
    for key in ("prediction_updates_enabled", "minimum_success"):
        old["training"].pop(key)
    data.update(
        schema=LEGACY_SCHEMA,
        config=old,
        fingerprint=fingerprint(old),
        interface=interface(old),
    )
    data["model_config"].pop("position_error_cap_m")
    torch.save(data, path)
    return model


def test_cap_matches_frozen_intervention_without_clipping_critic(small):
    model = new_model(small)
    raw_model = new_model(
        dict(small, policy=dict(small["policy"], position_error_cap_m=0))
    )
    raw_model.load_state_dict(model.state_dict())
    env = CTBRBatch(small)
    obs, priv = env.observe()
    obs = obs.clone()
    priv = priv.clone()
    obs[:, :3] = torch.tensor([1.0, 0, 0])
    priv[:, :3] = obs[:, :3]
    clipped = obs.clone()
    clipped[:, :3] *= 5 / 30
    actual = model.sample_tensor(obs, priv, True)
    expected = raw_model.sample_tensor(clipped, priv, True)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[3], expected[3])
    assert torch.equal(model.critic_features(priv)[:, :3], priv[:, :3])
    assert torch.equal(obs[:, :3], torch.tensor([[1.0, 0, 0]]).expand(2, 3))
    mean, std = model.distribution_parameters(obs)
    assert torch.allclose(model.logprob(mean, std, actual[1]), actual[2])


def test_legacy_inference_preserves_unbounded_policy_and_requires_explicit_migration(
    small, tmp_path
):
    path = tmp_path / "v700.pt"
    original = legacy_checkpoint(small, path)
    restored, c, _ = restore(path)
    assert c["policy"]["position_error_cap_m"] == 0
    obs = torch.randn(3, 66)
    assert torch.equal(original.predict_tensor(obs), restored.predict_tensor(obs))
    with pytest.raises(ValueError, match="migrate"):
        restore(path, optimizers=True)


def test_migration_preserves_weights_resets_std_and_invalidates_old_critic(
    small, tmp_path
):
    path = tmp_path / "v700.pt"
    original = legacy_checkpoint(small, path)
    c = copy.deepcopy(small)
    c["policy"]["initial_std"] = 0.03
    migrated, receipt = migrate(path, c, tmp_path / "new.pt")
    assert all(
        torch.equal(v, migrated.actor.state_dict()[k])
        for k, v in original.actor.state_dict().items()
    )
    assert torch.allclose(migrated.log_std.exp(), torch.full((4,), 0.03))
    assert not migrated.actor_optimizer.state_dict()["state"]
    model, _, saved = restore(tmp_path / "new.pt")
    assert (
        not model.critic_policy_match and saved["progress"]["critic_needs_calibration"]
    )
    assert receipt["reset_std"] and receipt["source_schema"] == LEGACY_SCHEMA
    with pytest.raises(FileExistsError):
        migrate(path, c, tmp_path / "new.pt")
    c["physics"]["tau_rate"] = 0.2
    with pytest.raises(ValueError, match="Migration changes"):
        migrate(path, c, tmp_path / "bad.pt")


def test_precision_failure_short_circuits_retention(small, tmp_path, monkeypatch):
    calls = []

    def report(*args, **kwargs):
        calls.append(1)
        return (
            dict(
                seed=79042, success_rate=1.0, mean_tail_position_rmse=1.0, failures={}
            ),
            None,
        )

    monkeypatch.setattr("pursuit_rl.ctbr.curriculum.evaluate", report)
    anchor = dict(
        name="old", config=small, reference=[dict(seed=89042, tail_rmse=0.01)]
    )
    result = gate(
        None, small, [anchor], tmp_path, 2, 42, 0.95, {}, "fast", fail_fast=True
    )
    assert not result["passed"] and not result["complete"] and not result["retention"]
    assert len(calls) == 1


def test_promotion_does_not_overwrite_better_initial_checkpoint(
    small, tmp_path, monkeypatch
):
    import pursuit_rl.ctbr.training as training

    c = copy.deepcopy(small)
    c["training"]["bc_updates"] = 0
    calls = []
    original = training.evaluate

    def evaluated(*args, **kwargs):
        report, data = original(*args, **kwargs)
        if not kwargs.get("stochastic", False):
            calls.append(1)
            error = 0.01 if len(calls) == 1 else 0.2
            report["mean_tail_position_rmse"] = error
            for episode in report["episodes_detail"]:
                episode["tail_position_rmse"] = error
        return report, data

    monkeypatch.setattr(training, "evaluate", evaluated)
    root = tmp_path / "run"
    result = train(c, root, promotion_check=lambda m, c, p: p["ppo_updates"] >= 1)
    assert result["promotion_passed"] and result["best_score"][2] == pytest.approx(
        -0.01
    )
    initial, _, _ = restore(root / "checkpoints/initial.pt")
    best, _, state = restore(root / "checkpoints/best_current.pt")
    assert all(
        torch.equal(v, best.actor.state_dict()[k])
        for k, v in initial.actor.state_dict().items()
    )
    assert state["progress"]["ppo_updates"] == 0
    _, _, feasible = restore(root / "checkpoints/best_feasible.pt")
    assert feasible["progress"]["ppo_updates"] > 0


def test_qualified_initialization_can_promote_without_ppo(small, tmp_path):
    result = train(small, tmp_path / "run", promotion_check=lambda *args: True)
    assert result["promotion_passed"] and result["ppo_updates"] == 0
    assert result["best_feasible_origin"] in ("bc_initial", "bc_dagger_initial")
    assert (tmp_path / "run/checkpoints/best_feasible.pt").exists()


def test_experiment_abc_full_pipeline_and_restart(small, tmp_path):
    source = tmp_path / "v700.pt"
    legacy_checkpoint(small, source)
    c = copy.deepcopy(small)
    c["policy"]["decision_steps"] = 10
    c["training"].update(
        physics_steps=320,
        rollout_steps=8,
        validation_every=160,
        bc_updates=0,
        dagger_rounds=0,
        transfer_dagger_updates=0,
        prediction_updates_enabled=False,
    )
    c["task"][
        "maximum_tail_rmse"
    ] = 1e-8  # Failure to reach precision is a recorded result, not a runtime error.
    for name, std, budget in [("A", 0.07, 0), ("B", 0.07, 320), ("C", 0.03, 320)]:
        d = copy.deepcopy(c)
        d["policy"]["initial_std"] = std
        d["training"]["physics_steps"] = budget
        (tmp_path / (name + ".json")).write_text(json.dumps(d))
    for name in ("hover", "recent"):
        d = copy.deepcopy(c)
        d["task"]["name"] = name
        (tmp_path / (name + ".json")).write_text(json.dumps(d))
    plan = dict(
        schema="pursuit-ctbr-experiment-v1",
        variants={k: k + ".json" for k in "ABC"},
        evaluation_profiles=dict(near="A.json", hover="hover.json"),
        retention_tasks=["hover"],
        replay_profiles=dict(recent="recent.json", anchor="hover.json"),
        test_episodes=2,
        stochastic_episodes=2,
        validation_episodes=2,
        gate_seed_offsets=[97000, 197000],
        retention_seed_offsets=[87000, 187000],
        test_seed_offset=210000,
        training_validation_seed_offset=470000,
    )
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    # Migration correctly rejects the 1 kHz v700 source. Rebuild the fixture with matching decision timing.
    legacy_checkpoint(c, source)
    root = tmp_path / "experiment"
    result = experiment(p, source, root)
    assert result["42"]["A"]["local_ppo_updates"] == 0
    assert result["42"]["B"]["local_ppo_updates"] == 2
    assert result["42"]["C"]["local_ppo_updates"] == 2
    assert result["42"]["B"]["role"] == "best_current_not_promoted"
    assert result == experiment(p, source, root)
    a, _, _ = restore(root / "seed_42/initializations/A.pt")
    cc, _, _ = restore(root / "seed_42/initializations/C.pt")
    assert torch.allclose(a.log_std.exp(), torch.full((4,), 0.07))
    assert torch.allclose(cc.log_std.exp(), torch.full((4,), 0.03))
