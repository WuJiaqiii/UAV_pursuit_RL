"""V12 pure-PPO CPU smoke tests and batched episode-record equivalence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pursuit_rl.ctbr.checkpoint import new_model, load as load_checkpoint
from pursuit_rl.ctbr.config import load, validate
from pursuit_rl.ctbr.environment import CTBRBatch
from scripts import run_pure_ppo


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(114)


def case(task="01_static", steps=10):
    c = load("configs/first_entry_highspeed/" + task + ".json", device="cpu")
    c["num_envs"] = 3
    c["prediction"]["enabled"] = False
    c["safety"]["predictive_enabled"] = False
    c["safety"]["constraint_enabled"] = False
    c["reward"]["actuator_risk"] = 0
    c["task"]["duration"] = 0.03
    c["task"]["distance"] = [1.0, 1.1]
    c["policy"]["decision_steps"] = steps
    return validate(c)


@pytest.mark.parametrize("task,steps", [
    ("01_static", 1), ("01_static", 10),
    ("02_low_tail", 10), ("04_low_random", 10),
])
def test_fast_episode_records_match_legacy(task, steps):
    base = case(task, steps)
    cslow = copy.deepcopy(base)
    cfast = copy.deepcopy(base)
    cfast["training"]["fast_terminal_records"] = True
    slow = CTBRBatch(cslow, seed=9001)
    fast = CTBRBatch(cfast, seed=9001)
    actions = torch.zeros(3, 4)
    for _ in range(4 if steps == 10 else 32):
        (os, cs), rs, ds, ts, records_s = slow.step(actions, actions)
        (of, cf), rf, df, tf, records_f = fast.step(actions, actions)
        assert torch.allclose(os, of, atol=1e-6)
        assert torch.allclose(cs, cf, atol=1e-6)
        assert torch.allclose(rs, rf, atol=1e-6)
        assert torch.equal(ds, df) and torch.equal(ts, tf)
        assert len(records_s) == len(records_f)
        ss = {v["slot"]: v for v in records_s}
        ff = {v["slot"]: v for v in records_f}
        assert set(ss) == set(ff)
        for slot in ss:
            for k, v in ss[slot].items():
                other = ff[slot][k]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert other == pytest.approx(v, abs=1e-5, rel=1e-5), (k, v, other)
                else:
                    assert other == v, (k, v, other)
        if bool(slow.finished.all()):
            break
    assert bool(slow.finished.all())


@pytest.mark.parametrize("speed,expected", [(10., "success"), (17., "unsafe_entry")])
def test_fast_records_preserve_first_entry_terminal_reason(speed, expected):
    cfg = case(steps=1)
    cfg["training"]["fast_terminal_records"] = True
    env = CTBRBatch(cfg, n=1, seed=10)
    u, t = env.plant.pursuer, env.plant.target
    u.pos.copy_(t.pos.float())
    u.pos[0, 0] += 0.505
    u.vel.zero_()
    u.vel[0, 0] = -speed
    _, _, done, _, records = env.step(torch.zeros(1, 4))
    assert done.item()
    assert records[0]["reason"] == expected


def test_v12_config_is_strictly_pure_ppo_and_minimal():
    args = SimpleNamespace(
        task="static", preset="quick", device="cpu", seed=19,
        num_envs=None, history_steps=None, batch_size=None,
        epochs=None, rollout_steps=None, physics_steps=None,
        legacy_terminal_records=False,
    )
    c = run_pure_ppo.make_config(args)
    model = new_model(c, optimizers=True)
    assert c["prediction"]["enabled"] is False
    assert c["training"]["prediction_updates_enabled"] is False
    assert c["training"]["bc_updates"] == 0
    assert c["training"]["aux_pd_coef"] == 0
    assert c["training"]["critic_updates"] == 0
    assert c["safety"]["constraint_enabled"] is False
    assert c["safety"]["predictive_enabled"] is False
    assert c["policy"]["history_steps"] == 1
    assert c["training"]["fast_terminal_records"] is True
    assert c["training"]["checkpoint_every_updates"] == 10
    assert model.predictor is None
    assert model.cost_critic is None
    assert model.cost_optimizer is None
    assert model.obs_dim == 38


def test_minimal_training_from_scratch_with_sparse_checkpoints(tmp_path, monkeypatch):
    c = case()
    c["num_envs"] = 2
    c["training"].update(
        physics_steps=100, rollout_steps=2, batch_size=4, epochs=1,
        mc_episodes=2, critic_updates=0, validation_every=100,
        validation_episodes=2, mc_calibration_every_updates=0,
        prediction_updates_enabled=False, fast_terminal_records=True,
        checkpoint_every_updates=10, save_every_updates=100,
    )
    c["training"]["bc_updates"] = 0
    c["policy"]["history_steps"] = 1
    c["training"]["aux_pd_coef"] = 0
    c = validate(c)
    from pursuit_rl.ctbr.training import train
    result = train(c, tmp_path / "fresh")
    assert result["ppo_updates"] > 0
    assert result["physics_steps"] >= 100
    assert result["training_wall_s"] >= 0
    metrics = [
        json.loads(x) for x in (tmp_path / "fresh/metrics.jsonl").read_text().splitlines()
        if '"event": "ppo_update"' in x
    ]
    assert metrics
    assert metrics[0]["rollout_wall_s"] >= 0
    assert metrics[0]["optimizer_wall_s"] >= 0
    checkpoint = tmp_path / "fresh/checkpoints/latest.pt"
    model, old, _ = load_checkpoint(checkpoint, device="cpu", optimizers=True)
    assert old["prediction"]["enabled"] is False
    assert model.predictor is None
