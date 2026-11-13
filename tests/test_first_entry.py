import pytest
import torch

from pursuit_rl.ctbr.config import validate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.evaluation import evaluate, score
from pursuit_rl.ctbr.selection import passed


def entry_config(config):
    config["task"].update(objective="first_entry", radius=0.5, distance=[1, 2])
    return config


def test_safe_entry_terminates_before_tail(small):
    env = CTBRBatch(entry_config(small), n=1)
    u, t = env.plant.pursuer, env.plant.target
    u.pos.copy_(t.pos)
    u.pos[:, 0] += 0.499
    _, reward, terminated, truncated, records = env.step(torch.zeros(1, 4))
    assert terminated.item() and not truncated.item() and reward.item() > 0
    assert records[0]["success"] and not records[0]["tail_complete"]
    assert records[0]["entry_time_s"] == pytest.approx(env.dt)
    assert records[0]["entry_relative_speed_mps"] == pytest.approx(0)


def test_unsafe_entry_is_failure(small):
    env = CTBRBatch(entry_config(small), n=1)
    u, t = env.plant.pursuer, env.plant.target
    u.pos.copy_(t.pos)
    u.rot[0] = torch.diag(torch.tensor([1., -1., -1.]))
    _, reward, terminated, _, records = env.step(torch.zeros(1, 4))
    assert terminated.item() and reward.item() < 0
    assert records[0]["reason"] == "tilt"
    assert records[0]["entry_time_s"] is None


def test_timeout_and_entry_validation_metrics(small):
    config = entry_config(small)
    report, _ = evaluate(None, config, 2, 42, reference=True)
    assert report["failures"] == {"timeout": 2}
    assert report["mean_entry_time_s"] is None
    assert report["precision_passed"] is None
    assert score(report)[:2] == (0., 1.)
    assert not passed(report, config)
    assert passed(dict(report, success_rate=1), config)
    config["task"]["distance"] = [0.5, 2]
    with pytest.raises(ValueError, match="strictly outside"):
        validate(config)
