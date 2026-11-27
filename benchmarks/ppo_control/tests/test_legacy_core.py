"""Regression tests for scalar compatibility after removing duplicate code."""
import torch
from v15 import core as legacy
from v16 import core as current


def test_scalar_compatibility():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        old = legacy.Policy(4)
        torch.manual_seed(42)
        new = current.Policy(4, action_dim=1)
        assert old.state_dict().keys() == new.state_dict().keys()
        for key, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[key])
        x = torch.randn(8, 4)
        assert torch.equal(old(x), new(x))
        assert torch.equal(old.value(x), new.value(x))
        z = torch.randn(8, 1)
        assert torch.equal(old.logp(x, z), new.logp(x, z))
        new.load_state_dict(old.state_dict(), strict=True)
    assert legacy.gae is current.gae
    assert legacy.clipped_objective is current.clipped_objective
    assert legacy.explained_variance is current.explained_variance
