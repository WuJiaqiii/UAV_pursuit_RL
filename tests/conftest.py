import torch
import pytest
from pursuit_rl.ctbr.config import load


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def small():
    c = load("configs/ctbr/smoke.json")
    # Legacy one-tick boundary tests remain useful; dedicated v700 tests below
    # exercise the default 100 Hz + predictor + continuous-MC training route.
    c["policy"]["decision_steps"] = 1
    c["prediction"]["enabled"] = False
    c["training"].update(
        physics_steps=256,
        rollout_steps=64,
        batch_size=64,
        bc_samples=16,
        bc_updates=4,
        critic_updates=2,
        validation_every=128,
        mc_calibration_every_updates=0,
    )
    c["task"].update(
        duration=0.12,
        tail_seconds=0.06,
        distance=[0.0, 0.0],
        initial_velocity_noise=0.0,
        initial_tilt_deg=0.0,
    )
    return c
