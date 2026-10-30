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
    c["task"].update(
        distance=[0.0, 0.0], initial_velocity_noise=0.0, initial_tilt_deg=0.0
    )
    return c
