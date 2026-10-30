"""Regression for unindexed CUDA descriptors vs cuda:0 tensor devices."""

import pytest
import torch

from pursuit_rl.simulation.torch_device import resolve_torch_device
from pursuit_rl.config import load_config
from pursuit_rl.training.trainer import Trainer


def test_device_normalization_without_gpu(monkeypatch):
    # CI may have no GPU; simulated CUDA discovery catches the original defect.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    expected = torch.device("cuda:0")
    assert torch.device("cuda") != expected  # original V6.7.0 failure trigger
    assert resolve_torch_device("cuda") == expected
    assert resolve_torch_device(torch.device("cuda")) == expected
    assert resolve_torch_device("cuda:0") == expected
    assert resolve_torch_device("cpu") == torch.device("cpu")


def test_unavailable_cuda_reports_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="torch.cuda.is_available\(\) is False"):
        resolve_torch_device("cuda")


def test_tensor_sample_from_same_cpu_device(tmp_path):
    cfg = load_config("configs/basic_v67/smoke.yaml")
    cfg["device"] = "cpu"
    cfg["simulation_backend"] = "torch"
    trainer = Trainer(cfg, tmp_path / "cpu_zero_copy")
    try:
        assert trainer.model.device == trainer.env.device
        assert trainer.obs.device == next(trainer.model.actor.parameters()).device
        assert trainer.priv.device == next(trainer.model.critic.parameters()).device
        a, raw, lp, vals = trainer.model.sample_tensor(trainer.obs, trainer.priv)
        assert a.device == trainer.model.device
        assert raw.shape == (2, 4) and lp.shape == (2,) and vals.shape == (2,)
    finally:
        trainer.exp_logger.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA device")
def test_real_cuda_smoke_and_validation(tmp_path):
    # This is deliberately a real GPU smoke, not a mock: it guards CUDA:0
    # observation generation, PPO action sampling, validation and checkpoint.
    cfg = load_config("configs/basic_v67/smoke.yaml")
    cfg["device"] = "cuda"
    cfg["simulation_backend"] = "torch"
    trainer = Trainer(cfg, tmp_path / "cuda_smoke")
    assert trainer.model.device == torch.device("cuda", torch.cuda.current_device())
    assert trainer.env.device == trainer.model.device
    assert trainer.obs.device == trainer.model.device
    trainer.run()
    assert (tmp_path / "cuda_smoke" / "checkpoints" / "best_stage0.pt").exists()
