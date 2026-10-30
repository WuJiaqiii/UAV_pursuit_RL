"""Regression tests for the 40-second 1 kHz float32 evaluation clock.

The V6.7.1 bug was triggered by repeated float32 addition of .001. At 40,002
steps the accumulated clock was only ~39.99165 s, so nonterminal episodes
never reached the 40 s cutoff before the evaluator's horizon was exhausted.
"""
import torch
import pytest

from pursuit_rl.config import load_config
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.control.guidance import GuidanceController
import pursuit_rl.evaluation.torch_rollout as torch_rollout


def _standard_cpu_cfg():
    cfg = load_config("configs/basic_v67/s0_standard_pd02.yaml")
    cfg["device"] = "cpu"
    cfg["simulation_backend"] = "torch"
    cfg["training"]["num_envs"] = 2
    return cfg


def test_40_second_integer_step_clock_has_no_float32_cumulative_drift():
    cfg = _standard_cpu_cfg()
    dt = cfg["physics"]["dt"]
    assert dt == 0.001
    env = TorchPursuitBatch(cfg, n=2, seed=123, device="cpu")
    assert env.elapsed.dtype == torch.float32
    # 40k integrations *must* correspond to 40 seconds in 1 kHz evaluation.
    # This assertion fails for accumulated time (float32 drift ~8 ms).
    env.steps[:] = 40000
    env.elapsed = env.steps.to(env.dtype) * dt
    assert torch.equal(env.elapsed, torch.full((2,), 40.0))



def test_step_updates_elapsed_from_integer_steps_without_drift(monkeypatch):
    # This is the regression that fails on the V6.7.1 float32 += dt code.
    cfg = _standard_cpu_cfg()
    env = TorchPursuitBatch(cfg, n=2, seed=512, device="cpu")
    monkeypatch.setattr(env, "_pursuer_step", lambda action, dt: env.u.vel.clone())
    monkeypatch.setattr(env, "_target_step", lambda dt: None)
    # Some other integration tests set a large CPU thread count; limit the
    # tiny tensor operations in this regression to one thread for stable CI.
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(400):
            env.step(torch.zeros(2, 4))
    finally:
        torch.set_num_threads(old_threads)
    assert env.steps.tolist() == [400, 400]
    torch.testing.assert_close(
        env.elapsed,
        env.steps.to(env.dtype) * cfg["physics"]["dt"],
        rtol=0, atol=1e-7,
    )


def test_clock_and_terminal_mask_at_40s(monkeypatch):
    cfg = _standard_cpu_cfg()
    env = TorchPursuitBatch(cfg, n=2, seed=4321, device="cpu")
    # Keep pursuer and target stationary to force a *timeout*, not capture/fail.
    monkeypatch.setattr(env, "_pursuer_step", lambda action, dt: env.u.vel.clone())
    monkeypatch.setattr(env, "_target_step", lambda dt: None)
    env.steps[:] = 39997
    env.elapsed = env.steps.to(env.dtype) * cfg["physics"]["dt"]
    terminated = truncated = None
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(torch.zeros(2,4))
    assert env.elapsed.tolist() == [40.0,40.0]
    assert not terminated.any().item()
    assert truncated.tolist() == [True,True]


def test_standard_40_second_batch_evaluation_completes(monkeypatch):
    cfg = _standard_cpu_cfg()
    base = torch_rollout.TorchPursuitBatch

    class FastForwardBatch(base):
        """Exercise real evaluator close to 40 s without 40k costly CPU steps."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps[:] = 39997
            self.elapsed = self.steps.to(self.dtype) * self.cfg["physics"]["dt"]

        def _pursuer_step(self, action, dt):
            return self.u.vel.clone()

        def _target_step(self, dt):
            return None

    monkeypatch.setattr(torch_rollout, "TorchPursuitBatch", FastForwardBatch)
    report = evaluate(GuidanceController("PN"), cfg, stage=0, episodes=2, seed=888)
    assert report["episodes"] == 2
    assert report["failures"]["timeout"] == 2
    assert all(abs(r["time"] - 40.0) < 1e-6 for r in report["episodes_detail"])


def test_training_saves_recovery_checkpoint_before_evaluator_exception(tmp_path, monkeypatch):
    """The first scheduled validation must not discard all previously trained steps."""
    from pursuit_rl.training.trainer import Trainer
    import pursuit_rl.training.trainer as trainer_module

    cfg = load_config("configs/basic_v67/smoke.yaml")
    cfg["device"] = "cpu"
    cfg["simulation_backend"] = "torch"
    out = tmp_path / "failed_validation"

    def broken_evaluator(*args, **kwargs):
        raise RuntimeError("simulated evaluator crash")

    monkeypatch.setattr(trainer_module, "evaluate", broken_evaluator)
    trainer = Trainer(cfg, out)
    with pytest.raises(RuntimeError, match="simulated evaluator crash"):
        trainer.run()
    checkpoint = out / "checkpoints" / "latest.pt"
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["steps"] >= cfg["training"]["total_steps"]
    # A matching incomplete experiment may be resumed without losing weights.
    resumed = Trainer(cfg, out, resume=str(checkpoint))
    try:
        assert resumed.steps == saved["steps"]
    finally:
        resumed.exp_logger.close()
