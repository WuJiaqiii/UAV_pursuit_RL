"""Numerical evidence must be reproducible without changing the training policy."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from pursuit_rl.config import load_config
from pursuit_rl.learning.buffers import (
    RolloutBuffer,
    TensorRolloutBuffer,
    diagnostic_options,
    distribution_summary,
)
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.training.trainer import Trainer


class TestNumericalDiagnostics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="ppo-diagnostics-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def cfg(self, backend="torch"):
        cfg = load_config("configs/basic_v67/smoke.yaml")
        cfg["device"] = "cpu"
        cfg["simulation_backend"] = backend
        cfg["training"].update(total_steps=32, eval_every=32)
        return cfg

    def test_finite_distribution_and_quantiles_do_not_consume_rng(self):
        values = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf])
        state = torch.get_rng_state().clone()
        a, b = distribution_summary(values), distribution_summary(torch.tensor(values))
        self.assertEqual(a, b)
        self.assertEqual(a["nonfinite_count"], 3)
        self.assertEqual(a["mean"], 2.0)
        self.assertEqual(a["p50"], 2.0)
        json.dumps(a, allow_nan=False)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        empty = distribution_summary(torch.tensor([float("nan")]))
        self.assertIsNone(empty["mean"])
        self.assertEqual(empty["nonfinite_count"], 1)

    def test_quantiles_are_bounded_and_numpy_torch_agree(self):
        x = np.arange(10003, dtype=np.float32)
        a, b = distribution_summary(x, 19), distribution_summary(torch.tensor(x), 19)
        self.assertEqual(a["quantile_sample_count"], 19)
        self.assertEqual(a["max"], 10002.0)
        for key in ("mean", "std", "p01", "p50", "p99"):
            self.assertAlmostEqual(a[key], b[key], places=7)

    def test_options_reject_unknown_and_invalid_values(self):
        for settings in (
            {"typo": 1},
            {"max_quantile_samples": 1},
            {"critic_loss_threshold": float("nan")},
            {"enabled": 1},
        ):
            with self.assertRaises(ValueError):
                diagnostic_options(settings)

    def test_gae_matches_full_episode_mc_and_external_truncation(self):
        a = RolloutBuffer(3, 2, 31, 43)
        b = TensorRolloutBuffer(3, 2, 31, 43, "cpu")
        for buffer in (a, b):
            tensor = isinstance(buffer.reward, torch.Tensor)

            def set_field(name, data):
                value = (
                    torch.as_tensor(data, dtype=getattr(buffer, name).dtype)
                    if tensor
                    else data
                )
                getattr(buffer, name)[:] = value

            set_field("reward", [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            set_field("values", [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
            set_field("next_values", [[6.0, 6.0], [7.0, 7.0], [100.0, 100.0]])
            set_field("terminated", [[False, False], [False, False], [True, False]])
            set_field("truncated", [[False, False], [False, False], [False, True]])
            buffer.compute_gae(0.9, 1.0)
        np.testing.assert_allclose(a.returns[:, 0], [5.23, 4.7, 3.0], atol=1e-5)
        np.testing.assert_allclose(a.returns[:, 1], [78.13, 85.7, 93.0], atol=1e-5)
        for key in ("delta", "bootstrap_values", "returns", "advantage"):
            np.testing.assert_allclose(
                getattr(a, key), getattr(b, key).numpy(), atol=1e-5
            )
        self.assertEqual(a.bootstrap_values[-1, 0], 0.0)
        self.assertEqual(a.bootstrap_values[-1, 1], 100.0)

    def test_gae_reset_stops_recursion_and_masks_unused_terminal_nan(self):
        for buffer in (
            RolloutBuffer(2, 1, 31, 43),
            TensorRolloutBuffer(2, 1, 31, 43, "cpu"),
        ):
            buffer.terminated[0, 0] = True
            buffer.next_values[0, 0] = float("nan")
            buffer.reward[0, 0] = 7.0
            buffer.reward[1, 0] = float("nan")
            buffer.compute_gae(0.9, 0.9)
            self.assertEqual(float(buffer.returns[0, 0]), 7.0)

    def test_update_guard_does_not_mutate_weights_on_nonfinite_input(self):
        cfg = self.cfg()["training"]
        model = AsymmetricPPO(31, 43, cfg)
        buffer = RolloutBuffer(2, 2, 31, 43)
        buffer.reward[0, 0] = float("nan")
        buffer.compute_gae(cfg["gamma"], cfg["gae_lambda"])
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite rollout"):
            model.update(buffer)
        self.assertTrue(
            all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        )
        self.assertEqual(model.actor_optimizer.state_dict()["state"], {})

    def test_update_is_identical_to_original_implementation_for_finite_inputs(self):
        path = os.environ.get("PURSUIT_ORIGINAL_PPO")
        if not path:
            self.skipTest("Original source comparison is an external audit check")
        spec = importlib.util.spec_from_file_location("original_ppo", path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        cfg = self.cfg()["training"]
        old = legacy.AsymmetricPPO(31, 43, cfg)
        new = AsymmetricPPO(31, 43, cfg)
        new.load_state_dict(old.state_dict())
        buffer = RolloutBuffer(8, 2, 31, 43)
        rng = np.random.default_rng(781)
        buffer.obs[:] = rng.normal(size=buffer.obs.shape)
        buffer.priv[:] = rng.normal(size=buffer.priv.shape)
        actions, raw, lp, value = old.sample(
            buffer.obs.reshape(-1, 31), buffer.priv.reshape(-1, 43)
        )
        buffer.raw[:] = raw.reshape(8, 2, 4)
        buffer.logprob[:] = lp.reshape(8, 2)
        buffer.values[:] = value.reshape(8, 2)
        buffer.reward[:] = rng.normal(size=(8, 2))
        buffer.next_values[:-1] = buffer.values[1:]
        buffer.terminated[-1] = True
        buffer.compute_gae(cfg["gamma"], cfg["gae_lambda"])
        state = torch.get_rng_state().clone()
        original = old.update(buffer)
        torch.set_rng_state(state)
        diagnosed = new.update(buffer)
        self.assertTrue(
            all(
                torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()
            )
        )
        for key in original:
            self.assertEqual(original[key], diagnosed[key], key)
        self.assertLess(diagnosed["diagnostics"]["pre_update_logprob_error_max"], 1e-5)

    def test_diagnostics_preserve_training_for_both_backends(self):
        for backend in ("numpy", "torch"):
            cfg = self.cfg(backend)
            off = Trainer(
                cfg, self.root / (backend + "_off"), diagnostics={"enabled": False}
            )
            off.exp_logger.use_tensorboard = False
            off.run()
            on = Trainer(cfg, self.root / (backend + "_on"))
            on.exp_logger.use_tensorboard = False
            on.run()
            self.assertTrue(
                all(
                    torch.equal(v, on.model.state_dict()[k])
                    for k, v in off.model.state_dict().items()
                ),
                backend,
            )
            rows = [
                json.loads(line)
                for line in (on.out / "diagnostics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["rollout"]["events"]["episode_start"]["count"], 2)
            self.assertIn("actual_thrust_and_rates", rows[0]["rollout"]["actions"])
            self.assertFalse((off.out / "diagnostics.jsonl").exists())

    def test_first_anomaly_contains_pre_update_optimizer_and_replayable_rollout(self):
        cfg = self.cfg()
        trainer = Trainer(
            cfg, self.root / "anomaly", diagnostics={"value_abs_threshold": 1e-10}
        )
        trainer.exp_logger.use_tensorboard = False
        initial = copy.deepcopy(trainer.model.state_dict())
        trainer.run()
        paths = list((trainer.out / "diagnostics").glob("first_anomaly_*.pt"))
        self.assertEqual(len(paths), 1)
        saved = torch.load(paths[0], map_location="cpu", weights_only=True)
        self.assertTrue(
            all(
                torch.equal(v, saved["pre_update"]["actor_critic"][k])
                for k, v in initial.items()
            )
        )
        self.assertEqual(saved["steps_before_update"], 0)
        self.assertIn("delta", saved["rollout"]["tensors"])
        self.assertIn("terminal", saved["rollout"]["reward_components"])
        replay = AsymmetricPPO(31, 43, cfg["training"])
        replay.load_state_dict(saved["pre_update"]["actor_critic"])
        replay.actor_optimizer.load_state_dict(saved["pre_update"]["actor_optimizer"])
        replay.critic_optimizer.load_state_dict(saved["pre_update"]["critic_optimizer"])
        buffer = TensorRolloutBuffer(
            8, 2, 31, 43, "cpu", diagnostics={"enabled": False}
        )
        for name, value in saved["rollout"]["tensors"].items():
            getattr(buffer, name).copy_(value)
        torch.set_rng_state(saved["pre_update"]["torch_rng"])
        replay.update(buffer, aux_pd_coef=saved["aux_pd_coef"])
        self.assertTrue(
            all(
                torch.equal(v, saved["post_update"][k])
                for k, v in replay.state_dict().items()
            )
        )
        resumed = Trainer(
            cfg,
            trainer.out,
            resume=trainer.ckpt / "latest.pt",
            diagnostics={"enabled": False},
        )
        resumed.exp_logger.close()

    def test_exception_also_saves_first_anomaly_and_strict_json(self):
        cfg = self.cfg()
        trainer = Trainer(cfg, self.root / "exception")
        trainer.exp_logger.use_tensorboard = False
        collect = trainer.collect

        def corrupt():
            buffer, records, saturation = collect()
            buffer.reward[0, 0] = float("nan")
            buffer.compute_gae(cfg["training"]["gamma"], cfg["training"]["gae_lambda"])
            return buffer, records, saturation

        trainer.collect = corrupt
        with self.assertRaises(FloatingPointError):
            trainer.run()
        paths = list((trainer.out / "diagnostics").glob("first_anomaly_*.pt"))
        self.assertEqual(len(paths), 1)
        saved = torch.load(paths[0], map_location="cpu", weights_only=True)
        self.assertEqual(saved["reasons"], ["update_exception"])
        summary = json.loads(paths[0].with_suffix(".json").read_text())
        self.assertEqual(summary["metrics"]["rollout"]["reward"]["nonfinite_count"], 1)

    def test_calibration_restores_training_rng_environment_and_weights(self):
        for backend in ("numpy", "torch"):
            trainer = Trainer(self.cfg(backend), self.root / ("calibrate_" + backend))
            trainer.exp_logger.use_tensorboard = False
            weights = copy.deepcopy(trainer.model.state_dict())
            rng = torch.get_rng_state().clone()
            env_rng = (
                trainer.env.gen.get_state().clone()
                if trainer.tensor_backend
                else copy.deepcopy(trainer.env.rng.bit_generator.state)
            )
            report = trainer.calibrate_critic(episodes=3)
            self.assertEqual(report["episodes"], 3)
            self.assertEqual(report["predicted_values"]["count"], 3)
            self.assertEqual(report["policy"], "frozen_stochastic")
            self.assertEqual(trainer.steps, 0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(
                all(
                    torch.equal(v, trainer.model.state_dict()[k])
                    for k, v in weights.items()
                )
            )
            if trainer.tensor_backend:
                self.assertTrue(torch.equal(env_rng, trainer.env.gen.get_state()))
            else:
                self.assertEqual(env_rng, trainer.env.rng.bit_generator.state)
            trainer.exp_logger.close()


if __name__ == "__main__":
    unittest.main()
