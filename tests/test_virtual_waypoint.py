import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pursuit_rl.waypoint import OBS_DIM, Outcome, WaypointConfig, WaypointEnv, waypoint_pd
from pursuit_rl import waypoint_experiments as experiments
from scripts.run_waypoint_experiments import main


class VirtualWaypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def test_first_entry_freezes_completed_body_and_overspeed_takes_priority(self):
        env = WaypointEnv(WaypointConfig(), n=2, seed=13)
        self.assertEqual(env.observe().shape, (2, OBS_DIM))
        env.waypoint[:] = env.body.pos + torch.tensor([0.5005, 0.0, 0.0])
        env.body.vel[:, 0] = torch.tensor([1.0, 11.0])
        _, reward, done, elapsed = env.step(torch.zeros(2, 4))
        self.assertEqual(done.tolist(), [True, True])
        self.assertEqual(env.outcome.tolist(), [Outcome.GOAL_REACHED, Outcome.OVERSPEED])
        self.assertGreater(reward[0].item(), 1.0)
        self.assertLess(reward[1].item(), 0.0)
        self.assertTrue(torch.allclose(elapsed, torch.full((2,), env.dt)))
        position = env.body.pos.clone()
        _, reward, done, _ = env.step(torch.ones(2, 4))
        self.assertFalse(done.any())
        self.assertTrue((reward == 0).all())
        self.assertTrue(torch.equal(position, env.body.pos))

    def test_crossing_between_ticks_counts_with_both_endpoints_outside(self):
        config = WaypointConfig(radius_m=0.002, decision_ticks=1)
        env = WaypointEnv(config, n=1, seed=13)
        env.waypoint[:] = env.body.pos + torch.tensor([0.004, 0.0, 0.0])
        self.assertGreater(torch.dist(env.waypoint, env.body.pos), config.radius_m)
        env.body.vel[:, 0] = 8.0
        _, _, done, _ = env.step(torch.zeros(1, 4))
        self.assertTrue(done.item())
        self.assertEqual(env.outcome.item(), Outcome.GOAL_REACHED)
        self.assertGreater(torch.dist(env.waypoint, env.body.pos), config.radius_m)

    def test_hover_times_out_and_pd_reaches_same_independent_waypoints(self):
        config = WaypointConfig()
        zero = experiments.evaluate(config, experiments._zero, 8, "cpu")
        pd = experiments.evaluate(config, waypoint_pd, 8, "cpu")
        self.assertEqual(zero["outcomes"]["timeout"], 8)
        self.assertEqual(pd["outcomes"]["overspeed"], 0)
        self.assertEqual(pd["successes"], 8)
        self.assertEqual(
            [case["waypoint_ned_m"] for case in zero["cases"]],
            [case["waypoint_ned_m"] for case in pd["cases"]],
        )

    def test_suite_runs_ppo_bc_and_aux_updates_with_separate_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "suite"
            main([
                "--out", str(out), "--arms", "ppo", "bc_ppo", "aux_ppo",
                "--episodes", "2", "--envs", "2", "--rollout-steps", "4",
                "--updates", "1", "--batch-size", "8", "--bc-steps", "4",
                "--bc-epochs", "1", "--episode-seconds", "0.04",
            ])
            with (out / "summary.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                {(row["arm"], row["phase"]) for row in rows},
                {("ppo", "initial"), ("ppo", "final"), ("bc_ppo", "after_bc"),
                 ("bc_ppo", "final"), ("aux_ppo", "initial"), ("aux_ppo", "final")},
            )
            for arm in ("ppo", "bc_ppo", "aux_ppo"):
                directory = out / f"{arm}_seed42"
                metrics = json.loads((directory / "metrics.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(metrics["update"], 1)
                self.assertGreater(metrics["actor_minibatches"], 0)
                self.assertLess(metrics["diagnostics"]["pre_update_logprob_error_max"], 1e-4)
                self.assertEqual(metrics["aux_pd_coef"], 0.1 if arm == "aux_ppo" else 0.0)
                self.assertTrue((directory / "final.pt").is_file())
            progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["phase"], "completed")
            self.assertEqual(progress["completed_ppo_updates"], 3)
            self.assertTrue((out / "bc_ppo_seed42" / "bc.pt").is_file())


if __name__ == "__main__":
    unittest.main()
