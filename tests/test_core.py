import unittest, math
import numpy as np
import torch
from pursuit_rl.config import load_config
from pursuit_rl.simulation.geometry import exp_so3, segment_first_hit, cpa
from pursuit_rl.simulation.environment import PursuitBatch, ACTOR_DIM, CRITIC_DIM
from pursuit_rl.learning.buffers import RolloutBuffer
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.control.guidance import pd_intercept


class TestPhysics(unittest.TestCase):
    def test_so3(self):
        rot = exp_so3(np.array([[0.3, -0.4, 0.8], [0, 0, 0]]))
        self.assertLess(
            np.max(np.abs(rot @ np.transpose(rot, (0, 2, 1)) - np.eye(3))), 1e-12
        )
        self.assertLess(abs(np.linalg.det(rot[0]) - 1), 1e-12)

    def test_hover(self):
        c = load_config()
        env = PursuitBatch(c, n=8, seed=12, randomize=False)
        p = env.pursuer
        p.vel[:] = 0
        acts = np.zeros((8, 4))
        initial = p.pos.copy()
        for _ in range(40):
            p.step(acts, c["physics"]["dt"])
        self.assertLess(np.max(np.abs(p.pos - initial)), 1e-9)

    def test_ned_thrust_and_body_rates(self):
        c = load_config()
        e = PursuitBatch(c, n=1, randomize=False)
        for _ in range(30):
            e.pursuer.step(np.array([[0, 1, 0, 0]]), 0.05)
        self.assertGreater(e.pursuer.rot[0, 2, 1], 0)

    def test_first_hit(self):
        hit, t = segment_first_hit(
            np.array([[10.0, 0, 0], [10, 2, 0]]),
            np.array([[-10.0, 0, 0], [-10, 2, 0]]),
            1.0,
        )
        self.assertTrue(hit[0])
        self.assertAlmostEqual(t[0], 0.45)
        self.assertFalse(hit[1])
        d, tc = cpa(np.array([[10.0, 5, 0]]), np.array([[-10, 0, 0]]), 5.0)
        self.assertAlmostEqual(d[0], 5.0)
        self.assertAlmostEqual(tc[0], 1.0)


class TestEnvironment(unittest.TestCase):
    def test_observations(self):
        e = PursuitBatch(load_config(), n=12)
        self.assertEqual(e.observe()[0].shape, (12, ACTOR_DIM))
        self.assertEqual(e.observe()[1].shape, (12, CRITIC_DIM))

    def test_targets(self):
        c = load_config()
        for stage in range(4):
            e = PursuitBatch(c, n=128, seed=stage + 30, stage=stage)
            for _ in range(100):
                e.target.step(0.05)
            self.assertLessEqual(
                float(np.max(np.linalg.norm(e.target.acc, axis=1))),
                c["target"]["max_accel"] + 1e-9,
            )
            self.assertTrue(np.isfinite(e.target.pos).all())
            self.assertLessEqual(
                float(np.max(e.target.max_observed_speed)), c["target"]["max_speed"] + 2
            )

    def test_terminal_timeout(self):
        c = load_config()
        c["curriculum"][0]["time"] = 0.10
        e = PursuitBatch(c, n=5, stage=0)
        _, _, _, tr, _ = e.step(np.zeros((5, 4)))
        self.assertFalse(tr.any())
        _, _, _, tr, info = e.step(np.zeros((5, 4)))
        self.assertTrue(tr.all())
        self.assertTrue(all(v["is_truncated"] for v in info))

    def test_no_auto_reset(self):
        e = PursuitBatch(load_config(), n=3)
        e.step(np.zeros((3, 4)))
        self.assertTrue(np.all(e.elapsed > 0))
        e.reset(np.array([1, 0, 0], bool))
        self.assertEqual(e.elapsed[0], 0)
        self.assertGreater(e.elapsed[1], 0)


class TestPPO(unittest.TestCase):
    def test_gae_timeout_bootstraps(self):
        b = RolloutBuffer(2, 1, 31, 43)
        b.reward[:] = 1
        b.values[:] = 2
        b.next_values[:] = 3
        b.truncated[0, 0] = True
        b.terminated[1, 0] = True
        b.compute_gae(0.9, 1)
        self.assertAlmostEqual(b.advantage[0, 0], 1 + 0.9 * 3 - 2, places=5)
        self.assertAlmostEqual(b.advantage[1, 0], 1 - 2, places=5)

    def test_actor_update(self):
        c = load_config()["training"]
        c["epochs"] = 1
        c["batch_size"] = 8
        model = AsymmetricPPO(31, 43, c)
        buf = RolloutBuffer(8, 2, 31, 43)
        obs = np.random.randn(16, 31).astype("float32")
        priv = np.random.randn(16, 43).astype("float32")
        acts, raw, lp, v = model.sample(obs, priv)
        buf.obs[:] = obs.reshape(8, 2, 31)
        buf.priv[:] = priv.reshape(8, 2, 43)
        buf.raw[:] = raw.reshape(8, 2, 4)
        buf.logprob[:] = lp.reshape(8, 2)
        buf.values[:] = v.reshape(8, 2)
        buf.reward[:] = 0.1
        buf.next_values[:] = 0.2
        buf.compute_gae(0.99, 0.95)
        before = model.actor[0].weight.detach().clone()
        stats = model.update(buf)
        self.assertGreater(stats["updates"], 0)
        self.assertFalse(torch.allclose(model.actor[0].weight, before))


if __name__ == "__main__":
    unittest.main()
