import tempfile, unittest
from pathlib import Path
from pursuit_rl.config import load_config
from pursuit_rl.training.trainer import Trainer
from pursuit_rl.learning.checkpoints import load_model
from pursuit_rl.evaluation.rollout import evaluate


class TrainE2ETest(unittest.TestCase):
    def test_train_saves_and_loads(self):
        c = load_config("configs/smoke.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(c, tmp)
            s = trainer.run()
            self.assertGreaterEqual(s["steps"], c["training"]["total_steps"])
            self.assertTrue((Path(tmp) / "checkpoints/best.pt").exists())
            model, _ = load_model(Path(tmp) / "checkpoints/best.pt")
            result = evaluate(model, c, stage=0, episodes=2, seed=999)
            self.assertEqual(result["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
