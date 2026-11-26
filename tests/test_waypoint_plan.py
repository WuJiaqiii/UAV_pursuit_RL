import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pursuit_rl.waypoint_plan import DEFAULT_PLAN, load_plan
from scripts.run_waypoint_experiments import main


SAMPLE = Path(__file__).resolve().parents[1] / "configs/experiments/virtual_waypoint.json"


class WaypointPlanTests(unittest.TestCase):
    def write_plan(self, document):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "plan.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_default_and_temporary_cli_overrides_preserve_old_command(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            plan = main(["--dry-run"])
        self.assertEqual(plan, DEFAULT_PLAN)
        self.assertEqual(json.loads(output.getvalue())["task"]["max_speed_mps"], 10.0)

        with contextlib.redirect_stdout(io.StringIO()):
            plan = main(["--dry-run", "--arms", "zero", "pd", "--envs", "2",
                         "--episodes", "3", "--episode-seconds", "0.04"])
        self.assertEqual(plan.experiment.arms, ("zero", "pd"))
        self.assertEqual(plan.experiment.episodes, 3)
        self.assertEqual(plan.ppo.envs, 2)
        self.assertEqual(plan.ppo.rollout_steps, 128)
        self.assertEqual(plan.task.timeout_s, 0.04)

    def test_stage_and_cli_override_shared_settings_in_order(self):
        plan, stage = load_plan(SAMPLE)
        self.assertEqual(stage, "standard")
        self.assertEqual(plan.task.horizontal_distance_m, (2.0, 5.0))
        self.assertEqual(plan.experiment.seeds, (42, 43, 44))

        with contextlib.redirect_stdout(io.StringIO()):
            near = main(["--plan", str(SAMPLE), "--stage", "near", "--dry-run",
                         "--device", "cpu", "--updates", "3"])
        self.assertEqual(near.task.horizontal_distance_m, (0.75, 1.5))
        self.assertEqual(near.experiment.device, "cpu")
        self.assertEqual(near.ppo.updates, 3)
        self.assertEqual(near.teacher, plan.teacher)
        wide, _ = load_plan(SAMPLE, "wide", {"task": {"timeout_s": 7.0}})
        self.assertEqual(wide.task.horizontal_distance_m, (5.0, 8.0))
        self.assertEqual(wide.task.timeout_s, 7.0)

    def test_resolved_plan_can_be_used_as_input(self):
        plan, _ = load_plan(SAMPLE, "wide")
        recovered, stage = load_plan(self.write_plan(plan.document()))
        self.assertEqual(recovered, plan)
        self.assertIsNone(stage)

    def test_errors_point_to_unknown_fields_and_invalid_stage_overrides(self):
        with self.assertRaisesRegex(ValueError, r"stages.near.ppo: updatez"):
            load_plan(self.write_plan({"stages": {"near": {"ppo": {"updatez": 2}}}}))
        with self.assertRaisesRegex(ValueError, r"Available stages: near, standard, wide"):
            load_plan(SAMPLE, "missing")
        with self.assertRaisesRegex(ValueError, r"task: Require 0 < reference speed"):
            load_plan(SAMPLE, overrides={"task": {"max_speed_mps": 11.0}})
        with self.assertRaisesRegex(ValueError, r"ppo: envs must be a positive integer"):
            load_plan(SAMPLE, overrides={"ppo": {"envs": True}})


if __name__ == "__main__":
    unittest.main()
