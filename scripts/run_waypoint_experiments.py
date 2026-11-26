"""Resolve and run the independent static virtual-waypoint experiments."""

import argparse
import json
from pathlib import Path

from pursuit_rl.waypoint_plan import ARMS, SECTIONS, load_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help="JSON experiment plan")
    parser.add_argument("--stage", help="Named difficulty stage in the plan")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved JSON without running")
    parser.add_argument("--out", type=Path, help="New output directory for this run")

    experiment = parser.add_argument_group("experiment overrides")
    experiment.add_argument("--device")
    experiment.add_argument("--arms", nargs="+", choices=ARMS)
    experiment.add_argument("--seeds", nargs="+", type=int)
    experiment.add_argument("--episodes", type=int)
    experiment.add_argument("--cpu-threads", type=int)

    task = parser.add_argument_group("waypoint overrides")
    task.add_argument("--distance", dest="horizontal_distance_m", type=float, nargs=2)
    task.add_argument("--radius", dest="radius_m", type=float)
    task.add_argument("--episode-seconds", dest="timeout_s", type=float)

    ppo = parser.add_argument_group("PPO budget overrides")
    ppo.add_argument("--envs", type=int)
    ppo.add_argument("--rollout-steps", type=int)
    ppo.add_argument("--updates", type=int)
    ppo.add_argument("--batch-size", type=int)

    teacher = parser.add_argument_group("teacher overrides")
    teacher.add_argument("--bc-steps", type=int)
    teacher.add_argument("--bc-epochs", type=int)
    teacher.add_argument("--aux-coef", type=float)

    args = vars(parser.parse_args(argv))
    overrides = {
        section: {name: args[name] for name in names if name in args}
        for section, names in SECTIONS.items()
    }
    try:
        plan, stage = load_plan(args.get("plan"), args.get("stage"), overrides)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.get("dry_run"):
        print(json.dumps(plan.document(), indent=2, allow_nan=False))
        return plan
    if "out" not in args:
        parser.error("--out is required unless --dry-run is used")

    from pursuit_rl.waypoint_experiments import run_suite

    try:
        return run_suite(plan, args["out"], stage)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
