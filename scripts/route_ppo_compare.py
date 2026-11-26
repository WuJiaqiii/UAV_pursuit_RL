"""Run PPO ablations on fixed virtual survey routes."""

import argparse
import json
from pathlib import Path

from scripts.route_ppo_plan import (
    ARMS, MODES, PLAN, ROOT, IncompatibleRouteInput, check_route_source, load_plan,
    planned_updates,
)
from scripts.survey_route_profiles import PHASES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--mode", choices=MODES, default="pilot")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--dry-run", action="store_true", help="Show the resolved plan")
    preview.add_argument("--check-source", action="store_true", help="Verify source without Torch")
    parser.add_argument("--preflight", action="store_true", help="Evaluate only the BC baseline")
    parser.add_argument("--final-test", action="store_true", help="Test the validation selections")
    args = parser.parse_args(argv)

    if args.check_source:
        if args.preflight or args.final_test:
            parser.error("--check-source runs on its own")
        try:
            fingerprint = check_route_source(ROOT)
        except IncompatibleRouteInput as error:
            parser.error(str(error))
        print(f"Survey v4 source verified: {ROOT} ({fingerprint[:12]})")
        return
    if args.final_test and (args.mode != "confirm" or args.preflight):
        parser.error("--final-test requires --mode confirm and training")
    try:
        plan = load_plan(args.plan, args.mode)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({**plan, "arms": ARMS, "routes": PHASES,
                          "planned_updates": planned_updates(plan, args.preflight)}, indent=2))
        return
    if args.checkpoint is None or args.out is None:
        parser.error("--checkpoint and --out are required")
    try:
        source = check_route_source(ROOT)
        if not args.checkpoint.is_file():
            raise IncompatibleRouteInput(f"Checkpoint file not found: {args.checkpoint}")
        if args.out.exists():
            raise IncompatibleRouteInput(f"Choose a new output directory: {args.out}")
        from scripts import route_ppo_runtime
        from scripts.route_ppo_suite import run_suite

        run_suite(args, plan, source, route_ppo_runtime)
    except IncompatibleRouteInput as error:
        parser.error(str(error))
>>>>>>> origin/feat/route-ppo-ablation


if __name__ == "__main__":
    main()
