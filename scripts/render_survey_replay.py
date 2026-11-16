"""Render a survey comparison from existing NPZ trajectories, without simulating."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_random_survey import draw_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Directory containing the NPZ and report")
    parser.add_argument("--out", required=True, type=Path, help="Output file stem")
    args = parser.parse_args()

    with np.load(args.run / "trajectories.npz", allow_pickle=False) as archive:
        time, geometry_index, actor_index = np.intersect1d(
            archive["geometry_time"], archive["actor_time"], return_indices=True
        )
        geometry = archive["geometry"][geometry_index]
        actor = archive["actor"][actor_index]
    rows = json.loads((args.run / "report.json").read_text())["actor"]["episodes"]
    if len(time) < 2 or len(rows) != actor.shape[1] or geometry.shape != actor.shape:
        raise ValueError("Trajectory data and evaluation report do not agree")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    slot = draw_pair(args.out, time, geometry, actor, rows)
    print(f"Rendered scene {slot}: {args.out.with_suffix('.gif')}", flush=True)


if __name__ == "__main__":
    main()
