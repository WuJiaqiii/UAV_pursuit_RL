"""Check the CTBR foundational abilities of a frozen survey navigation Actor."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pursuit_rl.ctbr.checkpoint import load, write_json
from pursuit_rl.ctbr.config import interface, load as load_profile
from pursuit_rl.ctbr.evaluation import evaluate

PHASES = ("01_hover_small", "02_hover_recovery", "03_goto_hold")
SEEDS = (5690042, 5790042)
EPISODES = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, parent, _ = load(args.checkpoint, args.device)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    results = []
    for phase in PHASES:
        c = load_profile(ROOT / "configs" / "ctbr" / f"{phase}.json", args.device)
        if interface(c) != interface(parent):
            raise ValueError(f"Incompatible checkpoint interface: {phase}")
        seeds = []
        for seed in SEEDS:
            report, _ = evaluate(model, c, EPISODES, seed)
            report["checkpoint_sha256"] = digest
            write_json(args.out / phase / f"{seed}.json", report)
            qualified = round(report["joint_precision_success_rate"] * EPISODES)
            seeds.append(dict(
                seed=seed, qualified=qualified, episodes=EPISODES,
                mean_tail_position_rmse_m=report["mean_tail_position_rmse"],
                failures=report["failures"], passed=qualified >= 15,
            ))
        results.append(dict(phase=phase, passed=all(r["passed"] for r in seeds), seeds=seeds))
        write_json(args.out / "summary.json", dict(checkpoint_sha256=digest, phases=results))
        print(phase, [(r["qualified"], r["mean_tail_position_rmse_m"]) for r in seeds], flush=True)

    return 0 if all(row["passed"] for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
