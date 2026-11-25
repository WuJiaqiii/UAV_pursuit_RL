"""Compare legacy 1-kHz terminal CPU sync with 100-Hz batched terminal sync.

The action sequence, seeds, CTBR dynamics and final reason counts are matched.
This measures the environment only (no PPO optimizer, validation or filesystem).
Never infer end-to-end trainer speedup from this microbenchmark alone.
"""
import argparse
import copy
import json
import time
from collections import Counter

import torch

from pursuit_rl.ctbr.config import load, validate
from pursuit_rl.ctbr.environment import CTBRBatch


def bench(c, *, fast, decisions, warmup=2):
    config = copy.deepcopy(c)
    config["training"]["fast_terminal_records"] = fast
    stats = {}
    for phase, length in (("warmup", warmup), ("timed", decisions)):
        env = CTBRBatch(config, seed=9121)
        action = torch.zeros(env.n, 4, device=env.device)
        reason_counts = Counter()
        steps = 0
        if str(env.device).startswith("cuda"):
            torch.cuda.synchronize(env.device)
            torch.cuda.reset_peak_memory_stats(env.device)
        start = time.perf_counter()
        for _ in range(length):
            _, _, _, _, completed = env.step(action, action)
            steps += env.last_physics_steps
            reason_counts.update(record["reason"] for record in completed)
            if completed:
                env.reset(env.finished)
        if str(env.device).startswith("cuda"):
            torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - start
        stats[phase] = dict(
            wall_s=elapsed, physics_steps=steps,
            physics_steps_per_s=steps / max(elapsed, 1e-9),
            episodes=sum(reason_counts.values()),
            reasons=dict(reason_counts),
            peak_reserved_mib=(
                torch.cuda.max_memory_reserved(env.device) / (1024 * 1024)
                if str(env.device).startswith("cuda") else None
            ),
        )
    return stats["timed"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("static", "tail"), default="static")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--decisions", type=int, default=100)
    args = ap.parse_args(argv)
    if args.decisions < 1:
        ap.error("Expected positive decisions")
    c = load(
        "configs/first_entry_highspeed/" + (
            "01_static.json" if args.task == "static" else "02_low_tail.json"
        ), device=args.device,
    )
    c["num_envs"] = args.num_envs
    c["prediction"]["enabled"] = False
    c["safety"]["predictive_enabled"] = False
    c["safety"]["constraint_enabled"] = False
    c["reward"]["actuator_risk"] = 0
    c = validate(c)
    torch.set_num_threads(1)
    legacy = bench(c, fast=False, decisions=args.decisions)
    batched = bench(c, fast=True, decisions=args.decisions)
    result = dict(
        task=args.task, device=args.device, num_envs=args.num_envs,
        decisions=args.decisions, legacy=legacy, batched=batched,
        environment_speedup=(
            batched["physics_steps_per_s"] /
            max(legacy["physics_steps_per_s"], 1e-9)
        ),
        note="Environment-only microbenchmark; no claim about entire PPO training wall time.",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
