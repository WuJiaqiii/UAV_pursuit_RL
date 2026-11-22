"""Independent V11 critical-attitude recovery test: learned Actor vs observable PD.

Both methods restart in *identical* tilted/rotating initial states and target
random-number streams, with no run-time teacher mixing. Training curriculum
sampling and reward shaping are explicitly OFF in this evaluation. Records
termination reasons and actual maximum tilt, not predicted surrogate cost.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, write_json
from pursuit_rl.ctbr.config import load as load_profile, validate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.control import teacher


@torch.no_grad()
def rollout(model, c, total, batch_size, seed, critical_fraction, reference):
    reports = []
    digests = []
    for start in range(0, total, batch_size):
        n = min(batch_size, total - start)
        # A separate environment with an exactly replayed seed/initial reset
        # is used for each controller; no shared mutable plant state.
        env = CTBRBatch(c, n=n, seed=seed + start)
        env.critical_reset_fraction = critical_fraction
        env.reset()
        signature = torch.cat((
            env.initial_rel_pos,
            env.plant.pursuer.rot.flatten(1),
            env.plant.pursuer.rate,
            env.plant.pursuer.thrust[:, None],
        ), 1).cpu().numpy().tobytes()
        digests.append(hashlib.sha256(signature).hexdigest())
        obs, priv = env.observe()
        for _ in range(env.max_decision_steps + 1):
            if reference:
                action = teacher(env)
                raw = torch.atanh(action.clamp(-0.999999, 0.999999))
            else:
                action = model.predict_tensor(obs)
                raw = torch.atanh(action.clamp(-0.999999, 0.999999))
            (obs, priv), _, _, _, completed = env.step(action, raw)
            reports.extend(dict(
                episode=start + item["slot"],
                reason=item["reason"],
                success=item["success"],
                maximum_tilt_deg=item["maximum_tilt_deg"],
                time_s=item["time"],
            ) for item in completed)
            if bool(env.finished.all()):
                break
        else:
            raise RuntimeError("Incomplete recovery probe batch")
    reports.sort(key=lambda r:r["episode"])
    if len(reports) != total:
        raise RuntimeError("Missing recovery-probe episodes")
    return reports, digests


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile",
                        default="configs/experiments/holdout_random_stronger.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--critical-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=510001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if (args.episodes <= 0 or args.batch_size <= 0
            or not 0 <= args.critical_fraction <= 1):
        raise ValueError("Invalid sample count or critical fraction")
    torch.set_num_threads(1)
    model, base, saved = load_checkpoint(args.checkpoint, args.device)
    c = copy.deepcopy(base)
    c["task"] = copy.deepcopy(load_profile(args.profile, device=args.device)["task"])
    c["num_envs"] = min(args.episodes, args.batch_size)
    # No reward shaping is needed for evaluating success; disabling expensive
    # risk prediction prevents V11 training costs contaminating evaluation.
    c["safety"]["predictive_enabled"] = False
    c["safety"]["constraint_enabled"] = False
    c["reward"]["actuator_risk"] = 0.0
    c["training"]["teacher_method"] = "pd_observed"
    c = validate(c)
    policy, policy_digest = rollout(
        model, c, args.episodes, args.batch_size, args.seed,
        args.critical_fraction, reference=False,
    )
    pd, pd_digest = rollout(
        model, c, args.episodes, args.batch_size, args.seed,
        args.critical_fraction, reference=True,
    )
    if policy_digest != pd_digest:
        raise RuntimeError("PPO and PD initial conditions are not identical")
    def metrics(records):
        from collections import Counter
        reasons = Counter(x["reason"] for x in records)
        angles = torch.tensor([x["maximum_tilt_deg"] for x in records])
        return dict(
            success_rate=reasons["success"] / len(records),
            tilt_failure_rate=reasons["tilt"] / len(records),
            timeout_rate=reasons["timeout"] / len(records),
            failures=dict(reasons),
            maximum_tilt_p95_deg=float(torch.quantile(angles, 0.95)),
        )
    paired = dict(
        pd_recovers_ppo_tilt_failure=sum(
            p["reason"] == "tilt" and d["success"]
            for p, d in zip(policy, pd)
        ),
        ppo_recovers_pd_tilt_failure=sum(
            d["reason"] == "tilt" and p["success"]
            for p, d in zip(policy, pd)
        ),
        paired_initial_state_hashes=policy_digest,
    )
    result = dict(
        schema="v11-critical-recovery-probe-v1",
        source_checkpoint=str(Path(args.checkpoint).resolve()),
        seed=args.seed,
        task=c["task"]["name"],
        episodes=args.episodes,
        critical_fraction=args.critical_fraction,
        critical_tilt_range_deg=[
            c["safety"]["critical_tilt_min_deg"],
            c["safety"]["critical_tilt_max_deg"],
        ],
        policy=metrics(policy),
        pd_observed=metrics(pd),
        paired=paired,
        limitation="Recovery success under this simulator and perturbation family; no formal invariance guarantee.",
    )
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
