"""Reproducible V11 safety-forecast numerical-fidelity audit.

This is NOT a guarantee against actual closed-loop future violations: both
predictions assume the SAME one-decision CTBR command followed by a bounded
level-recovery controller. A 1 ms reference checks integration error against
the 10 ms training predictor. Out-of-distribution flight safety requires
independent policy rollout tests with actual dynamics and uncertainty.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch

from pursuit_rl.ctbr.checkpoint import load as load_checkpoint, write_json
from pursuit_rl.ctbr.config import load as load_profile, validate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.safety import predict_attitude_risk


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--profile", default="configs/first_entry_highspeed/04_low_random.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--critical-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=450001)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.episodes < 1 or args.batch_size < 1 or not 0 <= args.critical_fraction <= 1:
        raise ValueError("Invalid sample count or critical fraction")
    torch.set_num_threads(1)
    model, parent, checkpoint = load_checkpoint(args.checkpoint, args.device)
    c = copy.deepcopy(parent)
    profile = load_profile(args.profile, device=args.device)
    c["task"] = copy.deepcopy(profile["task"])
    c["safety"]["predictive_enabled"] = True
    c["num_envs"] = min(args.batch_size, args.episodes)
    c = validate(c)
    angles, risks, fine_angles, fine_risks = [], [], [], []
    for start in range(0, args.episodes, args.batch_size):
        count = min(args.batch_size, args.episodes - start)
        env = CTBRBatch(c, n=count, seed=args.seed + start)
        env.critical_reset_fraction = args.critical_fraction
        env.reset()
        obs, _ = env.observe()
        action = model.predict_tensor(obs)
        coarse_risk, coarse_peak, _ = predict_attitude_risk(env, action)
        fine_risk, fine_peak, _ = predict_attitude_risk(
            env, action, step_s=c["physics"]["dt"],
        )
        angles.extend((coarse_peak * (180.0 / math.pi)).cpu().tolist())
        fine_angles.extend((fine_peak * (180.0 / math.pi)).cpu().tolist())
        risks.extend(coarse_risk.cpu().tolist())
        fine_risks.extend(fine_risk.cpu().tolist())
    a = torch.tensor(angles)
    f = torch.tensor(fine_angles)
    e = (a - f).abs()
    r = torch.tensor(risks)
    fr = torch.tensor(fine_risks)
    limit = float(c["physics"]["max_tilt_deg"])
    is_bad = f >= limit
    predicted = a >= limit
    output = dict(
        schema="v11-safety-predictor-fidelity-v1",
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_source_fingerprint=checkpoint.get("source_fingerprint"),
        task=c["task"]["name"],
        seed=args.seed,
        samples=len(e),
        critical_initial_fraction=args.critical_fraction,
        predicted_rollout_assumption="first command for 10ms, bounded level-rate recovery thereafter",
        training_integrator_step_s=c["safety"]["prediction_step_s"],
        reference_integrator_step_s=c["physics"]["dt"],
        tilt_error_mean_deg=float(e.mean()),
        tilt_error_p95_deg=float(torch.quantile(e, 0.95)),
        tilt_error_max_deg=float(e.max()),
        risk_error_mean=float((r - fr).abs().mean()),
        reference_70deg_breaches=int(is_bad.sum()),
        reference_70deg_recall=(
            float((predicted & is_bad).sum() / is_bad.sum())
            if bool(is_bad.any()) else None
        ),
        limitation="Fidelity to matched 1ms recovery-proxy, NOT to future closed-loop Actor trajectories or real flight safety.",
    )
    write_json(args.out, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    run()
