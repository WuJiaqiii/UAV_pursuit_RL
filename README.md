# UAV_pursuit_RL — High-speed first safe entry (v8)

This repository now implements **one research objective**: a quadrotor policy
that issues four direct CTBR commands (mass-normalized collective thrust plus
three body-rate setpoints) and reaches the target's 3D sphere for the **first
time** before the deadline, while obeying physical limits and a closing-speed
constraint. Sustained-following, Survey and v7 compatibility code were removed.

## Flight/task envelope

| Property | Nominal setting |
| --- | --- |
| Quadrotor max horizontal/total speed | 60 m/s (norm of 3D velocity) |
| Target speed bands | 0 to 50 m/s across 13 stages |
| Maximum total specific thrust | 2.5g = 24.525 m/s² |
| First-entry sphere radius | 0.5 m |
| Relative speed at crossing | ≤ 15 m/s |
| Control / physical integration | 100 Hz / 1 kHz |
| Actor input | Current oracle target/pursuer state (no onboard estimation yet) |
| Teacher reference | PD, PN, PN+PD hybrid; **not blended** into PPO action |

**Research simulation only:** the selected drag, rate response and thrust
parameters are *not* validated flight-log estimates; 40–50 m/s training
convergence and successful real flight have **not** been demonstrated.

## Set up

```bash
python -m pip install -e ".[dev,plot]"
pytest -q
```

## Run

```bash
# Reference/geometry baselines; use fresh held-out seeds for final evaluation.
python main.py evaluate --reference --reference-method pn_hybrid \
  --config configs/first_entry_highspeed/01_static.json \
  --episodes 64 --out runs/reference_01.json --device cpu

# Single-stage PPO (optional PN imitation initialization in stage 01).
python main.py train --config configs/first_entry_highspeed/01_static.json \
  --out runs/stage_01 --device cuda

# Ordered 13-stage course; fails closed if promotion/retention gates fail.
python main.py curriculum --plan configs/first_entry_highspeed/plan.json \
  --out runs/course --device cuda --episodes 64 --minimum-success 0.8

# Compare teacher variants on *the same* scenario and seeds.
python main.py evaluate --reference --reference-method pn \
  --config configs/first_entry_highspeed/08_high_tail.json \
  --episodes 256 --seed 890001 --out runs/pn_08.json --device cuda

# Export episode traces and plot a representative trajectory.
python main.py evaluate --reference --reference-method pn_hybrid \
  --config configs/first_entry_highspeed/08_high_tail.json \
  --trace-dir runs/traces --episodes 2 --out runs/trace_report.json --device cpu
python scripts/plot_entry_trace.py runs/traces/trace_seed_42.npz \
  --slot 0 --out runs/trace_01.png
```

`--init-from` accepts a v8 first-entry checkpoint with identical action,
observation, physics, predictor and network contracts. Reward weights may differ
for controlled reward ablations; the Critic is recalibrated before PPO.
`--resume` additionally requires unchanged source and the original experiment directory.
**Older v7 and earlier checkpoints are intentionally unsupported.** Historical
source can be recovered from Git history, not the current working tree.

See [experiment design](docs/HIGH_SPEED_FIRST_ENTRY.md),
[low-tail reward/transfer-BC ablation plan](docs/LOW_TAIL_REWARD_ABLATION.md),
and [code architecture](docs/ARCHITECTURE.md).

For the matched low-tail reward campaign, use
`configs/experiments/03_low_tail_factorial.json` with
`scripts/run_reward_ablation.py`. It runs a 3×2×2 reward factorial, one-round
and two-round PD DAgger controls, and a no-BC control. BC initialization quality
is recorded but does not gate PPO by default; pass `--enforce-initialization-gate`
to require the configured floor. The runner writes per-run manifests and an
auditable `ablation_matrix.json`, and supports `--resume-existing` when the
checkpoint, config, and source fingerprint match. The older v1 plan remains
available for historical reproduction.

## Code map

- `main.py`: train / evaluate / curriculum CLI (no migrate command)
- `pursuit_rl/ctbr/`: first-entry environment, PN/PD teacher, training,
  curriculum, evaluation and strict checkpoint
- `pursuit_rl/simulation/`: parameterized NED rigid-body CTBR dynamics
- `pursuit_rl/learning/`: asymmetric PPO, rollout/GAE, LSTM predictor,
  ValueNorm
- `configs/first_entry_highspeed/`: 13 stage profiles and ordered plan
- `tests/`: environment, safety, training, curriculum and checkpoint regressions
- `scripts/plot_entry_trace.py`: trace visualization

## Long-horizon multi-parameter experiments

After verifying numerical integrity and completing the 4M/20M reward
screening, use the separate
[long-horizon plan](docs/LONG_HORIZON_MULTIFACTOR.md):
`configs/experiments/04_low_tail_long_optimization.json`.
It varies reward weights, DAgger rounds, Actor/Critic learning rates and
GAE advantage horizon, with matched 8M / 40M / 80M physics-step phases.
It reports independent low-head-on and low-random stress evaluations;
these are not included in the current-task promotion score.

The new plan is a proposed research campaign; the long GPU experiments
are not included in automated CI and have not been executed here.
