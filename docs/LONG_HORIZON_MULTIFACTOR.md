# v8 — Long-horizon multi-parameter campaign (low-tail)

## Purpose and relationship to the numerical fix

This campaign builds on the fixed PPO rollout value path in `experiment/v8-reward-ablation-bc-transfer`.
A previous experiment stored a per-step reward component in the buffer field reserved
for the Critic prediction, corrupting temporal-difference errors and causing exponential
value target growth. This branch already separates `value_pred` from `component`,
checks sampled value integrity in tests, and aborts with `first_anomaly.pt` when
rollout values, advantages or returns breach the magnitude guards.

**No full GPU training has been executed as part of preparing this campaign.**
All proposed settings are experimental hypotheses, not demonstrated convergence.

## Factor design

The earlier `03_low_tail_factorial.json` compares progress (3/6/12),
braking shaping (0/0.5) and entry-velocity shaping (0/0.5), with DAgger
controls. The new plan `04_low_tail_long_optimization.json` extends this
to 15 configurations addressing:

- Reward: progress 3 vs 6 vs 12; remove braking; remove entry velocity;
  remove both (the **hard 15 m/s entry-speed limit remains unchanged**)
- DAgger: none, 1×500, 2×500, 3×750, 4×1000 BC updates, aggregate
  behavior-policy states across rounds; 4-second rollouts for each round
- Actor learning rate: 1e-5, 3e-5, 6e-5
- Critic learning rate: 5e-5 vs 1e-4
- GAE advantage timescale: 1.5s, 3.0s, 5.0s

Only **one factor at a time** changes relative to the shared P6+BC2 baseline,
except the explicit removal of both speed-related shaping terms. This is
not a complete factorial for all learning rates and rewards. Quantify
interactions with the separate 03 factorial experiment if needed.

All variants share exactly the same target task (02_low_tail), 0.5 m
entry radius, 15 m/s entry relative-speed limit, 60 m/s pursuer speed cap,
2.5g thrust cap, body rate/tilt limits and base training environment.
The mixed rollout allocates approximately 80% of parallel environments
to the current moving target and 20% to the saved 01_static task. The
static task's reward/discount definitions are aligned with the current
variant, and task-level success counts are reported separately.

## Budget and repeatability

| Phase | Variants (default) | Seeds | PPO physics steps / run | Aggregate PPO steps |
|---|---:|---:|---:|---:|
| Screening | 15 | 42, 43 | 8M | 240M |
| Full | 5 selected | 42, 43, 44 | 40M | 600M |
| Extended | 2 selected | 42–46 | 80M | 800M |

The budget excludes BC interaction, Critic MC calibration, periodic validation,
final task evaluation, cross-geometry evaluation and retention checks.
The estimated aggregate PPO budget for all defaults is **1.64 billion physics
interactions**; this is **not** a wall-clock estimate. Run the stages in order,
and only launch Full/Extended after evaluating the preceding phase.

Validation during training uses 24–32 episodes at 2M–8M step intervals,
to control overhead. At each run's end the runner evaluates independent
seeds (96/128/256 episodes depending on stage), tests static retention
and performs out-of-distribution diagnostics on low_head_on and low_random
profiles. The latter are reported **separately**, not counted toward the
02_low_tail promotion gate.

A run must be labeled invalid if it experiences any numeric guard failure.
Do not zero, clip or serialize away NaN/Inf values and continue training.

## Suggested launch sequence

First verify the fixed code and teacher:

```bash
git fetch origin
git checkout experiment/v8-long-horizon-multifactor
git pull --ff-only
pip install -e ".[dev]"
pytest -q
python main.py evaluate --reference --reference-method pd \
  --config configs/first_entry_highspeed/02_low_tail.json \
  --episodes 128 --seed 78042 --out runs/v8long/pd_reference.json --device cuda
```

Dry-run the long plan (does not train):

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/04_low_tail_long_optimization.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/v8long/dry_run --stage screening --device cpu --dry-run
```

Run the limited screening on two matched seeds:

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/04_low_tail_long_optimization.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/v8long/screening --stage screening --device cuda
```

After screening, select Full candidates and run:

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/04_low_tail_long_optimization.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/v8long/full --stage full --device cuda \
  --variants reference_p6_bc2 p6_bc3_750 p6_no_braking p6_actor_lr_1e5 p12_bc2
```

Finally confirm the top two with 80M steps each × five seeds:

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/04_low_tail_long_optimization.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/v8long/extended --stage extended --device cuda \
  --variants reference_p6_bc2 p6_bc3_750
```

Use `--resume-existing` with the **same output directory, source
revision, plan, parent checkpoint and run configuration** after interruption.
The launcher compares source fingerprints and config fingerprints; it will
not silently resume after a code change. Stop and inspect
`first_anomaly.pt` if numeric guards trigger. Restart numerically
corrupted experiments from a clean parent, never from the corrupted PPO
checkpoint.

## What to compare

1. Independent safe-first-entry success rate for **02_low_tail**, by seed,
   with 95% Wilson binomial intervals; do not confuse mixed training
   aggregate success with moving-target success
2. Unsafe entry rate and physical failures; report time and speed at
   entry only on successful episodes
3. Minimum distance distribution; monitor near-terminal precision
4. DAgger per-round `initialization_bc.json`, `initialization_bc_round_*.json`,
   action prediction error and retained 01_static success
5. Value target mean/std, TD error, advantage/returns thresholds, Critic
   calibration and episode-level reward decomposition
6. Out-of-distribution low-head-on and low-random results separately from
   in-distribution promotion gates

Full/Extended seed comparisons should use matched seeds and report both
per-seed outcomes and aggregate confidence intervals. Do not select a single
seed's best success rate as the final result.

**Recommended go/no-go:** If short screening exhibits zero current-task
safe entries across both seeds for every candidate, prioritize improving
DAgger terminal state coverage before spending 600M+ physics steps on Full.
