# Low-tail first-entry PPO: factorial reward and DAgger experiment (v2)

## Why this campaign is different

The previous reward-ablation campaign is not a valid reward comparison: the PD-BC variants were stopped by the 50% initialization gate before PPO, while the no-BC run did enter PPO. Its PPO Critic targets were also corrupted by a Python local-variable overwrite in the rollout collector. The corrected collector stores the sampled Critic prediction in `buffer.values`; the numerical guard now aborts on non-finite values and on configured excessive magnitudes before a bad target can be optimized.

The v2 campaign therefore starts from a clean run directory and the same compatible `01_static` parent checkpoint for every variant. It records BC initialization performance, but does not enforce the 50% gate by default. This is intentional: the experiment asks whether PPO can improve a weak-but-finite BC initialization. Use `--enforce-initialization-gate` only when the research question explicitly requires PPO to start above the threshold.

## Experimental factors

The screening plan is a complete (3\times2\times2) factorial:

- Progress weight: 3, 6, 12.
- Braking shaping: 0, 0.5.
- Entry-velocity shaping: 0, 0.5.

All 12 factorial variants use 500 total BC updates split over two DAgger rounds (250 updates per round). Each round collects states under the current Actor, labels them with the PD teacher, fits the Actor, and evaluates the same fixed initialization seed. The plan also includes two controls:

- `baseline_pd_bc_1round`: same baseline reward and 500 BC updates in one round.
- `baseline_no_bc`: same baseline reward, no BC updates.

The hard first-entry definition, 0.5 m entry radius, 15 m/s relative-speed limit, 2.5g thrust ceiling, 60 m/s pursuer speed limit, and physical termination constraints remain unchanged. Zeroing braking or entry-velocity shaping never relaxes the hard 15 m/s crossing constraint.

## Budget and accounting

| Stage | Seeds | PPO physics steps per run | Validation | Purpose |
|---|---:|---:|---:|---|
| Screening | 42 | 4,000,000 | 32 episodes every 1M steps | Rank all 14 configurations |
| Full | 42, 43, 44 | 20,000,000 | 32 episodes every 4M steps | Confirm selected configurations |

The default Full set contains the baseline two-round DAgger variant, one-round DAgger control, no-BC control, no-braking variant, and no-entry-velocity variant. If screening shows a progress=12 variant is materially stronger, replace one of the two reward-ablation variants with that candidate using `--variants`.

The screening campaign allocates 56 million PPO physics steps across 14 single-seed runs. The default Full campaign allocates 300 million PPO physics steps across five variants and three seeds. These are aggregate environment-interaction budgets, not wall-clock estimates; the runner executes runs sequentially to avoid competing GPU jobs.

The `physics_steps` budget is **PPO rollout physics interactions only**. The following are tracked separately and do not consume the PPO budget:

- DAgger/BC state-collection interactions;
- Monte-Carlo Critic warm-up and periodic calibration interactions;
- initial, periodic, final, and retention evaluation interactions.

Every run writes a `run_manifest.json` with commit SHA, source fingerprint, config fingerprint, parent checkpoint SHA256, seed, task mixture, PPO budget, BC update/round count, and gate setting. The matrix records completed PPO steps, PPO updates, local interaction totals, task counts, BC rounds, independent evaluation metrics, and whether the PPO budget and final gates actually completed.

The runner refuses to overwrite existing run directories. For an interrupted run, use `--resume-existing`; it resumes from `checkpoints/latest.pt` only when the config, parent checkpoint, and source fingerprint match. A failed numerical run remains marked failed and is not counted as a trained result.

## Before screening: run the matched PD reference

Use the same fixed validation seed as BC initialization:

```bash
python main.py evaluate --reference --reference-method pd \
  --config configs/first_entry_highspeed/02_low_tail.json \
  --episodes 128 --seed 78000 \
  --out runs/low_tail_factorial_v2/pd_reference_78000.json --device cuda
```

This establishes whether the PD teacher itself can solve the validation scenarios. It must not be conflated with Actor BC success.

## Screening

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/03_low_tail_factorial.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/low_tail_factorial_v2/screening \
  --stage screening --device cuda
```

The default run tests all 14 variants with seed 42. PPO is allowed to start even if BC success is below 50%; `initialization_bc.json` and per-round BC reports remain mandatory diagnostics. The no-BC control is the only variant without BC initialization.

## Full confirmation

After screening, inspect `ablation_matrix.json`, Critic diagnostics, and the independent current-task/retention evaluations. Do not promote a variant whose run is incomplete, has a numerical failure, or violates the hard safety gate.

The default Full set is:

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/03_low_tail_factorial.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/low_tail_factorial_v2/full \
  --stage full --device cuda
```

To include a progress=12 candidate selected by screening:

```bash
python scripts/run_reward_ablation.py \
  --plan configs/experiments/03_low_tail_factorial.json \
  --parent-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/low_tail_factorial_v2/full_progress12 \
  --stage full --variants p6_b05_ev05 p12_b05_ev05 baseline_no_bc p6_b0_ev05 \
  --device cuda
```

If the process is interrupted, rerun the same command with `--resume-existing`. Do not change code or config between interruption and resume; the source fingerprint is part of the resume contract.

## Decision criteria

1. **Numerical validity first:** all rollout fields, GAE targets, Critic predictions, PPO losses, and gradients must remain finite and below configured hard magnitude thresholds. A run failing this condition is invalid, not a poor-performing policy.
2. **BC transfer:** compare the same-seed PD reference, each DAgger-round validation report, and no-BC initial validation. Determine whether BC is limited by teacher capability, state coverage, or regression fit.
3. **Reward factors:** compare progress, braking, and entry-velocity main effects and interactions. Use the three full seeds rather than a single screening seed for conclusions.
4. **Task performance:** report safe-entry success, failure reasons, minimum distance, relative speed at entry, and time-to-entry.
5. **Retention:** static-task retention must be reported separately; a high aggregate mixed-task success rate cannot conceal zero success on the current moving-target task.
6. **Promotion:** require completed PPO budget, no numerical anomaly, zero unsafe entries, and the configured held-out success threshold. Screening alone is never a final claim.

The campaign is simulation research only. It does not establish real-flight performance or validate the rigid-body parameterization against flight logs.
