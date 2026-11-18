# v9 unattended four-task generalization campaign

## Scope

This branch extends v8 CTBR first-entry PPO to four concurrent target-motion
distributions and frozen old-policy distillation. No changes to the 4D CTBR
action, 1 kHz physics, 2.5g thrust, 60 m/s pursuer cap, 0.5 m entry sphere,
or 15 m/s entry-speed gate.

The parent checkpoint should be the best existing low-tail model
`p6_bc4_1000/seed_42/checkpoints/best_current.pt`. The separate static
checkpoint should be `01_static/seed_42/checkpoints/best_feasible.pt`.
The model is a direct Actor at inference; PN/PD are reference baselines and
DAgger labels, not a blended runtime flight controller.

## Staged training

The JSON plan at `configs/experiments/05_v9_generalization.json` defines:

| Phase | Variants | Seeds | PPO physics steps / run |
|---|---:|---:|---:|
| Pilot | 5 | 42, 43 | 8 million |
| Full | best 3 from pilot | 42, 43, 44 | 40 million |
| Confirmation | best 2 from full | 42–46 | 80 million |

The full run begins at the **same v8 parent checkpoint** for each
variant/seed. The stages compare independent replications; they do not
continue fine-tuning pilot checkpoints. The campaign ranks candidate
models first by worst-task independent success rate and then by four-task
macro average, using a 0.15 worst-task pilot threshold before Full and
0.35 after Full before Confirmation. If no candidate passes, no more
costly stages launch. Full physics totals are upper bounds for a completed
all-stage run, not run-time estimates.

Default mixture uses 20% static, 25% tail, 25% head-on and 30% random
target; 64 parallel environments use integer allocations 13/16/16/19.
All four task profiles share identical reward/physics/policy contracts.
Frozen old teachers are assigned only to static and tail slots. Their
deterministic action targets add Smooth-L1 actor loss; rollout actions,
PPO old log probabilities, Critic and CTBR output are not replaced.
Two/four-round DAgger variants collect labeled actor-visited states from
all four task distributions; initial and intermediate BC checkpoints
and validation reports are retained.

## One-command background launch

From the repository root, after `pip install -e ".[dev]"` and
`pytest -q`, run:

```bash
bash scripts/launch_v9.sh \
  runs/v8long/screening/p6_bc4_1000/seed_42/checkpoints/best_current.pt \
  runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  runs/v9_unattended
```

This launches a `nohup` process that survives SSH logout, records its PID
and appends to `runs/v9_unattended/campaign.log`. Inspect progress:

```bash
tail -f runs/v9_unattended/campaign.log
cat runs/v9_unattended/campaign_results.json
```

After a machine reboot, invoke the identical launcher command; it detects
the campaign manifest and supplies `--resume-existing`. If an incomplete
run has `checkpoints/latest.pt`, the trainer restores Actor, Critic,
optimizers and RNG; unfinished episodes restart. A run stopped before
the first PPO checkpoint cannot safely resume BC mid-round. On the next
`--resume-existing` run, the launcher **archives the entire unfinished
directory** as `seed_XX__precheckpoint_attempt_NN` and cleanly restarts
that one experiment from the original parent. The original outputs are
preserved without overwriting or deletion. A run containing a numerical
`first_anomaly.pt` is never automatically resumed or restarted. The SHA256 of the plan, parent and static teacher,
and the repository Python source fingerprint must match.

For a preflight that generates no GPU rollouts:

```bash
python scripts/run_generalization_campaign.py \
  --parent-checkpoint runs/v8long/screening/p6_bc4_1000/seed_42/checkpoints/best_current.pt \
  --static-checkpoint runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt \
  --out runs/v9_dry_run --stage pilot --variants joint_ppo \
  --seeds 42 --device cpu --dry-run
```

Use a **different directory** from the real campaign for dry-run.

## Matched controller comparison

At every final evaluation, the campaign tests the same 4 target tasks,
same independent seed and same physics using:
- `ppo_v9`: selected checkpoint, deterministic
- `ppo_v8`: frozen low-tail parent checkpoint, deterministic
- `pn_ctbr`: PN lateral guidance with longitudinal speed regulation
- `pd_ctbr`: geometric PD
- `pn_pd_ctbr`: PN–PD hybrid

`pn_ctbr` is **not a pure lateral-only textbook PN**, because the repo's
guidance law also includes longitudinal braking and target acceleration
feed-forward. Every evaluation asserts an identical initial-state digest
across the five methods. Identical digest ensures matched initial states,
**not** necessarily a pre-recorded identical full random target
trajectory. Use episode-level trajectory replay for a stronger
counterfactual paired comparison.

Reports include safe-entry fraction and per-task failures, unsafe
entry fraction, successful entry time and relative speed, minimum
distance and action saturation. The original report contains individual
episodes; pooled Wilson intervals supplement rather than replace
between-training-seed variability.

## Numerical and resource safety

Do not resume an experiment that emitted `first_anomaly.pt` after a
non-finite or oversize PPO value signal. Failed/incomplete runs are
listed in `campaign_results.json` and excluded from completed model
selection. CUDA OOM is recorded and re-raised. Training includes periodic
checkpointing, code fingerprints, reward consistency checks and fresh
Critic calibration on transferred model. The pilot/full/confirmation
long GPU campaigns have **not** been run in this development session.

## Held-out distribution-shift evaluation

Full and Confirmation also evaluate three **never-trained** profile distributions:
`holdout_tail_wider` (5–9 m/s and 10–25 m separation),
`holdout_head_faster` (6–10 m/s and 20–40 m separation), and
`holdout_random_stronger` (6–10 m/s target, 2 m/s² maneuvers and
12% randomization factor). Each is run against PPO-v8/v9, PN-CTBR,
PD-CTBR and PN–PD-CTBR with checked identical initial-state digests.
They are stored under `ood_comparison/` and reported separately as
`summary_by_variant.<variant>.heldout`. They are **not** used for
checkpoint selection or phase promotion. In-distribution validation
scores on the four training task distributions remain separate.

## Critic and checkpoint-storage behavior

Before multi-task PPO, the transferred Critic is recalibrated using Monte Carlo
returns collected from **all four** training scenarios rather than only the
primary tail scenario. Periodic tail-only MC corrections are disabled in v9,
since they would skew the shared Critic away from the mixed training
objective. PPO still performs its regular joint-task value updates.

`latest.pt` is saved at every completed PPO rollout for device-interruption
recovery. Numbered archival `update_XXXXX.pt` files are saved every 20 PPO
updates to reduce multi-run disk pressure. Never delete the latest or selected
best checkpoint while the campaign is active.

## Automatic comparison tables

At each finished stage, the launcher writes
`comparison_summary.csv` in the campaign root. Rows are split by
stage, variant, distribution, task and controller; metrics include
successful safe entries and Wilson intervals, success-conditioned entry
time/relative speed, unsafe-entry frequency, failures and difference
in success rate versus PN-CTBR. Different trained variants are never
pooled into a fictitious shared policy. To regenerate after a partial
run:

```bash
python scripts/summarize_v9_campaign.py runs/v9_unattended/campaign_results.json
```
