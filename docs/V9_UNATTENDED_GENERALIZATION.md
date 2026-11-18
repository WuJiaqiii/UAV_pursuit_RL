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
the first PPO checkpoint cannot safely resume: it is reported as such
and must be restarted in a new clean directory. No output is silently
deleted or overwritten. The SHA256 of the plan, parent and static teacher,
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
