# v10: Direct CTBR with safe-state DAgger and observable-PD PPO supervision

The learned Actor **always outputs 4D CTBR directly**. The PD controller
provides demonstration targets, optional auxiliary on-policy PPO targets,
and independent baselines. PD is never blended into learned rollout actions
or invoked by the Actor at inference.

## Code changes

1. BC/DAgger keeps the **entire Actor observation** (including body rates,
   thrust, raw-action history and task context), rather than zeroing features
   after index 21.
2. BC minimizes the squared error of the **actually executed deterministic
   policy**, including the same raw-action autoregressive smoothing used by
   PPO. It uses frozen predictor features if present.
3. A configurable predictive soft tilt guard begins warning at 55 degrees
   and penalizes current and short-horizon inclination. The hard limit
   remains 70 degrees; this soft shaping cannot guarantee safe flight.
4. Optional `training.aux_pd_coef` adds on-policy teacher regression to
   PPO's post-filter deterministic action mean. Baseline default is 0.
   The teacher never overwrites behavior actions or on-policy log probabilities.

## PD baselines and observation parity

The existing `pd_ctbr` is a **privileged oracle baseline**: it accesses
actual randomized drag/mass and target acceleration from the simulator.

For fairer comparison the campaign additionally reports:

- `pd_observed`: pure PD using only the same ideal target/pursuer-state
  observation values available to the Actor, plus known nominal design
  constants. No Critic-only latents or randomized mass/drag values.
- `pd_observed_smoothed`: the above PD command with the same autoregressive
  raw-action filter as the Actor, for action-response parity.

Both PD references generate direct CTBR, pass through the same simulated
plant, and do not use RL. This removes privileged per-episode parameter
information, but PD still incorporates an explicit physics model. It does
not make the internal controller architectures identical.

All methods are checked for matching initial-state digests. With random
target maneuvers, this is **not equivalent to exact target-trajectory
replay**; trace-level replay is the next step toward strict paired testing.

## Matched ablation plan

`configs/experiments/06_v10_pd_assisted_ctbr.json` defines two arms:

| Training arm | On-policy PD auxiliary coefficient | DAgger rounds | BC updates | Tilt guard |
|---|---:|---:|---:|---:|
| `dagger4_guard_no_aux` | 0 | 4 | 1000 | enabled |
| `dagger4_guard_pd_aux` | 0.02 | 4 | 1000 | enabled |

Both use **observed-state PD** as their DAgger teacher and start independently
from the same v8 parent checkpoint. 64 parallel environments and the
4-task mix are unchanged. Pilot, Full and Confirmation use 8M, 40M and
80M physics steps per arm/seed respectively; both arms are retained for
matched causal comparison (`top_k=2` with zero promotion thresholds).

Example, from the repository root after installing the package:

```bash
python -m scripts.run_generalization_campaign \
  --plan configs/experiments/06_v10_pd_assisted_ctbr.json \
  --parent-checkpoint /path/to/v8_low_tail_best_current.pt \
  --static-checkpoint /path/to/v8_static_best_feasible.pt \
  --out runs/v10_pd_ablation --stage pilot --device cuda
```

For a full staged run, use `--stage all` and a fresh output directory.
Do **not** point to an existing v9 campaign directory. Resumption only
works under the same plan, checkpoints, config hashes and source fingerprint.

Check installation/regressions before large GPU runs:

```bash
python -m pip install -e ".[dev]"
pytest -q tests/test_v10_pd_assisted_ctbr.py tests/test_v9_generalization.py
```

This research environment uses perfect state observations and a simplified
closed-loop rigid-body model. Performance improvements and hard physical
safety are **not claimed** until independent experiments are completed.

## Medium, high and extreme speed independent stress tests (stages 05–13)

The v10 plan additionally registers the **original first-entry task definitions**
from `configs/first_entry_highspeed/05_*.json` through `13_*.json`:

| Group | Original stages | Target speed (m/s) | Encounter types |
|---|---|---|---|
| medium | 05–07 | 15–20 | tail, head-on, random |
| high | 08–10 | 40–45 | tail, head-on, random |
| extreme | 11–13 | 45–50 | tail, head-on, random |

Distances, pursuer initial speeds, random-target motion intensity and episode
deadlines are inherited from those exact original task profiles. **Only the
task portion is overlaid on the parent checkpoint's common physics, reward
and Actor contract**, so all comparisons use the same model interface,
2.5g thrust, 70-degree attitude limit and 15 m/s first-entry speed gate.

These stress tasks are **not** added to the four-task training mixture,
per-iteration validation, checkpoint selection or stage promotion; they
are an *independent post-training test set*. They must not be used to claim
training convergence at 40–50 m/s before evaluating the resulting models.

Run after the V10 experiment has completed its chosen stage:

```bash
# One-seed/one-variant screening, just medium speed, 16 episodes per task.
python -m scripts.run_highspeed_stress \
  --campaign-results runs/v10_pd_assisted_ctbr/campaign_results.json \
  --stage confirmation --groups medium --episodes 16 \
  --variants dagger4_guard_pd_aux --seeds 42 --device cuda \
  --out runs/v10_medium_screen

# Full three-band benchmark, both trained variants and all completed seeds.
# Default: 32 episodes per (task, controller, seed).
python -m scripts.run_highspeed_stress \
  --campaign-results runs/v10_pd_assisted_ctbr/campaign_results.json \
  --stage confirmation --groups medium high extreme --device cuda \
  --out runs/v10_highspeed_final
```

Alternatively, start a persistent background benchmark:

```bash
V10_STRESS_EPISODES=32 bash scripts/launch_v10_stress.sh \
  runs/v10_pd_assisted_ctbr medium high extreme
```

The stress evaluator writes `stress_manifest.json`, `stress_results.json`,
`stress_summary.csv`, and per-controller/per-task episode JSON files. It
supports `--resume-existing` with exactly the same selected checkpoints,
plan, source, groups, seeds, methods and episode count. A different workload
requires a fresh `--out` directory. A `--dry-run` validates the intended
selection and prints a manifest without launching evaluations.

Default controller set is `ppo_v9` (the newly trained V10 model under the
legacy report key), `pd_observed`, `pd_observed_smoothed` and the privileged
`pd_ctbr` oracle. `--methods` may also select `ppo_v8`, `pn_ctbr` and
`pn_pd_ctbr`. The CSV reports success-rate deltas against the observed
PD reference and the filtered observed PD reference.

**Interpretation caveats:** the historical medium/high/extreme configurations
may contain geometries that are not dynamically reachable under the 2.5g and
first-safe-entry constraints. A zero success rate, even for PD, does not
by itself prove the goal is infeasible. Evaluate reachability separately.
Matching initial-state digests do not establish identical stochastic target
trajectories at every step, so fully paired counterfactual conclusions require
trajectory replay.

Experimental configuration changes invalidate the previous V10 campaign
manifest. If training is already running, do not overwrite its plan/branch
mid-run; use an unchanged source snapshot or a fresh experiment directory.
