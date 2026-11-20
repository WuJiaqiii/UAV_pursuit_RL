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
