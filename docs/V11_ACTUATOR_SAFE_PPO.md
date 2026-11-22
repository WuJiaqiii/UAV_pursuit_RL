# V11: Action-conditioned safety risk + constrained PPO + critical-state curriculum

Status: experimental / CPU-CI tested; no success-rate, runtime or 24 GB peak-VRAM
improvement is claimed until a fresh independent campaign finishes.

## Research question and causal ablation

Current V10 achieves very high low-speed ID success but has nonzero tilt
failures on low-random and out-of-distribution random / wider-tail tasks.
V11 deliberately preserves 4D normalized **direct CTBR** output, policy input,
PPO on-policy ratios, and the V10 4-round observable-PD DAgger initialization.

Four matched variants in
`configs/experiments/08_v11_actuator_safe_ppo.json`:

| Arm | Action-aware predicted risk reward | Cost Critic + PPO-Lag | Critical initial-state curriculum |
| --- | --- | --- | --- |
| `v10_control` | No | No | No |
| `v11a_predictive_reward` | Yes | No | No |
| `v11b_constrained_ppo` | Yes | Yes | No |
| `v11c_constrained_curriculum` | Yes | Yes | Yes |

All arms use the same PD teacher (observable state), DAgger data **budget**
(10,000 examples/round, 400 updates across 4 rounds), auxiliary PD coefficient
(0.02), old-policy distillation coefficient (0.05), 4-task mixture and seeds.
Critical curriculum changes the training initial-state distribution on purpose;
do not interpret it as matched training *trajectories*. Final evaluation seeds
and target tasks remain matched and independent of the curriculum.

## A. CTBR-action-conditioned risk predictor

`pursuit_rl/ctbr/safety.py` approximates the existing kinematic rigid-body
backend. For the first 10 ms, it applies the candidate command's body-rate
setpoint with exact first-order rate lag; subsequent 10 ms commands use a
bounded level-attitude recovery rate law:

```text
omega[k+1] = omega[k] + (omega_cmd[k]-omega[k])*(1-exp(-h/tau_rate))
R[k+1]     = R[k] * exp_so3(h*(omega[k]+omega[k+1])/2)
risk       = clamp((max_tilt + uncertainty - warning)/(limit-warning),0,1)^2
```

Training risk simulation uses h=10 ms for efficiency, while the plant advances
at 1 ms. `scripts/audit_v11_safety.py` compares 10 ms and 1 ms predictions
under the **same recovery policy assumption**. Because future Actor commands,
changing target dynamics and model mismatch may differ, numerical fidelity to
this reference does **not** guarantee future policy safety or estimate the
true viability kernel. The risk model reads privileged actuator-response
parameters during **training only**; Actor inference is unchanged.

A separate terminal attitude cost is recorded if the *actual* physics rollout
crosses 70 degrees. The new reward weight `reward.actuator_risk` is applied
only once per 100 Hz policy decision, unlike physics-rate shaping.

## B. Constrained on-policy PPO (V11-B/C)

A Cost Critic, with the same privileged observation features as the reward
Critic, estimates accumulated risk costs. Cost GAE respects true terminal
flags, and the Actor minibatch advantage is

```text
A_combined = standardize(A_reward) - lambda * (A_cost - mean(A_cost))
```

where `lambda` is a nonnegative, checkpointed dual variable. At each PPO
update:

```text
lambda <- clamp(lambda + lagrangian_lr*(mean(on-policy decision risk)
                                         - cost_limit), 0, lambda_max)
```

`cost_limit` is therefore a bound on *mean per-decision surrogate risk*,
**not** a formal bound on episode failure probability. The objective must be
validated against independent actual tilt failures and their uncertainty.
The Cost Critic is not used at deployment; no run-time PD or CBF safety filter
is required for the experimental Actor.

The V11 network adds a Cost Critic only for constrained arms. The reward
Actor/critic, forecast features and CTBR dimensions remain identical, and
checkpoint loading permits new Cost Critic / dual keys when explicitly
warm-starting from V8/V9/V10. The entire optimizer/dual state is saved atomically
in the standard checkpoint. Resume still requires identical source fingerprint
and experiment manifest; **never** resume an old V10 directory with V11 code.

## C. On-policy risk-aware curriculum (V11-C)

The V11-C training environment gradually increases the probability of a
nonlevel initial state up to 15% over the first 40% of its PPO physics budget.
Sampled perturbations begin at 35–50 degrees with a moderate outward body rate.
This augments the initial distribution, not off-policy replay: all PPO
transitions are collected from the current Actor. The normal 4-task mixture
is retained for anti-forgetting.

The curriculum is **disabled during heldout evaluation**: new CTBRBatch
evaluation environments initialize with a 0% critical-reset fraction. A
recovery-success oracle and recovery-region calibration are future
extensions; a perturbed starting state is not assumed to be recoverable.

## Campaign budget (24 GB GPU-friendly defaults, not verified peak usage)

| Stage | Seeds | Physics steps / individual run | Runs | Total |
| --- | --- | ---: | ---: | ---: |
| Pilot | 42,43 | 2,000,000 | 8 | 16,000,000 |
| Full | 42,43 | 8,000,000 | 8 | 64,000,000 |
| Confirmation | 42,43,44 | 30,000,000 | 12 | 360,000,000 |
| **Total** | | | **28** | **440,000,000** |

This excludes validation, imitation interactions, MC calibration, and
independent benchmarks. The predictor adds GPU computation; *24 GB-friendly*
describes storage/batch limits, not a guaranteed wall-clock speed or VRAM peak.

The independent holdouts remain `holdout_tail_wider`,
`holdout_head_faster`, `holdout_random_stronger`. The historical
05–13 medium/high/extreme profiles are not in PPO training.

## Prepare and dry-run

```bash
git fetch origin
git switch feat/v11-actuator-aware-safe-ppo
git pull --ff-only origin feat/v11-actuator-aware-safe-ppo
pip install -e ".[dev]"
pytest -q tests/test_v11_safety.py tests/test_v10_pd_assisted_ctbr.py

# Set these to the same audited V8 low-tail and V8 static parents
# for all four arms; never confuse these with the newly trained best.pt.
export PARENT=/absolute/path/to/v8_low_tail_best.pt
export STATIC=/absolute/path/to/v8_static_best.pt

python -m scripts.run_generalization_campaign \
    --plan configs/experiments/08_v11_actuator_safe_ppo.json \
    --parent-checkpoint "$PARENT" --static-checkpoint "$STATIC" \
    --stage pilot --device cuda --dry-run --out runs/v11_dryrun
```

## Run Pilot before committing to the complete campaign

```bash
python -m scripts.run_generalization_campaign \
    --plan configs/experiments/08_v11_actuator_safe_ppo.json \
    --parent-checkpoint "$PARENT" --static-checkpoint "$STATIC" \
    --stage pilot --device cuda --out runs/v11_actuator_safe_ppo
```

Inspect the resulting `campaign_results.json`, `comparison_summary.csv`,
per-run `metrics.jsonl`, plus `safety_cost_mean`,
`lagrange_multiplier`, `cost_critic_loss`, `completed_tilt_failure_rate`
and `curriculum_critical_reset_fraction` at each PPO update. If dual
multipliers remain exactly zero, do not assume constraints are active: inspect
whether observed mean cost exceeds `safety.cost_limit`. Validate risk
prediction numerical error before interpreting results.

Once the Pilot result and new tests are satisfactory, resume into the full
campaign without redoing already completed seeds:

```bash
bash scripts/launch_v11.sh "$PARENT" "$STATIC" runs/v11_actuator_safe_ppo
tail -f runs/v11_actuator_safe_ppo/campaign.log
```

To run historical high-speed OOD tests separately, *after* Confirmation:

```bash
V10_STRESS_EPISODES=32 bash scripts/launch_v11_stress.sh \
    runs/v11_actuator_safe_ppo medium high extreme
```

Use `medium` only for the first stress Pilot if GPU/time-limited.

## Safety model numerical-fidelity audit

```bash
python -m scripts.audit_v11_safety \
    --checkpoint runs/v11_actuator_safe_ppo/pilot/v11b_constrained_ppo/seed_42/checkpoints/best_current.pt \
    --profile configs/experiments/holdout_random_stronger.json \
    --episodes 512 --critical-fraction 0.5 --device cuda \
    --out runs/v11_actuator_safe_ppo/safety_audit.json
```

This compares model forecasts against a 1 ms numerical reference using the
**same** one-decision-then-recovery maneuver. It does *not* measure the
probability that closed-loop PPO under actual target maneuvers stays safe.

## Independent reporting and go/no-go criteria

For each arm, task and training seed, report at least:
- First-entry success / Wilson confidence interval, actual tilt-failure rate,
  timeout fraction and unsafe-entry fraction.
- Maximum tilt P95 (per seed; do not average P95 into a purported pooled P95),
  mean maximum tilt, fraction of episodes crossing the 55° warning level.
- Interception time, relative entry speed, control saturation, worst-task
  success and OOD gaps to observable-PD.
- Safety Cost Critic calibration, mean risk, dual activity, numerical risk
  prediction bias/P95 error, rollout throughput and GPU-memory peak.

Success must be assessed on matched holdout seeds with a separate
confirmation evaluation and with the trained Actor alone at inference.
No result can be claimed superior to V10's 80M-step historical checkpoint
without matching total training budget, initial model, sample distribution
and seed protocol. If V11-C fails more than V11-B, reassess whether the
perturbed initial states are recoverable before increasing their fraction.

Do not change plan/source after a campaign starts: the manifest and checkpoint
source hashes intentionally reject silent resumes after code edits.


## Paired critical-state recovery probe (separate from normal heldout set)

```bash
python -m scripts.run_v11_recovery_probe \
  --checkpoint runs/v11_actuator_safe_ppo/pilot/v11b_constrained_ppo/seed_42/checkpoints/best_current.pt \
  --profile configs/experiments/holdout_random_stronger.json \
  --episodes 256 --batch-size 32 --critical-fraction 0.5 \
  --device cuda \
  --out runs/v11_actuator_safe_ppo/recovery_probe.json
```

The script creates two independent, identically seeded environments at each
batch: a trained deterministic PPO Actor and the observable-PD controller.
Both start from identical relative positions, initial rotations, initial
body rates and thrust. Target noise streams are seeded to match. It reports
actual terminal outcomes, per-controller peak-tilt P95 and the number of
states for which PD succeeds while PPO tilts out. It does **not** prove a
state is in the true viability kernel or that PD is globally safe. Select
probe seeds independently of training and of algorithm tuning.

For the original V10 baseline, run the same probe and pass its checkpoint;
the script disables training-only forecast reward/cost logic during replay.
