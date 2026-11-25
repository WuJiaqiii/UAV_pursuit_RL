# V13: Pure PPO pursuit convergence repair

This is an opt-in repair branch: existing V12 defaults, experiments and
checkpoints retain their observation, reward and model-selection semantics.
New experiments use --pursuit-v13. This implementation still uses pure PPO,
a feed-forward Actor and direct four-dimensional CTBR. It does not use LSTM,
PD demonstration, DAgger, BC, Safety Critic or learned trajectory prediction.

## Evidence from the uploaded V12 C0-C7 campaign

The complete 55M-step C0-C7 experiment ran 678 PPO updates, ~5731 seconds,
and produced approximately 3469 completed training episodes, only 15 of
which were successful (~0.43%). C0-C3 selected random_initial or
transfer_initial weights, not improved PPO actors. C5-C7 failed all 16
independent task-matched evaluation episodes.

The actual code suggests several contributing mechanisms:

1. **Model selection favored passive waiting.** The old score sorts by
   success rate, then fraction of safe exits (success OR timeout), and only
   then proximity. On C7 High Random, 0-success initial and best checkpoints
   differed mostly in safe timeout rate. The selected checkpoint was
   *farther* from the target (127.64m vs 124.95m mean closest distance).
2. **Saturating distance reward** uses
   k * [d_previous/(d_previous+8) - d_current/(d_current+8)].
   Its per-meter sensitivity falls approximately as 8/(d+8)^2, becoming
   small at 100-220m.
3. **Input scale** divides target-relative position by 100m and relative
   velocity by 50m/s even for near-static and 3-6m/s tasks. A 3m relative
   error contributes an input value of only 0.03, whereas rotation matrix
   and other features are of order one.
4. **Coordinate mismatch in representation complexity**: the existing
   Actor receives world-frame relative geometry but commands BODY rates,
   so it must learn implicit rotation and multistep attitude-to-acceleration
   dynamics. The PD baseline calculates this transform explicitly.
5. **Sparse successful entry and narrow flight constraints**: the
   controller needs enough correlated body-rate commands to tilt,
   accelerate and close a 0.5m first-entry sphere while satisfying
   the 15m/s relative-speed condition. Random exploration is unlikely
   to discover this reliably over long distances. Head-on can be solved
   passively by the approaching target, so head-on successes alone are
   insufficient evidence of active pursuit.
6. **Numerical stability is not convergence**: the runs did not show
   general NaN or runaway KL, but zero success and sparse completions mean
   the policy-gradient signal is poorly aligned with the task.

## What changes and what stays compatible

The following changes only apply with --pursuit-v13:

- policy.relative_frame = body: rotate relative position/velocity, target
  acceleration and ego velocity by R transpose into the drone body frame.
- policy.relative_position_encoding = asinh with scale 5m; keep direction
  and magnitude visible from ~1m to >100m, without linear huge far inputs.
- policy.relative_velocity_scale_mps = 10 for more informative velocities.
- reward.progress_mode = initial_normalized:
  k*(d_previous - d_current)/max(d_initial, capture_radius+0.25m).
  A similar FRACTION of the initial distance earns a similar shaping
  return at different difficulties. Braking, entry velocity, success,
  failure, time and tilt terms remain unchanged.
- training.selection_mode = distance_first:
  rank by success rate, then SMALLER mean closest distance, then safe
  exit rate. Unlike the old rank this does not automatically prefer an
  idle safe-timeout policy to a closer zero-success policy. It is NOT
  safety-certified: always compare tilt/outside failures separately.
- The old meter-based Actor position bound was inconsistent with the
  hardcoded position normalization. The bound now works in physical
  meters for both linear and asinh inputs.
- Observed-PD diagnostic controller explicitly decodes the new body-frame
  observation to preserve the original physical control calculation.

All V12 defaults remain unchanged if --pursuit-v13 is omitted. The
checkpoint loader accepts old implicit world/linear/100m observations
without pretending that the weights were trained for a V13 body-frame
input. A V12 checkpoint cannot be silently resumed under V13 semantics.

## Reproducible experiments on Ubuntu

Prepare:

~~~bash
git fetch origin
git switch --track origin/feat/v13-pursuit-convergence-fixes
pip install -e ".[dev]"
pytest -q tests/test_v13_pursuit.py tests/test_v12_curriculum.py
~~~

Read-only config:

~~~bash
python -m scripts.run_pure_ppo --task near_static --preset quick \
  --pursuit-v13 --out runs/v13_read_only --device cuda --dry-run
~~~

Evaluate whether the physics and zero-action baseline can approach the
target, without training:

~~~bash
python -m scripts.diagnose_pursuit_baselines \
  --tasks near_static static tail head_on random \
  --device cuda --episodes 24 --out runs/v13_controllability.json
~~~

Run Stage W0 near-static (0.75-1.5m, radius 0.5m, duration 6s) then original
Static (2-5m). 3M + 5M PPO physics steps, 24 held-out episodes per profile:

~~~bash
V12_CURRICULUM_PLAN=configs/experiments/16_v13_near_static_probe.json \
  bash scripts/launch_pure_ppo_curriculum.sh runs/v13_near_probe_01
tail -f runs/v13_near_probe_01.log
python -m scripts.run_pure_ppo_curriculum \
  --out runs/v13_near_probe_01 --status
~~~

A longer V13-only curriculum (W0 Near Static 3M -> Static 5M ->
Low Tail 6M -> Low Random 8M) is available:

~~~bash
V12_CURRICULUM_PLAN=configs/experiments/17_v13_pursuit_low_curriculum.json \
  bash scripts/launch_pure_ppo_curriculum.sh runs/v13_low_01
~~~

Matched single-task from-scratch STATIC comparison (same PPO budget,
seed, minibatch settings, 2M/5M etc). This compares the ENTIRE V13 bundle,
not individual features:

~~~bash
python -m scripts.run_pure_ppo --task static --preset standard \
  --device cuda --seed 42 --out runs/v13_static_legacy

python -m scripts.run_pure_ppo --task static --preset standard \
  --device cuda --seed 42 --pursuit-v13 --out runs/v13_static_repaired
~~~

Evaluate trained PPO versus zero-action and geometric PD on exactly the
same episode seeds. PD remains EVALUATION ONLY, never blended into PPO:

~~~bash
python -m scripts.diagnose_pursuit_baselines \
  --checkpoint runs/v13_near_probe_01/curr_02_static_seed42/checkpoints/best_current.pt \
  --tasks near_static static tail head_on random \
  --device cuda --episodes 32 \
  --out runs/v13_pursuit_model_comparison.json
~~~

The audit reports success count/rate, actual initial-to-closest distance
reduction, timeout/tilt/outside reasons, and maximum tilt. If even PD
fails a task, review physical feasibility before attributing everything
to PPO. If zero CTBR succeeds at Head-on, that result is passive pursuit
and must not be presented as acquired tracking ability.

## Scientific interpretation and next steps

1. If Near Static still cannot be learned, inspect the PPO action
   distribution, policy-gradient response, per-episode reward terms and
   real motion/rotation traces. Do not increase speed yet.
2. If Near Static succeeds but Static fails, add an intermediate 1.5-3m
   curriculum and extend coherent action exploration, not a blind extra
   100M steps.
3. If Static succeeds but Tail fails, audit closing speed and braking.
4. For high-speed pursuit, separately analyze dynamic feasibility under
   speed, 2.5g, tilt and capture constraints.
5. If direct body-rate PPO fails consistently while a PD baseline works,
   run a separate hierarchical PPO ablation that learns desired
   acceleration mapped to CTBR by a known lower-level controller. This
   changes the policy action space, so it is NOT the baseline in this
   branch. It would isolate difficult low-level control credit assignment.

The V13 code is a testable set of hypotheses, not proof of successful
learning. Do not change the source or plan of a running V12 campaign.
Run smoke/CPU tests first and report actual GPU evaluation before claiming
convergence.
