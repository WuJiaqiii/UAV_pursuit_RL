# V12.1 Direct CTBR PPO: Pursuit Failure Root Causes and Recovery

This is a code-grounded review of the prior uploaded v12_scratch_to_high_01.zip
and a controlled research experiment. It does NOT claim the repaired strategy
has already learned to intercept a UAV.

## Confirmed findings from the preceding C0-C7 campaign

The old 8-stage run used 55 million PPO physics steps (~5731 seconds wall time).
Selected checkpoints achieved C0 Static 0/16, C1 Low Tail 1/16, C2 Low
Head-on 16/16, C3 Low Random 0/16, C4 Medium Tail 2/16, C5 Medium
Random 0/16, C6 High Tail 0/16, C7 High Random 0/16 on fixed stage
difficulty evaluations. C0's best_origin was random_initial, and C1-C3
were transfer_initial. Most of the apparent Low Head-on success was already
present *before* any PPO training (32/32 initially): it must be compared
with a zero-action baseline to exclude passive target interception.

In High Random, selection favored 96.875% success-or-timeout rather than
34.375%, even as mean minimum target distance grew from about 124.95m
to 127.64m, with zero actual success. The legacy lexicographic score
prioritizes the probability of safe timeout ahead of making progress.

Legacy shaping is r = w*[d_old/(d_old+L) - d_new/(d_new+L)].
Its sensitivity to closure is w*L/(d+L)^2, which collapses at long range.
For w=6, L=8, d=160m, only ~0.000170 scaled reward is gained per meter,
versus ~0.0397 at 3m. This is over a 200x discrepancy.

Legacy pure PPO also sets all Actor output-head weights to zero and
the first-layer weights after input feature 21 to zero. Body rates,
thrust state and raw action history are ignored on the first update;
backpropagated action gradients to earlier layers vanish initially.
Actor input relative position is divided by 100m, yielding near-static
C0 offsets of only 0.02 to 0.05 while rotation elements can be order one.

The exploratory action correlation time was 20ms, shorter than the
60ms rate and 65ms thrust actuator response; stochastic rate commands
are filtered by actuator lags unless sustained coherently. Stable
hover from the zero-mean Actor produces little useful pursuit experience.

PPO critic explained variance near one does not imply control mastery:
a Critic can accurately learn to forecast a consistent timeout penalty.
The inherited three-second GAE advantage horizon also makes multi-second
action credit assignment difficult, although this is a hypothesis to test.

## Implemented opt-in interventions (old modes remain available)

1. Reward progress_mode options:
   legacy_saturating (historical), relative_distance_normalized,
   pursuer_closure_normalized.
   Relative distance: w*(d_old-d_new)/max(d_initial,d_floor).
   Pursuer-only: w*(norm(target_new-pursuer_old)
                    - norm(target_new-pursuer_new))/max(d_initial,d_floor).
   Pursuer-only shaping isolates the effect of pursuer motion and does NOT
   credit a static pursuer when a head-on target approaches it. A 1m active
   closure from 160m yields ~0.00375 scaled reward, roughly 22x the
   original 160m-local effect, with w=6, reward.scale=0.1.

2. Checkpoint selection pursuit_progress:
   score = (actual success fraction,
            mean episode-normalized closure - 0.35*hard_failure_rate,
            success-or-timeout fraction).
   The actual success rate remains primary. Progress gains below ~1%
   are not grounds for arbitrary checkpoint replacement. Hard physical
   failures remain penalized. Legacy selection can be restored at any time.

3. Observation encoding policy.relative_position_mode=asinh:
   obs_position = asinh(relative_position/scale_m).
   This nonlinear transform is odd, invertible and smooth, making near
   offsets more visible while compressing extreme distances. Relative
   velocity scale is separately configurable. The evaluation-only
   geometric PD control decodes asinh before calculating physical commands.

4. Exploration-time correlation and initial_std exposed as CLI flags.
   The recovery plan sets correlation time=0.18 seconds and std=0.18
   (versus previous 0.02s and std=0.15) to allow sustained rate actions
   that can overcome physical actuator filtering. These are hypotheses:
   too much coherent exploration can also cause tilt failures.

5. Actor initialization small_gain:
   last-layer orthogonal gain 0.02, zero bias, with input features
   preserved, rather than a completely zero output head. This permits
   gradients to reach hidden layers during the first PPO update.

6. A near_static task with 0.7-1.5m initial separation, 0.5m capture
   radius and 6-second time limit, initial zero pursuer speed and
   level orientation. Starts are strictly outside the success sphere.

The pure PPO model remains an asymmetric Actor-Critic with a 4D direct
CTBR action. No PD teacher data, BC, LSTM, risk predictor or auxiliary
critic loss is added. The dynamics still run at 1kHz with 100Hz control.

Important scientific caveat: these distance and pursuer-only rewards
are not claimed to satisfy the strict discount-corrected potential-based
shaping policy-invariance theorem. Compare alternative reward modes and
inspect trajectories for cyclic reward exploits or unsafe rushing.

## First: prove simulator controllability and detect passive success

From repository root:

    git fetch origin
    git switch --track origin/feat/v12-1-pursuit-learning-fixes
    pip install -e ".[dev]"
    pytest -q tests/test_v12_1_pursuit_fixes.py

    python -m scripts.audit_pursuit_learnability \
      --task near_static --device cuda --episodes 16 --pursuit-fixes \
      --out runs/v12_1_audit_near_static.json

    python -m scripts.audit_pursuit_learnability \
      --task head_on --device cuda --episodes 16 \
      --out runs/v12_1_audit_headon.json

These evaluations compare a zero-action hover CTBR command against
the repository's observed-state geometric PD reference on identical
seeds. The PD reference is used ONLY as an analysis baseline and never
trains or blends with the PPO Actor. If PD cannot succeed, inspect
the simulator/CTBR sign, authority and task geometry before PPO tuning.

## Next: run the from-scratch recovery curriculum

File: configs/experiments/16_v12_1_pursuit_recovery.json

  R0 Near Static (0.7-1.5m): 6 million PPO physics steps
  R1 Static      (2-5m)  : 8 million PPO physics steps
  R2 Low Tail    (3-6m/s): 8 million PPO physics steps
  Total: 22 million PPO physics steps, plus heldout evaluations.

Default: envs=64, rollout=128, minibatch=512, PPO epochs=2,
std=0.18, correlated exploration time=0.18 seconds,
near-hover small_gain head, asinh position scale=8m, velocity scale=15m/s,
pursuer-only progress reward and chase-aware checkpoint selector.
The test matrix covers Near Static, Static, Tail, Head-on and Random.
24 heldout episodes per task are diagnostic, not enough for a final
statistical claim. Repeat with 64+ episodes and several training seeds.

    python -m scripts.run_pure_ppo_curriculum \
      --plan configs/experiments/16_v12_1_pursuit_recovery.json \
      --out runs/v12_1_recovery_01 --device cuda --dry-run

    V12_CURRICULUM_PLAN=configs/experiments/16_v12_1_pursuit_recovery.json \
      bash scripts/launch_pure_ppo_curriculum.sh runs/v12_1_recovery_01

    tail -f runs/v12_1_recovery_01.log

    python -m scripts.run_pure_ppo_curriculum \
      --out runs/v12_1_recovery_01 --status

This is new from-scratch training, no checkpoint required.
Mastery mode remains record_only so the full diagnostic matrix is
produced; a recorded FAIL is not evidence the lesson was mastered.

## Controlled single-task ablation (do NOT claim each flag helped)

A: historical near-static PPO, 5M steps:

    python -m scripts.run_pure_ppo --task near_static --preset standard \
      --device cuda --out runs/v12_1_legacy_seed42

B: change only reward (5M steps):

    python -m scripts.run_pure_ppo --task near_static --preset standard \
      --device cuda --progress-mode pursuer_closure_normalized \
      --out runs/v12_1_reward_only_seed42

C: all interventions combined (5M steps):

    python -m scripts.run_pure_ppo --task near_static --preset standard \
      --device cuda --progress-mode pursuer_closure_normalized \
      --checkpoint-selection pursuit_progress \
      --relative-position-mode asinh --relative-position-scale-m 8 \
      --relative-velocity-scale-mps 15 \
      --initial-std 0.18 --exploration-time-s 0.18 \
      --actor-initialization small_gain \
      --out runs/v12_1_combined_seed42

Use more single-factor variants to attribute effects. Keep physical
budgets, seeds and evaluation episodes identical. Compare independently
measured SUCCESS/N and Wilson intervals, normalized minimum-distance
closure, unsafe/tilt/timeout failure reasons, action saturation,
3D pursuit trajectories, PPO KL/clip and update throughput. Evaluate
both best_current.pt and latest.pt, and include zero-action and
evaluation-only observed-PD references.

## If this still fails

1. Inspect signed CTBR roll/pitch commands, vehicle rotation and
   acceleration, and distance time series to confirm control authority.
2. Increase the PPO advantage-time horizon beyond 3s as a controlled
   ablation; avoid changing reward/exploration at the same time.
3. Verify independent rollout success feedback and critic bootstrap
   at timeout boundaries; successful initial PD tests isolate
   model-learning failure from task infeasibility.
4. Compare direct body-rate commands against acceleration setpoint
   commands in a SEPARATE research experiment. Direct CTBR demands
   learned vehicle attitude stabilization and target pursuit jointly.
5. Only add medium/high/extreme tasks when near/static/trailing success
   and normalized closure improve on heldout seeds.

No fresh GPU training is run or claimed by this patch. Source changes
intentionally break old in-place resume; use new output directories.
The checkpoint loader preserves legacy fingerprints for reading older
checkpoints, but new observation semantics require new from-scratch
training rather than silent weight transfer.
