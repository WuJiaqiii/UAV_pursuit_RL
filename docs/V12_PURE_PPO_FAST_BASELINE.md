# V12: minimal direct-CTBR PPO and fast-simulation baseline

This independent research branch isolates the smallest reasonable
state-observation PPO experiment. V11 was a **28-run, 440-million-physics-step
campaign**; its ~40-hour total is not the duration of one PPO training run.
V11 also invoked LSTM target forecasting, a per-action recovery-risk
predictor, DAgger, cost critics, many evaluations, and 1-kHz physics.
No numerical throughput or success-rate improvement is claimed without a
fresh run on the target GPU.

## Algorithm contract

- From-scratch, feed-forward Actor (38 state dimensions -> 4 CTBR outputs),
  privileged reward Critic trained by PPO, PPO's on-policy GAE.
- Target forecast fully disabled. No LSTM, predictor optimizer, or learned
  extrapolation. One previous raw action is still an observation feature,
  not a recurrent network.
- No DAgger/BC, PD aux supervision, fixed-policy distillation, Cost Critic,
  predictive safety rollout, risk curriculum, or MC Critic pretraining.
- The usual physical safety termination and existing task reward remain.
  The simulator stays at 1 kHz and policy at 100 Hz.
- The state is simulator/oracle observation; this is not a camera-only result.
- Optional warm start from **another pure PPO checkpoint** still uses PPO
  alone; it is not imitation learning.

## Progressive experiments (run separately)

| Task | Meaning | Command argument |
| --- | --- | --- |
| Static | approach/entry to stationary target | --task static |
| Tail | follow low-speed target from behind | --task tail |
| Head-on | handle approaching target and closing-speed limit | --task head_on |
| Random | follow basic low-speed random maneuvers | --task random |

| Preset | PPO physics steps | Validation frequency | Validation episodes |
| --- | ---: | ---: | ---: |
| smoke | 100,000 | 100,000 | 8 |
| quick | 500,000 | 250,000 | 12 |
| pilot | 2,000,000 | 500,000 | 24 |
| standard | 5,000,000 | 1,000,000 | 32 |

These are *budgets*, not promised clock times or proof of convergence.
Defaults: 64 environments, rollout 128, PPO minibatch 512, PPO epochs 2,
Actor learning rate 1e-4, critic learning rate 3e-4, initial std 0.15.
The final layer starts with zero mean (neutral CTBR/hover thrust). The
natural from-scratch reward may still be too sparse for some harder tasks;
do not interpret a failed 100k smoke as an algorithmic impossibility.

## Preparation

From repository root, activate your Python environment and run:

    git fetch origin
    git switch --track origin/feat/v12-pure-ppo-fast-baseline
    pip install -e ".[dev]"
    pytest -q tests/test_v12_pure_ppo.py tests/test_training_integration.py

Dry-run configuration inspection (no training):

    python -m scripts.run_pure_ppo --task static --preset quick \
      --device cuda --out runs/v12_static_quick --dry-run

Smallest complete training smoke:

    python -m scripts.run_pure_ppo --task static --preset smoke \
      --device cuda --out runs/v12_static_smoke

One unattended Pilot **without the V11 campaign**:

    bash scripts/launch_pure_ppo.sh static pilot runs/v12_static_pilot \
      --num-envs 64 --batch-size 512 --rollout-steps 128
    tail -f runs/v12_static_pilot.log

Then run a separate Tail from scratch:

    bash scripts/launch_pure_ppo.sh tail pilot runs/v12_tail_scratch

Optional PPO-only transfer experiment (no PD demonstrations):

    bash scripts/launch_pure_ppo.sh tail pilot runs/v12_tail_transfer \
      --init-from runs/v12_static_pilot/checkpoints/best_current.pt

Use independent test/validation seeds and equal budgets when comparing the
from-scratch and PPO-pretrained variants.

Resume an existing pure PPO run with an *unchanged source and configuration*,
increasing only its physics-step limit:

    python -m scripts.run_pure_ppo --out runs/v12_static_pilot \
      --resume runs/v12_static_pilot/checkpoints/latest.pt \
      --device cuda --physics-steps 4000000

Never reuse a V11 campaign directory or V11 + LSTM checkpoint for V12.

## Throughput improvements

1. Single training run instead of 28 sequential ablation jobs.
2. No teacher or trajectory-forecast data collection.
3. No action-conditioned 12-step GPU safety prediction.
4. No stochastic Monte-Carlo Critic warmup/calibration for the pure-PPO mode;
   GAE and PPO value loss remain unchanged.
5. No unused next-target-prediction buffer when forecasting is off.
6. Latest checkpoint saved every 10 PPO updates, at validation and at the
   final update rather than on every update; earlier updates may need to be
   replayed following a crash.
7. Batched GPU terminal materialization **once per 100-Hz decision** instead
   of converting 1-kHz per-tick terminal events to CPU. Physics integration,
   timeout/tilt/success test, and first crossing retain their original timing.
   Legacy terminal recording remains available via
   --legacy-terminal-records.

Actual GPU throughput must still be measured; the simulator launches many
small tensor kernels at 1 kHz. A large GPU memory reserve or unused VRAM
does not imply high compute utilization.

Use an environment-only A/B benchmark:

    python -m scripts.benchmark_pure_ppo \
      --task static --device cuda --num-envs 64 --decisions 100

This reports physics steps per second and GPU peak reserved memory, but
excludes PPO updates, evaluation and checkpoint writes.

Each event=ppo_update in the training metrics file includes:

- rollout_wall_s: batched dynamics + environment interaction;
- optimizer_wall_s: policy/critic gradient update;
- physics_steps_per_rollout_second: actual environment throughput;
- training_wall_s: elapsed core PPO trainer time.

The single-run pure_ppo_summary.json records total wall time including
setup, validation, and checkpointing, allowing direct bottleneck comparison.

## Correctness and experimental caveats

- The faster event path has a CPU equivalence suite for static/cruise/random,
  including first-entry success and unsafe-entry reasons. Run tests before
  a long GPU experiment and the benchmark before claiming speedup.
- Old V8–V11 archived checkpoints remain loadable in this branch. Normal
  source-fingerprint and PPO resume checks stay strict.
- The new pure PPO Actor uses a shorter 1-action observation history than
  previous V11 policies; it is a **different input architecture**. Scientific
  comparisons must retrain from a suitable initialization rather than load
  incompatible V11 LSTM weights.
- The result is *not* formal flight safety or a measured 1–2 hour guarantee.
  Start with smoke and quick; increase budgets only if learning curves merit it.
