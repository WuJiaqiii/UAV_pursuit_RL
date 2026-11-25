# V12: PPO-only staged curriculum and per-difficulty success rates

This is a **true transfer curriculum**, not the earlier V12 campaign of
independent from-scratch experiments. Each subsequent course imports the
previous stage's validation-selected best_current.pt and continues PPO only.
No LSTM, DAgger, BC, PD teacher, target prediction, or Safety Critic is
enabled. The Actor remains 4D direct CTBR with 100Hz control and 1kHz physics.

## Course definitions

The current V12 first-entry curriculum is:

| Level | Task | Target motion | Distance | Goal radius |
| --- | --- | --- | --- | --- |
| C0 | Static | stationary | 2–5m | 0.5m |
| C1 | Low Tail | cruise, 3–6m/s | 5–15m | 0.5m |
| C2 | Low Head-on | cruise, 3–6m/s | 15–35m | 0.5m |
| C3 | Low Random | random, 4–8m/s; max accel 1m/s² | 10–30m | 0.5m |

This follows the registered V12 difficulty stages. It is **not** an exact
reproduction of the older V6/V7 Straight/Dubins/Singer/Hybrid S0–S3
curriculum, which used different capture radii, initial distances and
trajectory generators. Do not relabel current random/cruise motion as
those unimplemented trajectory generators.

Default plan: configs/experiments/11_v12_pure_ppo_curriculum.json

Each course trains for 5,000,000 PPO physics steps with seed 42.
The three higher levels also have matched direct-from-scratch control runs:
4 curriculum jobs + 3 controls = 7 training jobs / 35M PPO physics steps.
The C0 checkpoint is reused as the common initial-stage result, not trained
twice. Add identical seeds 43/44 to all stages for research replication.

After **every** completed course, the selected Actor is evaluated on all
four levels with 64 deterministic episodes each and identical independent
evaluation seeds across stages and methods. This results in the matrix:

| Training checkpoint | C0 Static | C1 Tail | C2 Head-on | C3 Random |
| --- | --- | --- | --- | --- |
| After C0 | measured | measured | measured | measured |
| After C1 | measured | measured | measured | measured |
| After C2 | measured | measured | measured | measured |
| After C3 | measured | measured | measured | measured |
| C1–C3 direct controls | measured | measured | measured | measured |

No success percentages are filled in before actual GPU evaluations.

Results include episode count, success count, success rate, real tilt
failure fraction, timeout fraction, unsafe entry fraction, peak tilt P95
and trained checkpoint source. C0 performance after C3 exposes catastrophic
forgetting. Curriculum minus direct-control success for the same stage is
reported in percentage points.

The per-stage mastery threshold defaults to 80%. The default gate mode is
record_only: it marks each course PASS/FAIL but **continues** even when a
course fails, so all difficulty results can be diagnosed. This is forced
diagnostic progression, NOT mastery-based advancement. Change gate_mode to
stop in a **new plan/output directory** for strict sequential advancement.

## Commands on the training server

~~~bash
git fetch origin
git switch --track origin/feat/v12-pure-ppo-staged-curriculum
# If already present: git switch feat/v12-pure-ppo-staged-curriculum
git pull --ff-only origin feat/v12-pure-ppo-staged-curriculum
pip install -e ".[dev]"

pytest -q tests/test_v12_curriculum.py tests/test_v12_campaign.py

# Inspect the exact expanded job list; no GPU work:
python -m scripts.run_pure_ppo_curriculum \
  --plan configs/experiments/11_v12_pure_ppo_curriculum.json \
  --out runs/v12_curriculum_01 --device cuda --dry-run

# Run the full curriculum and matched controls in the background:
bash scripts/launch_pure_ppo_curriculum.sh runs/v12_curriculum_01

# Monitor whole-campaign and current stage:
tail -f runs/v12_curriculum_01.log

# Read the saved status without a GPU call:
python -m scripts.run_pure_ppo_curriculum \
  --out runs/v12_curriculum_01 --status

# Read the per-difficulty success matrix:
column -s, -t < runs/v12_curriculum_01/curriculum_success_rates.csv | less -S
~~~

Results appear incrementally:

    runs/v12_curriculum_01/
      campaign_manifest.json
      campaign_progress.json
      curriculum_results.json
      curriculum_success_rates.csv
      .campaign.lock
      logs/
        curr_01_static_seed42.log
        curr_02_tail_seed42.log
        direct_02_tail_seed42.log
      curr_01_static_seed42/
        training_summary.json
        metrics.jsonl
        difficulty_matrix.json
        checkpoints/best_current.pt
        checkpoints/latest.pt
      ...

Overall progress is weighted by each job's physical training-step budget.
The evaluation phase is shown separately and adds overhead. ETA is only an
approximation based on observed wall-clock progress.

## Resume and scientific controls

The same launcher resumes the existing directory using --resume-existing.
Both plan and source fingerprints must match; completed stages are not
retrained, incomplete stages resume their latest compatible checkpoint,
and a missing difficulty matrix is regenerated from the saved best checkpoint.
Never edit the active plan or source code during a running campaign.

The stage control ablation holds the **stage's PPO budget** and independent
test seeds fixed; transferred Actors have earlier-stage pretraining,
so cumulative compute is **not matched**. Do not claim equal-total-compute
superiority from these comparisons.

The 5M-stage plan is intentionally longer than the previous 500k-step
Quick diagnostic (only ~7 updates, almost no completed training episodes).
Start with a read-only dry-run and CPU tests. If you want a short pipeline
smoke, copy the plan, set every stage's physics_steps to 100000,
evaluation.episodes to 8, and compare_direct_baseline to false; use a new
output directory. A smoke is not a convergence experiment.

Only model/solver-specific on-server GPU runs can establish learning
success. CI checks stage dependency logic, per-difficulty reporting,
fixed test seeds, recording and resume. The result is still oracle-state
PPO research, not autonomous flight-system validation.
