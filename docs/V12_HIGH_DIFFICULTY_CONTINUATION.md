# V12 pure-PPO: extend after C3 into medium / high / extreme courses

The existing V12 PPO curriculum trains four stages (C0 Static, C1 Low Tail,
C2 Low Head-on, C3 Low Random). **It is now possible to continue from the
already-trained C3 checkpoint** without rerunning those stages or modifying
the original campaign manifest.

Unlike a standalone 13-profile evaluation, the extension **actually performs
new PPO parameter updates** on each harder task, warm-starting each stage from
the previous stage's validation-selected `best_current.pt`. Nothing in the
PPO policy, critic, action space, simulator, 1 kHz physics or 100 Hz CTBR
control changes. LSTM, DAgger, PD teacher, and Safety Critic remain disabled.

## Courses and budgets

Three independent continuation plans are supplied. Run only one high plan
at a time; both are alternatives, not jobs to concatenate automatically.

### Option A: compact medium -> high (recommended first)

File: `configs/experiments/12_v12_high_continue_compact.json`

| Stage | Pure-PPO training task | PPO physics steps |
| --- | --- | ---: |
| C4 | Medium Tail, target 15–20 m/s | 5,000,000 |
| C5 | Medium Random, target 15–20 m/s | 8,000,000 |
| C6 | High Tail, target 40–45 m/s | 10,000,000 |
| C7 | High Random, target 40–45 m/s | 12,000,000 |
| **Total** | **4 new trainings** | **35,000,000** |

Intermediate Head-on lessons are omitted here to limit the time and keep
the training curve focused on tail/random. The jumps in target speed remain
difficult and are not assured to converge.

### Option B: full medium -> high (smoother coverage)

File: `configs/experiments/13_v12_high_continue_full.json`

| Stage | PPO training task | PPO physics steps |
| --- | --- | ---: |
| C4 | Medium Tail | 6,000,000 |
| C5 | Medium Head-on | 6,000,000 |
| C6 | Medium Random | 8,000,000 |
| C7 | High Tail | 10,000,000 |
| C8 | High Head-on | 10,000,000 |
| C9 | High Random | 12,000,000 |
| **Total** | **6 new trainings** | **52,000,000** |

### Option C: extreme continuation (optional, after High Random)

File: `configs/experiments/14_v12_extreme_continue.json`

| Stage | PPO training task | PPO physics steps |
| --- | --- | ---: |
| C10 | Extreme Tail, target 45–50 m/s | 10,000,000 |
| C11 | Extreme Head-on, target 45–50 m/s | 10,000,000 |
| C12 | Extreme Random, target 45–50 m/s | 12,000,000 |
| **Total** | **3 new trainings** | **32,000,000** |

All these use the exact existing named first-entry task profiles; they do not
replace target trajectories, radii, starting states or physical constraints.
The target interception radius is still 0.5m, and the successful entry must
satisfy the original relative-speed condition.

## Run on Ubuntu

Do not change source code in a workspace **while a previous C0–C3 campaign is
still running**. Wait for its C3 completion, or run the extension from a
separate checkout/worktree with a stable copy of C3 best_current.pt.

~~~bash
git fetch origin
git switch --track origin/feat/v12-curriculum-higher-difficulty
# If local branch already exists, use git switch then git pull --ff-only.
pip install -e ".[dev]"
pytest -q tests/test_v12_curriculum.py tests/test_v12_campaign.py

# Point to the original C3 best checkpoint (default old plan, seed 42).
C3="runs/v12_curriculum_01/curr_04_random_seed42/checkpoints/best_current.pt"
test -f "$C3"

# Inspect the HIGH compact plan WITHOUT starting anything.
python -m scripts.run_pure_ppo_curriculum \
  --plan configs/experiments/12_v12_high_continue_compact.json \
  --out runs/v12_high_01 --device cuda --start-from "$C3" --dry-run

# Four real training stages, preserving the C0-C3 output.
V12_CURRICULUM_PLAN=configs/experiments/12_v12_high_continue_compact.json \
  bash scripts/launch_pure_ppo_curriculum.sh runs/v12_high_01 \
  --start-from "$C3"

# Monitor the overall weighted campaign and current course.
tail -f runs/v12_high_01.log

# Read current held-out results and gate status:
python -m scripts.run_pure_ppo_curriculum \
  --out runs/v12_high_01 --status
~~~

For the full medium/high sequence, substitute plan
`13_v12_high_continue_full.json` and use a fresh output directory
`runs/v12_high_full_01`. Keep the same C3 parent.

For the extreme sequence, pass the High Random best checkpoint, not C3.
The last High Random stage is named differently between plans:

~~~bash
# After the compact HIGH extension completes:
HIGH="runs/v12_high_01/curr_04_high_random_seed42/checkpoints/best_current.pt"
test -f "$HIGH"

V12_CURRICULUM_PLAN=configs/experiments/14_v12_extreme_continue.json \
  bash scripts/launch_pure_ppo_curriculum.sh runs/v12_extreme_01 \
  --start-from "$HIGH"

# After the FULL high extension, use
# runs/v12_high_full_01/curr_06_high_random_seed42/checkpoints/best_current.pt
~~~

A continuation plan requires --start-from; the runner validates that the
checkpoint is **pure PPO**, has the expected preceding task (C3 Low Random
for high, High Random for extreme), and preserves Actor/Critic interface and
physics. It records the checkpoint byte SHA256, origin, and original source
fingerprint. A checkpoint from a prior branch can be warm-started if its
interface matches; code/source changes still invalidate resuming an already
started *extension* campaign.

The same launcher command with the **same plan, output and --start-from**
resumes an interrupted extension, skipping completed stages and continuing
unfinished PPO from its own latest.pt. An existing low-speed Campaign is
never overwritten by the extension.

## Success rates *before and after each hard lesson*

The recorder evaluates the preceding checkpoint on the **same fixed-seed
difficulty profiles** before advancing. After PPO training on the new task,
it evaluates the selected best checkpoint on those exact profiles again.

Two files are updated stage by stage:

- `curriculum_success_rates.csv`: each difficulty's successes / episodes,
  success_rate, Wilson95 CI, `before_success_rate`, and
  `learning_gain_pp` (after-minus-before percentage points), plus tilt,
  timeout, unsafe entry and maximum tilt P95.
- `curriculum_results.json`: full stage-by-stage evaluation and external
  parent baseline; each stage still has its own difficulty_matrix.json.

The parent model's initial held-out evaluation lives in
`external_parent_difficulty_matrix.json`. This is *evaluation only* and does
not train the parent. It ensures that a higher-degree task with a favorable
geometry is not misreported as PPO learning improvement.

Each stage's own task appears in the profile list. The compact plan tests
6 fixed difficulties × 32 episodes × 4 stage-end evaluations (plus initial
parent baseline); full plan tests 8 × 32 × 6. Evaluation time is extra,
not counted in the physics-budget progress percentage.

The default `gate_mode=record_only` is deliberate for stress testing: it
continues higher-level training even if the earlier level has <80% success,
but explicitly records FAIL. This is NOT evidence of mastered progression.
For strict mastery advancement, change to `gate_mode=stop` in a NEW plan
and output directory. Under a failed gate, further curriculum stages block.

## Optional independent all-13-level final stress test

After the compact or full medium/high course completes:

~~~bash
python -m scripts.evaluate_v12_curriculum_difficulties \
  --campaign runs/v12_high_01 \
  --device cuda --episodes 64 --seed 970001
~~~

The independent report includes per-profile and aggregated static/low/
medium/high/extreme success rates. Its 64 evaluation episodes per profile
are larger than the 32 interim ones. By default this script selects the
final stage; use --all-stages only if you intentionally want the more
expensive full stage × difficulty matrix.

## Limitations and interpretation

A prior C3 can be validation-selected random_initial, in which case
continuation may start from an essentially untrained Actor. Read the C3
training_summary.json best_origin and own-task success before attributing
higher-level results to transfer. If C4 never improves from the pretraining
baseline, more PPO steps may not be sufficient: inspect reward-scale,
failure termination and action trajectories. The fixed 0.5m capture radius,
15m/s relative-speed limit and high target speeds are demanding.

This implementation supplies code and an experiment protocol, **not
measured high-speed training results**. Use --dry-run and tests first,
then execute on the actual GPU server and inspect the resulting success
matrix and per-stage logs.
