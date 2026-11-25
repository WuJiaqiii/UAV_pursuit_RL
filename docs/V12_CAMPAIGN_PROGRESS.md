# V12 Pure PPO — Whole-Campaign Progress

The outer campaign scheduler runs every planned task and seed in sequence,
shows the **overall experiment progress** and **current PPO training progress**
together, and persists snapshots for read-only monitoring and safe resume.
It does not alter the PPO optimizer, CTBR physics or checkpoint format.

## Default plan: four independent pure-PPO tasks

Plan file: configs/experiments/10_v12_pure_ppo_campaign.json

| Stage | Task | Seed | Preset | PPO physics budget |
| --- | --- | --- | --- | ---: |
| 1 | Static | 42 | quick | 500,000 |
| 2 | Low Tail | 42 | quick | 500,000 |
| 3 | Low Head-on | 42 | quick | 500,000 |
| 4 | Low Random | 42 | quick | 500,000 |
| **Total** | | | | **2,000,000** |

These quick experiments provide diagnostic feedback, not guaranteed
convergence. Every task starts with independent random PPO initialization;
there is no LSTM, DAgger, PD teacher or safety Critic. To run a larger
campaign, copy the plan and change defaults.preset from quick to pilot
(2M per stage) or set a custom stages[].physics_steps. Add more seeds using
stages[].seeds. Keep a unique output directory for each altered plan.

## Commands (execute from repository root)

~~~bash
git fetch origin
git switch feat/v12-pure-ppo-fast-baseline
git pull --ff-only origin feat/v12-pure-ppo-fast-baseline
pip install -e ".[dev]"
pytest -q tests/test_v12_campaign.py tests/test_progress.py

# Display the expanded plan and total budget; does not train:
python -m scripts.run_pure_ppo_campaign \
  --plan configs/experiments/10_v12_pure_ppo_campaign.json \
  --out runs/v12_pure_ppo_campaign --device cuda --dry-run

# Start all stages in the background:
bash scripts/launch_pure_ppo_campaign.sh runs/v12_pure_ppo_campaign

# Follow the global live progress:
tail -f runs/v12_pure_ppo_campaign.log

# Read a single current snapshot without GPU work:
python -m scripts.run_pure_ppo_campaign \
  --out runs/v12_pure_ppo_campaign --status
~~~

Illustrative output (not measured training results):

~~~text
[CAMPAIGN] Overall [███████████░░░░░░░░░░░] 50.0% | 2/4 runs | 1,000,000/2,000,000 steps | Elapsed 00:32:00 | ETA 00:32:00 | running
[CURRENT]  Current [█████░░░░░░░░░░░░░░░░░] 25.0% | Stage 3 · Low Head-on / head_on seed=42 | 125,000/500,000 steps | PPO 25 | Train SR 54.2% | running
~~~

The two bars show global and current-stage percentages. The overall bar
weights each run by its planned physics steps, **not** by run count:

    overall_percent = 100 * sum(min(completed_i, budget_i)) / sum(budget_i)

ETA uses observed effective wall-clock progress, including optimizer and
validation overhead, and may fluctuate. While a PPO update is in flight, the
global bar shows the latest **completed PPO update**; it does not claim that
incomplete optimization steps have already committed.

Terminal mode redraws two lines. nohup mode appends a concise pair of
[CAMPAIGN] and [CURRENT] lines every 10 seconds, plus extra lines at stage
boundaries. For a log containing only progress use:

~~~bash
grep -E '^\[(CAMPAIGN|CURRENT)\]' runs/v12_pure_ppo_campaign.log | tail -n 10
~~~

## Persistent files and restart

~~~text
runs/v12_pure_ppo_campaign/
  campaign_manifest.json         # exact plan, runner and source fingerprints
  campaign_progress.json         # atomic overall status and ETA
  campaign_results.json          # completed run summaries
  .campaign.lock                 # prevent two concurrent writers
  logs/
    01_static_seed42.log
    02_tail_seed42.log
    ...
  01_static_seed42/
    config.json
    metrics.jsonl
    training_summary.json
    checkpoints/latest.pt
    ...
~~~

To resume an interrupted campaign, **rerun the same shell launcher**.
When it sees the existing manifest it adds --resume-existing. Alternatively:

~~~bash
python -m scripts.run_pure_ppo_campaign \
  --plan configs/experiments/10_v12_pure_ppo_campaign.json \
  --out runs/v12_pure_ppo_campaign \
  --device cuda --resume-existing
~~~

Resume validates the exact plan hash, source fingerprint, script hash, device,
and per-run config. Completed stages are skipped. Incomplete stages resume
from their latest compatible checkpoint, or initial.pt if there is no latest.
Progress is reconstructed from checkpoint steps rather than optimistic
pre-crash metrics: a small rollback after interruption is expected.

A stage interrupted before its first checkpoint, or with incompatible source
or plan, stops with a clear error and does not overwrite data. Manually
archive that partial run directory before retrying with a fresh run.
The global ETA excludes time spent with the launcher stopped.

The plan/runner is separate from the older V11 28-run campaign. It does NOT
retroactively attach a new progress bar to an already-running V11 process.
This V12 setup executes one subprocess per task, preventing concurrent
training jobs from fighting for the same GPU.

Unit tests validate weighted progress, append-only log rendering, checkpoint
recovery, manifest validation and completed-run skipping. Confirm the latest
CI and run a quick GPU experiment before launching an extended campaign.
