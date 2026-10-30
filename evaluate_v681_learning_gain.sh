#!/usr/bin/env bash
# Run from the V6.8.1 source root; reads models, writes a new evaluation directory.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-runs/v681_curriculum}"
PHASES="${PHASES:-09_mixed_speed8}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SEEDS="${SEEDS:-1000000}"
EPISODES="${EPISODES:-64}"
NUM_ENVS="${NUM_ENVS:-64}"
stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-runs/v681_gain_audit/$stamp}"
cd "$PROJECT_ROOT"
[[ -f scripts/challenge.py ]] || { echo 'Run from the V6.8.1 source root or set PROJECT_ROOT' >&2; exit 2; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'EPISODES must be positive' >&2; exit 2; }
[[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_ENVS must be positive' >&2; exit 2; }
[[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || { echo 'Invalid training seed' >&2; exit 2; }
for seed in $SEEDS; do [[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Invalid seed' >&2; exit 2; }; done
for phase in $PHASES; do
  for policy in deterministic stochastic; do
    for kind in initial latest; do
      filename=latest.pt
      if [[ "$kind" == initial ]]; then filename=bc.pt; fi
      checkpoint="$RESULT_ROOT/$phase/seed_${TRAIN_SEED}/checkpoints/$filename"
      [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
      destination="$OUT_ROOT/$phase/$policy/$kind"
      mkdir -p "$destination"
      if [[ "$policy" == stochastic ]]; then
        "$PYTHON" scripts/challenge.py --checkpoint "$checkpoint" --device "$DEVICE" \
          --scenarios training_task --episodes "$EPISODES" --num-envs "$NUM_ENVS" \
          --seeds $SEEDS --no-reference --stochastic --out "$destination" \
          2>&1 | tee "$destination/console.log"
      else
        "$PYTHON" scripts/challenge.py --checkpoint "$checkpoint" --device "$DEVICE" \
          --scenarios training_task --episodes "$EPISODES" --num-envs "$NUM_ENVS" \
          --seeds $SEEDS --no-reference --out "$destination" \
          2>&1 | tee "$destination/console.log"
      fi
    done
  done
done
"$PYTHON" - "$OUT_ROOT" <<'PY'
import json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);comparisons=[]
for phase in sorted(p for p in root.iterdir() if p.is_dir()):
    phase_result={'phase':phase.name,'policies':{}}
    for policy in ('deterministic','stochastic'):
        groups={}
        raw={}
        for kind in ('initial','latest'):
            directory=phase/policy/kind
            paths=sorted(directory.glob('training_task__seed_*__learned__*.json'))
            if not paths:raise RuntimeError('Missing evaluation reports: '+str(directory))
            reports=[json.loads(p.read_text()) for p in paths]
            raw[kind]={r['seed']:r for r in reports}
            count=sum(r['episodes'] for r in reports)
            records=[e for r in reports for e in r['episodes_detail']]
            successful=[r['time'] for r in records if r['success']]
            penalized=sum(sum(e['time'] if e['success'] else r['evaluation_contract']['task']['time']
                              for e in r['episodes_detail']) for r in reports)/count
            groups[kind]={'episodes':count,'successes':sum(r['successes'] for r in reports),
                          'success_rate':sum(r['successes'] for r in reports)/count,
                          'mean_capture_time_success':statistics.mean(successful) if successful else None,
                          'penalized_mean_capture_time':penalized,
                          'mc_return_mean':sum(r['mc_return_mean']*r['episodes'] for r in reports)/count,
                          'value_bias':sum(r['critic_bias']*r['episodes'] for r in reports)/count,
                          'value_mc_rmse':math.sqrt(sum(r['critic_rmse']**2*r['episodes'] for r in reports)/count)}
        if set(raw['initial'])!=set(raw['latest']):raise RuntimeError('Evaluation seed sets differ')
        for seed,r in raw['initial'].items():
            other=raw['latest'][seed]
            if r['initial_state_digest']!=other['initial_state_digest']:raise RuntimeError('Initial states differ')
            if r['evaluation_contract']!=other['evaluation_contract']:raise RuntimeError('Evaluation tasks differ')
        before=groups['initial'];after=groups['latest']
        groups['comparison']={'paired_initial_conditions':True,
            'relative_penalized_time_change':after['penalized_mean_capture_time']/before['penalized_mean_capture_time']-1,
            'mc_return_change':after['mc_return_mean']-before['mc_return_mean'],
            'critic_calibration_matches_policy':policy=='stochastic',
            'note':'Point estimates on new evaluation seeds; no claim of statistical significance or multiple training seeds.'}
        phase_result['policies'][policy]=groups
    comparisons.append(phase_result)
result={'schema':'v681-initial-latest-gain-audit-v1','phases':comparisons,
        'note':'No training was performed. Initial is transfer initialization (bc.pt), not a randomly initialized actor.'}
(root/'gain_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
PY
echo "Saved: $OUT_ROOT/gain_comparison.json"
