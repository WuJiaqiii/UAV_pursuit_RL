#!/usr/bin/env bash
# Run from any directory. First initialize, then train, then compare all checkpoints.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
PROFILE="${PROFILE:-configs/rescue/velocity_50hz_2m.json}"
RUN_ROOT="${RUN_ROOT:-runs/v680_rescue}"
SEEDS="${SEEDS:-42}"
PPO_STEPS="${PPO_STEPS:-2000000}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
TEST_SEED="${TEST_SEED:-990000}"
MIN_VALID_SUCCESS="${MIN_VALID_SUCCESS:-0.95}"
[[ "$PPO_STEPS" =~ ^[0-9]+$ ]] || { echo 'PPO_STEPS must be a nonnegative integer' >&2; exit 2; }
[[ "$EVAL_EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'EVAL_EPISODES must be positive' >&2; exit 2; }
[[ -f "$PROFILE" ]] || { echo "Profile missing: $PROFILE" >&2; exit 2; }
profile_name="$(basename -- "$PROFILE" .json)"
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  run_dir="$RUN_ROOT/$profile_name/seed_$seed"
  mkdir -p "$run_dir"
  latest="$run_dir/checkpoints/latest.pt"
  if [[ ! -f "$latest" ]]; then
    "$PYTHON" scripts/rescue.py train --profile "$PROFILE" --device "$DEVICE" --seed "$seed" \
      --physics-steps 0 --out "$run_dir" 2>&1 | tee "$run_dir/initialization_console.log"
  fi
  completed="$({ "$PYTHON" - "$latest" <<'PY'
import sys, torch
saved=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if saved.get('schema')!='pursuit-rescue-checkpoint-v1':raise ValueError('Wrong checkpoint interface')
print(saved['progress']['physics_steps'])
PY
  })"
  if (( PPO_STEPS > completed )); then
    "$PYTHON" scripts/rescue.py train --profile "$PROFILE" --device "$DEVICE" --seed "$seed" \
      --physics-steps "$PPO_STEPS" --resume "$latest" --out "$run_dir" \
      2>&1 | tee "$run_dir/ppo_console_${PPO_STEPS}.log"
  fi
  for name in bc best latest; do
    "$PYTHON" scripts/rescue.py evaluate --checkpoint "$run_dir/checkpoints/$name.pt" \
      --device "$DEVICE" --episodes "$EVAL_EPISODES" --seed "$TEST_SEED" \
      --out "$run_dir/test_${name}.json"
  done
  "$PYTHON" - "$run_dir" "$MIN_VALID_SUCCESS" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]);minimum=float(sys.argv[2])
rows=[]
for name in ('bc','best','latest'):
    data=json.loads((root/f'test_{name}.json').read_text())
    rows.append({'checkpoint':name,'origin':data['checkpoint_origin'],'ppo_updates':data['ppo_updates'],
                 'physics_steps':data['physics_steps'],'capture_radius_m':data['capture_radius_m'],
                 'successes':data['successes'],'episodes':data['episodes'],
                 'success_rate':data['success_rate'],'mean_capture_time_success':data['mean_capture_time_success']})
validation_files=list(root.glob('validation_*.json'))
last=max(validation_files,key=lambda p:int(p.stem.split('_')[-1])) if validation_files else root/'initial_validation.json'
validation=json.loads(last.read_text())
summary={'schema':'pursuit-rescue-comparison-v1','checkpoints':rows,
         'latest_validation_file':last.name,'validation_passed':validation['success_rate']>=minimum,
         'minimum_validation_success':minimum,
         'note':'BC is initialization, not PPO. Test episodes are reports; only the validation set controls this gate.'}
(root/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
if not summary['validation_passed']:
    raise SystemExit('Latest validation missed the required success rate. Inspect comparison.json before extending training.')
PY
  echo "Completed: $run_dir/comparison.json"
done
