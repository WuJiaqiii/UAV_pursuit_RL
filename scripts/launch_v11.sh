#!/usr/bin/env bash
# V11 unattended campaign. Four matched variants; archive existing attempts
# via the Python runner, never silently overwrite source-incompatible runs.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/launch_v11.sh TAIL_PARENT.pt STATIC_PARENT.pt [RUN_DIR]"
  exit 2
fi
PARENT=$(realpath "$1")
STATIC=$(realpath "$2")
OUT=$(realpath -m "${3:-runs/v11_actuator_safe_ppo}")
PLAN="configs/experiments/08_v11_actuator_safe_ppo.json"
[[ -f "$PARENT" && -f "$STATIC" && -f "$PLAN" ]] || { echo "Missing parent checkpoint or plan"; exit 2; }
mkdir -p "$OUT"
if [[ -f "$OUT/campaign.pid" ]] && kill -0 "$(cat "$OUT/campaign.pid")" 2>/dev/null; then
  echo "V11 already running: $(cat "$OUT/campaign.pid")"; exit 1
fi
EXTRA=()
if [[ -f "$OUT/campaign_manifest.json" ]]; then EXTRA+=(--resume-existing); fi
nohup python -u -m scripts.run_generalization_campaign \
  --plan "$PLAN" --parent-checkpoint "$PARENT" \
  --static-checkpoint "$STATIC" --out "$OUT" \
  --stage all --device "${V11_DEVICE:-cuda}" "${EXTRA[@]}" \
  >> "$OUT/campaign.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/campaign.pid"
echo "V11 started PID $(cat "$OUT/campaign.pid"); log: $OUT/campaign.log"
