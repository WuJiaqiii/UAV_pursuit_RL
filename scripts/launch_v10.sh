#!/usr/bin/env bash
# One command starts the complete v10 pilot → full → confirmation campaign.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/launch_v10.sh TAIL_BEST.pt STATIC_BEST.pt [OUTPUT_DIR]"
  exit 2
fi
PARENT=$(realpath "$1")
STATIC=$(realpath "$2")
OUT=$(realpath -m "${3:-runs/v10_generalization}")
[[ -f "$PARENT" && -f "$STATIC" ]] || { echo "Missing input checkpoint"; exit 2; }
mkdir -p "$OUT"
PLAN="configs/experiments/05_v10_generalization.json"
if [[ -e "$OUT/campaign.pid" ]] && kill -0 "$(cat "$OUT/campaign.pid")" 2>/dev/null; then
  echo "v10 campaign already running, PID $(cat "$OUT/campaign.pid")"
  exit 1
fi
EXTRA=()
if [[ -f "$OUT/campaign_manifest.json" ]]; then
  EXTRA+=(--resume-existing)
fi
nohup python -u scripts/run_generalization_campaign.py \
    --plan "$PLAN" --parent-checkpoint "$PARENT" \
    --static-checkpoint "$STATIC" --out "$OUT" \
    --stage all --device cuda "${EXTRA[@]}" \
    >> "$OUT/campaign.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/campaign.pid"
echo "Started v10 PID $(cat "$OUT/campaign.pid")"
echo "Follow progress: tail -f $OUT/campaign.log"
echo "Structured results: $OUT/campaign_results.json"
