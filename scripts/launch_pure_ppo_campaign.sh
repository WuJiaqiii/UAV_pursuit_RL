#!/usr/bin/env bash
# Start/resume all four V12 pure-PPO stages with weighted overall progress.
set -euo pipefail
OUT=$(realpath -m "${1:-runs/v12_pure_ppo_campaign}")
if [[ $# -gt 0 ]]; then shift; fi
PLAN="${V12_CAMPAIGN_PLAN:-configs/experiments/10_v12_pure_ppo_campaign.json}"
DEVICE="${V12_DEVICE:-cuda}"
[[ -f "$PLAN" ]] || { echo "Missing campaign plan: $PLAN" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
if [[ -f "${OUT}.pid" ]] && kill -0 "$(cat "${OUT}.pid")" 2>/dev/null; then
    echo "Campaign is still running: PID $(cat "${OUT}.pid")" >&2
    exit 2
fi
RESUME=()
if [[ -f "$OUT/campaign_manifest.json" ]]; then
    RESUME=(--resume-existing)
elif [[ -e "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
    echo "Nonempty directory without campaign manifest: $OUT" >&2
    exit 2
fi
nohup python -u -m scripts.run_pure_ppo_campaign \
    --plan "$PLAN" --out "$OUT" --device "$DEVICE" \
    "${RESUME[@]}" "$@" >> "${OUT}.log" 2>&1 < /dev/null &
echo "$!" > "${OUT}.pid"
echo "V12 campaign started PID $(cat "${OUT}.pid")"
echo "Overall progress: tail -f ${OUT}.log"
echo "Status snapshot: python -m scripts.run_pure_ppo_campaign --out '$OUT' --status"
echo "Each run log: ${OUT}/logs/"
