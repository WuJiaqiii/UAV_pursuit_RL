#!/usr/bin/env bash
# Sequential PPO-only curriculum: transfer C0 -> C1 -> C2 -> C3 + scratch controls.
set -euo pipefail
OUT=$(realpath -m "${1:-runs/v12_pure_ppo_curriculum}")
if [[ $# -gt 0 ]]; then shift; fi
PLAN="${V12_CURRICULUM_PLAN:-configs/experiments/11_v12_pure_ppo_curriculum.json}"
DEVICE="${V12_DEVICE:-cuda}"
[[ -f "$PLAN" ]] || { echo "Missing curriculum plan: $PLAN" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
if [[ -f "${OUT}.pid" ]] && kill -0 "$(cat "${OUT}.pid")" 2>/dev/null; then
    echo "Curriculum already running PID $(cat "${OUT}.pid")" >&2
    exit 2
fi
RESUME=()
if [[ -f "$OUT/campaign_manifest.json" ]]; then
    RESUME=(--resume-existing)
elif [[ -e "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
    echo "Nonempty output without manifest: $OUT" >&2
    exit 2
fi
nohup python -u -m scripts.run_pure_ppo_curriculum \
    --plan "$PLAN" --out "$OUT" --device "$DEVICE" \
    "${RESUME[@]}" "$@" >> "${OUT}.log" 2>&1 < /dev/null &
echo "$!" > "${OUT}.pid"
echo "Curriculum PID $(cat "${OUT}.pid")"
echo "Whole campaign progress: tail -f ${OUT}.log"
echo "Success matrix: ${OUT}/curriculum_success_rates.csv"
echo "Status: python -m scripts.run_pure_ppo_curriculum --out '$OUT' --status"
