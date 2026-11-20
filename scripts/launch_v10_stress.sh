#!/usr/bin/env bash
# Launch independent high-speed stress benchmarks; never trains the Actor.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/launch_v10_stress.sh CAMPAIGN_DIR [medium|high|extreme ...]"
  echo "Example: V10_STRESS_EPISODES=32 bash scripts/launch_v10_stress.sh runs/v10_pd_assisted_ctbr medium"
  exit 2
fi
CAMPAIGN=$(realpath "$1")
shift
GROUPS=( "$@" )
if [[ ${#GROUPS[@]} -eq 0 ]]; then
  GROUPS=(medium)
fi
for group in "${GROUPS[@]}"; do
  case "$group" in
    medium|high|extreme) ;;
    *) echo "Unknown group: $group" >&2; exit 2 ;;
  esac
done
if [[ ! -f "$CAMPAIGN/campaign_results.json" ]]; then
  echo "Missing campaign results: $CAMPAIGN/campaign_results.json" >&2
  exit 2
fi
GROUP_NAME=$(IFS=_; echo "${GROUPS[*]}")
OUT="$CAMPAIGN/highspeed_stress_$GROUP_NAME"
LOG="$CAMPAIGN/highspeed_stress_$GROUP_NAME.log"
PIDFILE="$CAMPAIGN/highspeed_stress_$GROUP_NAME.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Stress test already running, PID $(cat "$PIDFILE")"
  exit 1
fi
EXTRA=()
if [[ -f "$OUT/stress_manifest.json" ]]; then
  EXTRA+=(--resume-existing)
fi
EPISODES=${V10_STRESS_EPISODES:-32}
DEVICE=${V10_STRESS_DEVICE:-cuda}
nohup python -u -m scripts.run_highspeed_stress \
    --campaign-results "$CAMPAIGN/campaign_results.json" \
    --plan configs/experiments/06_v10_pd_assisted_ctbr.json \
    --stage confirmation --groups "${GROUPS[@]}" \
    --episodes "$EPISODES" --device "$DEVICE" --out "$OUT" \
    "${EXTRA[@]}" >> "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PIDFILE"
echo "Started V10 independent high-speed stress PID $(cat "$PIDFILE")"
echo "Monitor: tail -f $LOG"
echo "Result: $OUT/stress_results.json"
echo "Summary: $OUT/stress_summary.csv"
