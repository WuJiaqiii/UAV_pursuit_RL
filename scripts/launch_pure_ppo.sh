#!/usr/bin/env bash
# Minimal unattended direct-CTBR pure PPO experiment. No V11 campaign.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/launch_pure_ppo.sh static|tail|head_on|random smoke|quick|pilot|standard [RUN_DIR] [additional Python flags]"
  exit 2
fi
TASK="$1"; PRESET="$2"; shift 2
case "$TASK" in static|tail|head_on|random) ;; *) echo "Invalid task: $TASK"; exit 2;; esac
case "$PRESET" in smoke|quick|pilot|standard) ;; *) echo "Invalid preset: $PRESET"; exit 2;; esac
OUT=$(printf "runs/pure_ppo_%s_%s" "$TASK" "$PRESET")
if [[ $# -gt 0 && "$1" != --* ]]; then OUT="$1"; shift; fi
OUT=$(realpath -m "$OUT")
if [[ -e "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
  echo "Run directory already has data: $OUT"
  if [[ " $* " != *" --resume "* ]]; then
    echo "Use a fresh run dir; or explicitly supply --resume checkpoint."
    exit 2
  fi
fi
mkdir -p "$(dirname "$OUT")"
if [[ -f "$OUT.pid" ]] && kill -0 "$(cat "$OUT.pid")" 2>/dev/null; then
  echo "Process still running: $(cat "$OUT.pid")"; exit 2
fi
DEVICE=$(printenv PURE_PPO_DEVICE 2>/dev/null || true)
DEVICE=$(printf "%s" "$DEVICE")
if [[ -z "$DEVICE" ]]; then DEVICE="cuda"; fi
nohup python -u -m scripts.run_pure_ppo \
  --task "$TASK" --preset "$PRESET" --out "$OUT" \
  --device "$DEVICE" "$@" \
  >> "$OUT.log" 2>&1 < /dev/null &
echo "$!" > "$OUT.pid"
echo "Pure PPO started PID $(cat "$OUT.pid")"
echo "Log: $OUT.log"
echo "Results: $OUT/pure_ppo_summary.json"
