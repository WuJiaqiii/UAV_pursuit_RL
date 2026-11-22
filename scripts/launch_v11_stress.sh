#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: bash scripts/launch_v11_stress.sh CAMPAIGN_DIR [medium|high|extreme ...]"; exit 2; }
V10_PLAN=configs/experiments/08_v11_actuator_safe_ppo.json \
  bash scripts/launch_v10_stress.sh "$@"
