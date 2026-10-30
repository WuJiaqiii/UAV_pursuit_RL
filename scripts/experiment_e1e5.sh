#!/usr/bin/env bash
set -euo pipefail
CHECKPOINT="${1:-examples/v5_legacy_best.pt}"
OUTPUT="${2:-runs/e1e5_full}"
python main.py experiments --checkpoint "$CHECKPOINT" --suites E1 E2 E3 E4 E5 \
    --episodes 128 --calibration-episodes 32 --seed 930000 --out "$OUTPUT"
