#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${1:-runs/ctbr_v701}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-runs/ctbr_v700/seed_42/15_distance_30/seed_42/checkpoints/best.pt}"
read -r -a SEED_LIST <<< "${SEEDS:-42}"
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    printf 'Source checkpoint missing: %s\nSet SOURCE_CHECKPOINT to the audited v700 stage-15 best.pt.\n' "$SOURCE_CHECKPOINT" >&2
    exit 2
fi
"$PYTHON_BIN" main.py experiment --plan configs/ctbr_v701/experiment.json \
    --source-checkpoint "$SOURCE_CHECKPOINT" --out "$OUT_ROOT" --device "$DEVICE" \
    --seeds "${SEED_LIST[@]}" --test-episodes "${TEST_EPISODES:-64}"
