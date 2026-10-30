#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$(python -c "from pursuit_rl.artifacts.layout import Experiment; print(Experiment.plan('train',name='smoke').directory)")}"
python main.py train --config configs/smoke.yaml --out "$OUT"
python main.py benchmark --run "$OUT" --algorithms PPO PN APN --checkpoint "$OUT/checkpoints/best.pt" --stage 0 --episodes 4
