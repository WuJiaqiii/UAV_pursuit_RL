#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for SEED in 42 43 44; do
  python main.py train --name multi_seed --seed "$SEED" "$@"
done
