#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -gt 0 ]]; then
  OUT="$1"
  shift
  python main.py train --out "$OUT" "$@"
else
  python main.py train --name full_curriculum
fi
