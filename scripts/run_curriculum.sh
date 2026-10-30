#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Explicit CLI options take precedence; see scripts/curriculum.py --help.
exec "${PYTHON:-python}" scripts/curriculum.py "$@"
