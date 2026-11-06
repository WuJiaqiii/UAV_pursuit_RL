#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-runs/ctbr_v701_curriculum}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44}"
failures=0
for seed in "${SEED_LIST[@]}"; do
    seed_root="$OUT_ROOT/seed_$seed"
    if "$PYTHON_BIN" main.py curriculum --plan configs/ctbr/plan.json \
        --out "$seed_root" --device "$DEVICE" --seed "$seed" \
        --physics-steps "${PHYSICS_STEPS:-20000000}" \
        --episodes "${GATE_EPISODES:-64}" --minimum-success "${MIN_SUCCESS:-0.95}"; then
        test_seed=$((310000 + seed))
        checkpoint="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phases"][-1]["selected_checkpoint"])' "$seed_root/curriculum_summary.json")"
        profile="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["phases"][-1]["profile"])' "$seed_root/curriculum_summary.json")"
        "$PYTHON_BIN" main.py evaluate --checkpoint "$checkpoint" --config "$profile" \
                --device "$DEVICE" --seed "$test_seed" --episodes "${TEST_EPISODES:-256}" \
                --out "$seed_root/test_selected_deterministic.json"
        "$PYTHON_BIN" main.py evaluate --checkpoint "$checkpoint" --config "$profile" --stochastic \
                --device "$DEVICE" --seed "$test_seed" --episodes "${TEST_EPISODES:-256}" \
                --out "$seed_root/test_selected_stochastic.json"
    else
        printf 'Seed %s stopped; check the experiment gate and logs; inspect %s\n' "$seed" "$seed_root" >&2
        failures=$((failures + 1))
    fi
done
if (( failures > 0 )); then exit 1; fi
