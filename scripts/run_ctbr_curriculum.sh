#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-runs/ctbr_v690}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44}"
failures=0
for seed in "${SEED_LIST[@]}"; do
    seed_root="$OUT_ROOT/seed_$seed"
    if "$PYTHON_BIN" main.py curriculum --plan configs/ctbr/plan.json \
        --out "$seed_root" --device "$DEVICE" --seed "$seed" \
        --physics-steps "${PHYSICS_STEPS:-20000000}" \
        --episodes "${GATE_EPISODES:-64}" --minimum-success "${MIN_SUCCESS:-0.9}"; then
        test_seed=$((110000 + seed))
        for model in initial best latest; do
            checkpoint="$seed_root/10_random_high/seed_$seed/checkpoints/$model.pt"
            "$PYTHON_BIN" main.py evaluate --checkpoint "$checkpoint" \
                --device "$DEVICE" --seed "$test_seed" --episodes "${TEST_EPISODES:-256}" \
                --out "$seed_root/test_${model}_deterministic.json"
            "$PYTHON_BIN" main.py evaluate --checkpoint "$checkpoint" --stochastic \
                --device "$DEVICE" --seed "$test_seed" --episodes "${TEST_EPISODES:-256}" \
                --out "$seed_root/test_${model}_stochastic.json"
        done
    else
        printf 'Seed %s stopped at a failed curriculum gate; inspect %s\n' "$seed" "$seed_root" >&2
        failures=$((failures + 1))
    fi
done
if (( failures > 0 )); then exit 1; fi
