export PARENT="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_RF/code/runs/v8_curriculum/02_low_tail/seed_42/checkpoints/best_current.pt"
export STATIC="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_RF/code/runs/v8_curriculum/01_static/seed_42/checkpoints/best_feasible.pt"

python -m scripts.run_generalization_campaign \
  --plan configs/experiments/08_v11_actuator_safe_ppo.json \
  --parent-checkpoint "$PARENT" \
  --static-checkpoint "$STATIC" \
  --out runs/v11_dryrun \
  --stage pilot \
  --device cuda \
  --dry-run

python -m scripts.run_generalization_campaign \
  --plan configs/experiments/08_v11_actuator_safe_ppo.json \
  --parent-checkpoint "$PARENT" \
  --static-checkpoint "$STATIC" \
  --out runs/v11_actuator_safe_ppo \
  --stage pilot \
  --device cuda

bash scripts/launch_v11.sh \
  "$PARENT" \
  "$STATIC" \
  runs/v11_actuator_safe_ppo

python -m scripts.audit_v11_safety \
  --checkpoint /path/to/v11_checkpoint.pt \
  --profile configs/experiments/holdout_random_stronger.json \
  --episodes 512 \
  --critical-fraction 0.5 \
  --device cuda \
  --out runs/v11_safety_audit.json

python -m scripts.run_v11_recovery_probe \
  --checkpoint /path/to/v11_checkpoint.pt \
  --profile configs/experiments/holdout_random_stronger.json \
  --episodes 256 \
  --critical-fraction 0.5 \
  --device cuda \
  --out runs/v11_recovery_probe.json