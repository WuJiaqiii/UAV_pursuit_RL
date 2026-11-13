python -m pursuit_rl.ctbr.trim \
  --checkpoint A=runs/ctbr_v701/seed_42/initializations/A.pt \
  --checkpoint near10=runs/staged_v707_near_zero/update_0010/candidate.pt \
  --checkpoint recovery10=runs/staged_v707_recovery/update_0010/candidate.pt \
  --device cuda --out runs/trim_v708_1mps \
  --speeds 1 --drag-factors 0 0.5 1 --scenarios 8

for arm in recovery trim_recovery; do
  python -m pursuit_rl.ctbr.staged_behavior \
    --checkpoint runs/ctbr_v701/seed_42/initializations/A.pt \
    --device cuda --out "runs/staged_v708_${arm}" \
    --data-mix "$arm" --samples 28800 \
    --episodes 32 --cruise-episodes 8 --recovery-episodes 12 \
    --anchor-weight 1 --milestones 10 25 50 --rank-by-precision || break
done