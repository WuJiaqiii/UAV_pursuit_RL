python scripts/basic_experiment_matrix.py \
  --group standard \
  --only standard_pure standard_pd02 standard_pd02_no_progress \
  --seeds 42 43 44 \
  --device cuda \
  --backend torch \
  --out-root runs/v672_standard_formal