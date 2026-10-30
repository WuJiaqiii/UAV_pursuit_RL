#!/usr/bin/env python3
"""Real 1-ms plant with configurable policy update intervals; all runs are held-out."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
from pathlib import Path
from pursuit_rl.config import Config
from pursuit_rl.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='configs/high_frequency/ppo_1khz.yaml')
    parser.add_argument('--hz', type=int, nargs='+', default=[20, 50, 100, 200, 500, 1000])
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--scenarios', nargs='+')
    group.add_argument('--stage', type=int)
    parser.add_argument('--algorithms', nargs='+', default=['PPO','PN','APN'])
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=960000)
    parser.add_argument('--out', default='runs/frequency_study')
    parser.add_argument('--plots', action='store_true', help='Enable individual per-frequency plot')
    args=parser.parse_args()
    if any(h <= 0 for h in args.hz):
        parser.error('Frequencies must be positive')
    cfg=Config.load(args.config).to_dict()
    physics_dt=float(cfg['physics']['dt'])
    if abs(physics_dt - .001) > 1e-12:
        parser.error('This study requires physics.dt=0.001 s for matched physical integration')
    scenarios = args.scenarios if args.scenarios else (['ood_pulse_2m','ood_switchback_2m','precision_2m'] if args.stage is None else None)
    root=Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    all_rows=[]
    for hz in args.hz:
        interval=round(1.0/(hz*physics_dt))
        if abs(1.0/(hz*physics_dt)-interval) > 1e-10:
            parser.error(f'{hz} Hz cannot be represented by integral 1ms intervals')
        case=Config(cfg).with_overrides({'physics':{'policy_interval_steps':interval}}).to_dict()
        rows=run_benchmark(cfg=case, algorithms=args.algorithms,
             checkpoint=args.checkpoint, episodes=args.episodes, seed=args.seed,
             out=root/f'{hz:04d}hz', stage=args.stage, scenarios=scenarios, plots=args.plots)
        for row in rows:
            all_rows.append({'requested_hz':hz,'policy_dt_ms':1000.0/hz,
                             'physics_dt_ms':1.0,'episodes':args.episodes, **row})
    with (root/'frequency_summary.csv').open('w', newline='', encoding='utf8') as f:
        keys=['requested_hz','policy_dt_ms','physics_dt_ms','episodes','algorithm','scenario',
              'success_rate','mean_capped_time','lateral_accel_mean_square_mean',
              'control_variation_per_s_mean','high_envelope_any_ratio_mean']
        writer=csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    (root/'frequency_protocol.json').write_text(json.dumps({
      'schema':'frequency-study-v1', 'physics_dt_s':physics_dt, 'seed':args.seed,
      'hz':args.hz, 'scenarios':scenarios, 'stage':args.stage, 'algorithms':args.algorithms,
      'episodes':args.episodes, 'checkpoint':args.checkpoint,
      'note':'Identical model weights across rates: zero-order hold deployment ablation, not independently trained policies.'
    },indent=2),encoding='utf8')
    print(f'Wrote {len(all_rows)} comparison rows to {root / "frequency_summary.csv"}')


if __name__=='__main__':
    main()
