#!/usr/bin/env python3
"""Evaluate each best_stage0 checkpoint with an independent fixed test seed."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs-root',type=Path,default=ROOT/'runs'/'basic_v67_matrix')
    ap.add_argument('--report-dir',type=Path,default=ROOT/'runs'/'basic_v67_independent_eval')
    ap.add_argument('--episodes',type=int,default=256)
    ap.add_argument('--seed',type=int,default=990000)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--backend',choices=['torch','numpy'],default='torch')
    ap.add_argument('--checkpoint',choices=['best_stage0','best','latest'],default='best_stage0')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    found=list(sorted(args.runs_root.glob('*/checkpoints/'+args.checkpoint+'.pt')))
    if not found:
        print('No matching checkpoints under',args.runs_root,file=sys.stderr);return 2
    rows=[]
    for checkpoint in found:
        run=checkpoint.parent.parent
        config=run/'config.yaml'
        if not config.is_file():
            print('Missing resolved config',config,file=sys.stderr);return 2
        out=(args.report_dir/(run.name+'__'+args.checkpoint+'.json')).resolve()
        cmd=[sys.executable,str(ROOT/'main.py'),'evaluate','--config',str(config.resolve()),
            '--checkpoint',str(checkpoint.resolve()),'--algorithm','PPO',
            '--stage','0','--episodes',str(args.episodes),'--seed',str(args.seed),
            '--device',args.device,'--simulation-backend',args.backend,'--out',str(out)]
        print(' '.join(cmd),flush=True)
        if args.dry_run:continue
        args.report_dir.mkdir(parents=True,exist_ok=True)
        ret=subprocess.run(cmd,cwd=ROOT).returncode
        if ret:raise RuntimeError(f'Evaluation exited {ret}: {checkpoint}')
        result=json.loads(out.read_text(encoding='utf-8'))
        rows.append({'run':run.name,'checkpoint':str(checkpoint),'test_seed':args.seed,
                     'episodes':result['episodes'],'success_rate':result['success_rate'],
                     **{'failed_'+k:v for k,v in result['failures'].items()},
                     'mean_capped_time':result.get('mean_capped_time'),
                     'mean_capture_time_success':result.get('mean_reach_time_success')})
    if rows:
        with (args.report_dir/'independent_summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(f'Saved {args.report_dir/"independent_summary.csv"}')
    return 0

if __name__=='__main__':raise SystemExit(main())
