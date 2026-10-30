#!/usr/bin/env python3
"""Reproducible, sequential long-run PPO ablation launcher (one GPU).

Never compares models trained under different Stage0 configs as a controlled
ablation; 'easy' and 'standard' are separate cohorts. Preserves run/config.yaml
and refuses to overwrite completed/nonempty results. Optional --init-from uses
ONE shared checkpoint for every selected run; report these as fine-tuning.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PLANS={
    'easy': {
        'easy_pure': 's0_easy_pure.yaml',
        'easy_pd02': 's0_easy_pd02.yaml',
        'easy_pd05': 's0_easy_pd05.yaml',
    },
    'standard': {
        'standard_pure': 's0_standard_pure.yaml',
        'standard_pd02': 's0_standard_pd02.yaml',
        'standard_pd05': 's0_standard_pd05.yaml',
        'standard_pd02_no_progress': 's0_standard_pd02_no_progress.yaml',
        'standard_pd02_gae_long': 's0_standard_pd02_gae_long.yaml',
        'standard_pd02_lr_low': 's0_standard_pd02_lr_low.yaml',
        'standard_pd02_linear': 's0_standard_pd02_linear.yaml',
    },
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group',choices=['easy','standard','all'],default='easy')
    ap.add_argument('--only',nargs='*',help='subset of experiment names')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--backend',choices=['torch','numpy'],default='torch')
    ap.add_argument('--steps',type=int,help='override training.total_steps for EVERY run')
    ap.add_argument('--num-envs',type=int,help='override num_envs for EVERY run')
    ap.add_argument('--init-from',type=Path,help='same compatible checkpoint for ALL selected runs')
    ap.add_argument('--out-root',type=Path,default=ROOT/'runs'/'basic_v67_matrix')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--keep-going',action='store_true',help='continue after a failed run')
    args=ap.parse_args(argv)
    if args.steps is not None and args.steps<1:ap.error('--steps must be positive')
    if args.num_envs is not None and args.num_envs<1:ap.error('--num-envs must be positive')
    init_path=args.init_from.resolve() if args.init_from else None
    if init_path is not None and not init_path.is_file():ap.error('--init-from not found')
    plans={}
    for group in (('easy','standard') if args.group=='all' else (args.group,)):
        plans.update(PLANS[group])
    if args.only:
        unknown=set(args.only)-set(plans)
        if unknown:ap.error(f'Unknown experiment(s) in chosen group: {sorted(unknown)}')
        plans={k:v for k,v in plans.items() if k in args.only}
    if not plans:ap.error('No experiments selected')
    # Use literal plan file names, no dynamic Python import of arbitrary code.
    commands=[]
    for name,config_name in plans.items():
        config=(ROOT/'configs'/'basic_v67'/config_name).resolve()
        if not config.is_file():ap.error(f'Missing config {config}')
        for seed in args.seeds:
            out=(args.out_root/f'{name}_seed{seed}').resolve()
            cmd=[sys.executable,str(ROOT/'main.py'),'train','--config',str(config),
                 '--out',str(out),'--device',args.device,'--simulation-backend',args.backend,
                 '--seed',str(seed)]
            if args.steps is not None:cmd += ['--total-steps',str(args.steps)]
            if args.num_envs is not None:cmd += ['--num-envs',str(args.num_envs)]
            if init_path is not None:cmd += ['--init-from',str(init_path)]
            commands.append(dict(name=name,seed=seed,out=out,config=config,cmd=cmd))
    for item in commands:
        print(shlex.join(item['cmd']),flush=True)
    if args.dry_run:
        print(f'\nDRY RUN: {len(commands)} separate train processes, no files modified.')
        return 0
    args.out_root.mkdir(parents=True,exist_ok=True)
    failed=[]; skipped=[]
    for idx,item in enumerate(commands,1):
        out=item['out']
        if out.exists() and any(out.iterdir()):
            print(f'[{idx}/{len(commands)}] SKIP: existing output {out}',flush=True)
            skipped.append(str(out));continue
        metadata={'plan':item['name'],'seed':item['seed'],
                  'config_source':str(item['config']),
                  'config_file_sha256':sha256(item['config']),
                  'runner_script_sha256':sha256(__file__),
                  'init_checkpoint':str(init_path) if init_path else None,
                  'init_checkpoint_sha256':sha256(init_path) if init_path else None,
                  'cmd':item['cmd'],
                  'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'note':'Training-only PD teacher; inference evaluates pure PPO actor.'}
        # Trainer refuses preexisting nonempty output directory; place manifest beside it.
        manifest=out.with_suffix('.launch.json')
        manifest.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'[{idx}/{len(commands)}] START {item["name"]}, seed={item["seed"]}',flush=True)
        # stdout follows the console; use tmux/tee to persist entire launcher output.
        ret=subprocess.run(item['cmd'],cwd=ROOT).returncode
        metadata['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata['returncode']=ret
        manifest.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        if ret:
            failed.append((item['name'],item['seed'],ret))
            print(f'FAILED: {failed[-1]}',file=sys.stderr,flush=True)
            if not args.keep_going:break
    print('Summary:',json.dumps(dict(failed=failed,skipped=skipped,scheduled=len(commands)),ensure_ascii=False))
    return 1 if failed else 0

if __name__=='__main__':raise SystemExit(main())
