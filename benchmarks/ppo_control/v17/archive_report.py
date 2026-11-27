"""Retrospective audit of archived V16 standard-control experiments, no model execution."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .audit import Archive,ENVS,BACKENDS,log_audit,write,digest

def paired(a,b):
    # Pair at episode seed, never treat the pooled episodes as independent training runs.
    x={r['seed']:r['return_'] for r in a['episodes']}; y={r['seed']:r['return_'] for r in b['episodes']}
    if len(x)!=len(a['episodes']) or len(y)!=len(b['episodes']) or x.keys()!=y.keys(): raise ValueError('unmatched or duplicate episode seeds')
    d=np.array([y[k]-x[k] for k in sorted(x)])
    return dict(n=len(d),mean=float(d.mean()),improved=int((d>0).sum()),worse=int((d<0).sum()),equal=int((d==0).sum()))

def report(archive,out):
    ar=Archive(archive); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    try:
        rows=[]; groups=[]
        for e in ENVS:
            for b in BACKENDS:
                group=[]
                for s in range(42,47):
                    j=f'{e}_{b}_{s}'; r=json.loads(ar.read(j,'result.json')); logs=[json.loads(l) for l in ar.read(j,'rollout.jsonl').splitlines() if l]; audit=log_audit(r,logs,b)
                    if r['name']!=e or r['seed']!=s: raise ValueError('result identity mismatch')
                    p=paired(r['initial_test'],r['best_test']); q=paired(r['best_test'],r['final_test']); z=paired(r['initial_test'],r['final_test'])
                    row=dict(env=e,backend=b,seed=s,**audit,best_minus_initial=p['mean'],last_minus_best=q['mean'],last_minus_initial=z['mean'],last_worse_episodes=q['worse'],paired_episodes=q['n']); rows.append(row); group.append(row)
                groups.append(dict(env=e,backend=b,training_seeds=len(group),initial_mean=float(np.mean([r['initial_test'] for r in group])),best_mean=float(np.mean([r['best_test'] for r in group])),last_mean=float(np.mean([r['last_test'] for r in group])),last_minus_best_mean=float(np.mean([r['last_minus_best'] for r in group])),last_minus_best_sd=float(np.std([r['last_minus_best'] for r in group],ddof=1)),seeds_last_worse=sum(r['last_minus_best']<0 for r in group),seeds_best_at_initial=sum(r['best_update']==0 for r in group)))
        write(out/'archive_report.json',dict(archive_sha256=digest(Path(archive).read_bytes()),groups=groups,rows=rows,scope='Descriptive paired results. Validation-selected best vs last is not a causal test. Five independent training seeds, not pooled episode-level inference.'))
        for name,data in [('seed_summary.csv',rows),('group_summary.csv',groups)]:
            with (out/name).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        print(f'{len(rows)} archived training jobs, {len(groups)} groups audited')
    finally: ar.close()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--archive',required=True); p.add_argument('--out',required=True); a=p.parse_args(); report(a.archive,a.out)
if __name__=='__main__': main()
