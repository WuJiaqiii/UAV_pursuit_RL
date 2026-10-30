#!/usr/bin/env python3
"""Compact offline Stage0 diagnostic report; requires no checkpoint or traces."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    if not path.is_file():
        return []
    result=[]
    for ln in path.read_text(encoding="utf8").splitlines():
        if ln.strip():
            try:
                row=json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(row,dict):result.append(row)
    return result


def summarize(root):
    root=Path(root)
    train=read_jsonl(root/'train.jsonl')
    valid=read_jsonl(root/'validation.jsonl')
    eps=read_jsonl(root/'episodes.jsonl')
    # Training episode events may also live under per-session logs; don't silently invent them.
    if not eps:
        for path in (root/'logs').glob('*/episodes.jsonl'):
            eps.extend(read_jsonl(path))
    failure=Counter(e.get('reason','unknown') for e in eps)
    key=lambda r:r.get('stage_success',-1)
    best=max(valid,key=key) if valid else {}
    return {
        'run':str(root),
        'train_rows':len(train),'validation_rows':len(valid),'completed_episodes_logged':len(eps),
        'latest_training':({k:train[-1].get(k) for k in ('steps','stage','train_success','action_std','approx_kl','clip_fraction','explained_variance','aux_pd_coef','aux_pd_loss','train_failure_reasons','environment_steps_per_second')} if train else {}),
        'best_stage_validation':({k:best.get(k) for k in ('steps','stage','stage_success','challenge_success')} if best else {}),
        'latest_validation':({k:valid[-1].get(k) for k in ('steps','stage','stage_success','challenge_success')} if valid else {}),
        'failure_counts':dict(failure),
        'latest_reward_components':train[-1].get('reward_components') if train else None,
        'independent_challenge':valid[-1].get('challenge_independent') if valid else None,
        'basic_gate_passed':bool(best.get('stage_success',0) >= .8),
        'note':'This is a diagnostic summary, not a statistically independent model test.'
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    data=summarize(args.run)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(data,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
