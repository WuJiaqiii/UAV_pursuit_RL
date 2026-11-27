"""Descriptive audits for independent benchmarks; no automatic causal claims."""
import numpy as np
from .envs import make_env
from .eval import evaluate

def reward_audit(name,episodes=8):
    env=make_env(name,0); n=env.action_dim; rng=np.random.default_rng(2026)
    try:
        policies={'zero':lambda o:np.zeros(n,np.float32),'fixed':lambda o:np.full(n,.5,np.float32),'random':lambda o:rng.uniform(-1,1,n).astype(np.float32)}
        if hasattr(env,'reference_action') and name!='pendulum': policies['analytic']=env.reference_action
        return {k:evaluate(name,v,episodes=episodes,seed=30000) for k,v in policies.items()}
    finally: env.close()

def summarize(rows,planned,preset,threshold=.5):
    groups=[]
    for name,backend in sorted({(j[0],j[1]) for j in planned}):
        expected=sum(j[0]==name and j[1]==backend for j in planned); rs=[r for r in rows if r['env']==name and r['backend']==backend]
        # Improvement relative to absolute initial return, not a success probability.
        gains=[(r['best']-r['initial'])/max(abs(r['initial']),1e-8) for r in rs]
        gains_final=[(r['final']-r['initial'])/max(abs(r['initial']),1e-8) for r in rs]
        status='incomplete' if len(rs)!=expected else 'smoke_only' if preset=='smoke' else 'diagnostic_only' if not name.startswith(('linear','coupled')) else 'screen_pass' if min(gains_final)>=threshold else 'screen_review'
        groups.append(dict(env=name,backend=backend,expected=expected,completed=len(rs),screen=status,threshold=threshold if name.startswith(('linear','coupled')) else None,best_mean=float(np.mean([r['best'] for r in rs])) if rs else None,best_sd=float(np.std([r['best'] for r in rs],ddof=1)) if len(rs)>1 else None,mean_relative_gain=float(np.mean(gains)) if gains else None,worst_final_relative_gain=min(gains_final) if gains_final else None,final_mean=float(np.mean([r['final'] for r in rs])) if rs else None))
    notes=[]
    for name in sorted({g['env'] for g in groups}):
        g={g['backend']:g['screen'] for g in groups if g['env']==name}
        if g.get('custom')=='screen_review' and g.get('reference')=='screen_pass': notes.append(f'{name}: inspect differences between backends; not a causal finding.')
        elif g.get('custom')=='screen_review' and g.get('reference')=='screen_review': notes.append(f'{name}: inspect common benchmark, budget, and baseline; not proof of a software bug.')
        elif 'incomplete' in g.values(): notes.append(f'{name}: missing results, no comparative inference.')
    return dict(groups=groups,notes=notes,interpretation='Linear/coupled screening: all final held-out returns improve over initialization by at least threshold, not a mastery or statistical equivalence criterion. Other environments descriptive only. Smoke never establishes learning.')
