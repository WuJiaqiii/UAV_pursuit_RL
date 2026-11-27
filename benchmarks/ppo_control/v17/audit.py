"""Read-only checkpoint diagnostics for three fixed non-flight benchmarks."""
import argparse, hashlib, io, json, csv, os, tempfile, zipfile
from pathlib import Path
import numpy as np
import torch
from v16.core import Policy, gae, clipped_objective
from v16.envs import make_env, DependencyBlocked

ENVS=('linear16','doublependulum','hopper')
BACKENDS=('custom','reference')
STAGES=('initial','best','last')

def write(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); os.replace(tmp,path)

def digest(data): return hashlib.sha256(data).hexdigest()
def state_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Archive:
    def __init__(self,path):
        self.z=zipfile.ZipFile(path); self.members=set(self.z.namelist())
        manifests=[n for n in self.members if n.endswith('/manifest.json') and not n.startswith('__MACOSX/') and n.count('/')==1]
        if len(manifests)!=1: raise ValueError('expected exactly one top-level V16 manifest')
        self.root=manifests[0].rsplit('/',1)[0]; self.manifest=json.loads(self.z.read(manifests[0]))
    def read(self,job,file): return self.z.read(f'{self.root}/{job}/{file}')
    def close(self): self.z.close()

def math_audit():
    rng=np.random.default_rng(1701); shape=(7,3)
    r,v,nv=[rng.normal(size=shape) for _ in range(3)]; term=np.zeros(shape,bool); trunc=term.copy(); term[2,0]=1; trunc[4,1]=1; term[6,2]=1
    g,l=.99,.95; delta=r+g*nv*(~term)-v; expected=np.zeros(shape)
    # Explicit summation, independent of the recursive implementation.
    for t in range(shape[0]):
        for e in range(shape[1]):
            for k in range(t,shape[0]):
                expected[t,e]+=(g*l)**(k-t)*delta[k,e]
                if term[k,e] or trunc[k,e]: break
    a,ret=gae(*[torch.tensor(x) for x in (r,v,nv,term,trunc)],g,l)
    err=float(np.max(np.abs(a.numpy()-expected))); re=float(np.max(np.abs(ret.numpy()-(expected+v))))
    new=torch.tensor([-.4,-.1,.1,.4],dtype=torch.float64,requires_grad=True); old=torch.zeros_like(new); adv=torch.tensor([1.,-2.,3.,-4.],dtype=new.dtype)
    loss,_=clipped_objective(new,old,adv); ratio=np.exp(new.detach().numpy()); exact=-np.minimum(ratio*adv.numpy(),np.clip(ratio,.8,1.2)*adv.numpy()).mean(); loss.backward()
    f=lambda z:-np.minimum(np.exp(z)*adv.numpy(),np.clip(np.exp(z),.8,1.2)*adv.numpy()).mean()
    x=new.detach().numpy(); eye=np.eye(len(x))*1e-6; grad=np.array([(f(x+d)-f(x-d))/2e-6 for d in eye]); ge=float(np.max(np.abs(grad-new.grad.numpy())))
    p=Policy(64,16).double(); obs=torch.tensor(rng.normal(size=(8,64)),dtype=torch.float64); z=torch.tensor(rng.normal(size=(8,16)),dtype=torch.float64); normal=p.dist(obs); ref=(normal.log_prob(z)-torch.log1p(-z.tanh().square())).sum(-1); pe=float((p.logp(obs,z)-ref).abs().max().detach())
    metrics=dict(gae_max_error=err,return_max_error=re,loss_error=float(abs(float(loss.detach())-exact)),gradient_max_error=ge,joint_logp_error=pe)
    return dict(metrics,passed=all(v<1e-6 for v in metrics.values()),tolerance=1e-6,scope='Synthetic fixed tensors; not a cross-backend training equivalence test.')

class Frozen:
    def __init__(self,env,backend,raw,seed):
        self.backend=backend
        e=make_env(env,seed)
        try:
            if backend=='custom':
                m=Policy(e.obs_dim,e.action_dim); ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                if ck.get('name')!=env or ck.get('seed')!=seed: raise ValueError('checkpoint metadata mismatch')
                m.load_state_dict(ck['model']); self.model=m.eval()
            else:
                try: from stable_baselines3 import PPO
                except ImportError as ex: raise DependencyBlocked(str(ex)) from ex
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'model.zip'; path.write_bytes(raw); self.model=PPO.load(path,device='cpu').policy.eval()
                if self.model.observation_space.shape!=(e.obs_dim,) or self.model.action_space.shape!=(e.action_dim,): raise ValueError('SB3 checkpoint shape mismatch')
        finally: e.close()
    @torch.no_grad()
    def action_value(self,obs,stochastic):
        x=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        if self.backend=='custom':
            a=self.model.sample(x)[0] if stochastic else self.model(x)
            return a[0].numpy(),float(self.model.value(x)[0])
        a,_=self.model.predict(obs,deterministic=not stochastic)
        return a,float(self.model.predict_values(x).reshape(-1)[0])

def returns(rewards,gamma=.99,tail=0.):
    out=np.empty(len(rewards)); carry=tail
    for i in reversed(range(len(out))): carry=float(rewards[i])+gamma*carry; out[i]=carry
    return out

def episode(env,policy,seed,stochastic):
    e=make_env(env,seed); obs=e.reset(seed); rs=[]; vs=[]; actions=[]; errors=[]; h=hashlib.sha256()
    try:
        # Isolate random action draws per episode and keep process RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+900000)
            for _ in range(e.horizon):
                a,v=policy.action_value(obs,stochastic); vs.append(v); actions.append(np.asarray(a)); h.update(obs.tobytes()); h.update(np.asarray(a).tobytes())
                obs,r,term,trunc,info=e.step(a); rs.append(r)
                if info.get('error') is not None: errors.append(info['error'])
                if term or trunc: break
            if not (term or trunc): raise RuntimeError('episode horizon mismatch')
            _,tail=policy.action_value(obs,False)
        h.update(obs.tobytes()); h.update(np.asarray(rs,dtype=np.float64).tobytes()); h.update(bytes([term,trunc]))
        mc=returns(rs); boot=returns(rs,tail=tail if trunc and not term else 0.)
        a=np.stack(actions); v=np.asarray(vs)
        return dict(seed=seed,return_=float(sum(rs)),discounted_return=float(mc[0]),steps=len(rs),terminated=bool(term),truncated=bool(trunc),error=float(np.mean(errors)) if errors else None,action_std=np.std(a,axis=0).tolist(),action_saturation=(np.abs(a)>=.98).mean(axis=0).tolist(),trace_sha256=h.hexdigest(),value=v,mc=mc,bootstrap=boot)
    finally: e.close()

def stats(pred,target):
    e=np.asarray(pred)-np.asarray(target); var=np.var(target)
    return dict(samples=int(e.size),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),ev=float(1-np.var(e)/var) if var>1e-12 else None)

def calibration(rows):
    def cat(k,part): return np.concatenate([np.array_split(r[k],3)[part] for r in rows]) if part is not None else np.concatenate([r[k] for r in rows])
    result={}
    for name,part in [('all',None),('early',0),('middle',1),('late',2)]:
        v=cat('value',part)
        if not len(v): continue
        result[name]={k:stats(v,cat(k,part)) for k in ('mc','bootstrap')}
    return result

def public(row): return {k:v for k,v in row.items() if k not in ('value','mc','bootstrap')}

def replay(env,backend,seed,raw,episodes):
    policy=Frozen(env,backend,raw,seed); before=state_digest(policy.model)
    det=[episode(env,policy,510000+i,False) for i in range(episodes)]
    repeat=[episode(env,policy,510000+i,False) for i in range(episodes)]
    stoch=[episode(env,policy,610000+i,True) for i in range(episodes)]
    after=state_digest(policy.model)
    return dict(checkpoint_sha256=digest(raw),model_unchanged=before==after,deterministic_repeat_exact=all(a['trace_sha256']==b['trace_sha256'] for a,b in zip(det,repeat)),deterministic=[public(r) for r in det],stochastic=[public(r) for r in stoch],stochastic_calibration=calibration(stoch),interpretation='Finite-horizon MC is a realized return, not an exact estimate of continuing-task V at truncation. Bootstrap target uses the same frozen critic and is not independent ground truth. Samples within episodes are correlated.')

def log_audit(result,rows,backend):
    s=np.array([r['decision_steps'] for r in rows]); v=np.array([r['validation_return'] for r in rows]); peak=int(np.argmax(v));
    evkey='explained_variance' if backend=='custom' else 'train/explained_variance'
    ev=[r[evkey] for r in rows[-32:] if r.get(evkey) is not None]
    return dict(rows=len(rows),monotonic_steps=bool(np.all(np.diff(s)>0)),initial_validation=result['initial_validation']['return_'],best_update=result['best_update'],peak_logged_step=int(s[peak]),peak_logged_return=float(v[peak]),last_logged_return=float(v[-1]),late_ev=float(np.mean(ev)) if ev else None,kl_stop_count=sum(bool(r.get('kl_stop')) for r in rows) if backend=='custom' else None,initial_test=result['initial_test']['return_'],best_test=result['best_test']['return_'],last_test=result['final_test']['return_'])

def campaign(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); ar=Archive(args.archive)
    try:
        envs=ENVS if args.scope=='all' else ('linear16',); seeds=[42] if args.preset=='smoke' else [42,43,44] if args.preset=='quick' else list(range(42,47)); episodes={'smoke':2,'quick':8,'full':24}[args.preset]; backends=BACKENDS if args.backend=='both' else (args.backend,)
        jobs=[(e,b,s) for e in envs for b in backends for s in seeds]
        protocol=dict(scope=args.scope,preset=args.preset,backend=args.backend,episodes=episodes,archive_sha256=digest(Path(args.archive).read_bytes()),jobs=[list(j) for j in jobs],deterministic_seed=510000,stochastic_seed=610000,version=1)
        mf=out/'manifest.json'
        if mf.exists():
            if not args.resume or json.loads(mf.read_text())!=protocol: raise ValueError('output exists; use --resume with identical manifest or choose new output')
        elif any(out.iterdir()): raise ValueError('output not empty')
        else: write(mf,protocol)
        math=math_audit(); write(out/'math.json',math)
        if not math['passed']: raise RuntimeError('mathematical gate failed')
        statuses=[]; summaries=[]
        total=len(jobs)*len(STAGES)
        for e,b,s in jobs:
            j=f'{e}_{b}_{s}'; d=out/j; d.mkdir(exist_ok=True); result=json.loads(ar.read(j,'result.json')); rows=[json.loads(l) for l in ar.read(j,'rollout.jsonl').splitlines() if l]; la=log_audit(result,rows,b); write(d/'archive_audit.json',la)
            for stage in STAGES:
                file=d/f'{stage}.json'; status='complete'; error=None; reused=False
                try:
                    if args.resume and file.exists(): data=json.loads(file.read_text()); reused=True
                    else:
                        raw=ar.read(j,stage+('.pt' if b=='custom' else '.zip')); data=replay(e,b,s,raw,episodes); write(file,data)
                    if not data['model_unchanged'] or not data['deterministic_repeat_exact']: status='review'
                    det=data['deterministic']; sto=data['stochastic']; summaries.append(dict(env=e,backend=b,seed=s,checkpoint=stage,det_return=float(np.mean([r['return_'] for r in det])),stoch_return=float(np.mean([r['return_'] for r in sto])),mean_steps=float(np.mean([r['steps'] for r in det])),mc_rmse=data['stochastic_calibration']['all']['mc']['rmse'],mc_bias=data['stochastic_calibration']['all']['mc']['bias'],repeat_exact=data['deterministic_repeat_exact'],model_unchanged=data['model_unchanged']))
                except DependencyBlocked as ex: status='blocked'; error=str(ex)
                except Exception as ex: status='failed'; error=f'{type(ex).__name__}: {ex}'
                statuses.append(dict(job=j,checkpoint=stage,status=status,error=error,reused=reused)); write(out/'progress.json',dict(processed=len(statuses),total=total,items=statuses)); print(f'[{len(statuses)}/{total}] {j}/{stage} {status}',flush=True)
        fields=['env','backend','seed','checkpoint','det_return','stoch_return','mean_steps','mc_rmse','mc_bias','repeat_exact','model_unchanged']
        with (out/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summaries)
        write(out/'summary.json',dict(counts={k:sum(r['status']==k for r in statuses) for k in ('complete','review','blocked','failed')},rows=summaries,scope='read-only checkpoint diagnosis; no actor or critic optimization'))
        return 1 if any(r['status'] in ('failed','review') for r in statuses) else 2 if any(r['status']=='blocked' for r in statuses) else 0
    finally: ar.close()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--archive'); p.add_argument('--out',required=True); p.add_argument('--scope',choices=['linear','all'],default='all'); p.add_argument('--preset',choices=['smoke','quick','full'],default='quick'); p.add_argument('--backend',choices=['custom','reference','both'],default='both'); p.add_argument('--resume',action='store_true'); p.add_argument('--status',action='store_true'); a=p.parse_args(); torch.set_num_threads(1)
    if a.status: print(Path(a.out,'progress.json').read_text()); return 0
    if not a.archive: p.error('--archive required')
    return campaign(a)
if __name__=='__main__': raise SystemExit(main())
