"""Independent non-flight benchmark: frozen policy finite-horizon value probes.
No PPO update, policy export, or application integration. Labels are independent
Monte Carlo continuation averages, not self-bootstrapped targets.
"""
import argparse, copy, hashlib, json, csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from v16.envs import MultiLinear
from v17.audit import Archive, Frozen, state_digest, write, math_audit

GAMMA=.99
TIMES=(0,25,50,75)
PRESETS={'smoke':(4,4,40),'quick':(24,12,400),'full':(64,32,800)}

def step(x,v,a):
    a=np.clip(a,-1,1)
    r=-np.mean(x*x+.1*v*v+.01*a*a,axis=-1)
    v=v+.05*(3*a-.4*v)
    x=x+.05*v
    return x,v,r

def obs(x,v): return np.concatenate([x,v,np.zeros_like(x),np.zeros_like(v)],axis=-1).astype(np.float32)

@torch.no_grad()
def act(model,x,v,rng):
    o=torch.from_numpy(obs(x,v)); z=rng.normal(size=x.shape).astype(np.float32)
    return (model.actor(o)+model.log_std.clamp(-5,2).exp()*torch.from_numpy(z)).tanh().numpy().astype(np.float64)

@torch.no_grad()
def anchors(model,n,seed):
    rng=np.random.default_rng(seed)
    # Reset distribution exactly matches MultiLinear; independent initial states.
    x=np.stack([rng.uniform(-1,1,16) for _ in range(n)])
    v=np.stack([rng.uniform(-.5,.5,16) for _ in range(n)])
    X=[]; V=[]; T=[]; ids=[]
    for t in range(100):
        if t in TIMES:
            X.append(x.copy()); V.append(v.copy()); T.extend([t]*n); ids.extend(range(n))
        x,v,_=step(x,v,act(model,x,v,rng))
    return np.concatenate(X),np.concatenate(V),np.array(T),np.array(ids)

@torch.no_grad()
def mc_labels(model,x,v,t,k,seed):
    rng=np.random.default_rng(seed); n=len(t)
    x=np.repeat(x,k,axis=0); v=np.repeat(v,k,axis=0); tt=np.repeat(t,k)
    returns=np.zeros(n*k); disc=1.
    for i in range(100):
        active=tt+i<100
        if not np.any(active): break
        xx,vv,r=step(x[active],v[active],act(model,x[active],v[active],rng))
        x[active]=xx; v[active]=vv; returns[active]+=disc*r; disc*=GAMMA
    rr=returns.reshape(n,k)
    return rr.mean(1),rr.std(1,ddof=1)/np.sqrt(k)

def dataset(model,n,k,seed):
    x,v,t,ids=anchors(model,n,seed); y,se=mc_labels(model,x,v,t,k,seed+1)
    return obs(x,v),t,ids,y,se

def features(o,t,aware):
    tau=(100-t)/100 if aware else np.zeros(len(t))
    return torch.from_numpy(np.column_stack([o,tau]).astype(np.float32))

def probe(model,train,aware,epochs,seed):
    # Same initialization, dimensions, optimizer and batches for both arms.
    net=copy.deepcopy(model.critic)
    old=net[0]; first=nn.Linear(old.in_features+1,old.out_features)
    with torch.no_grad():
        first.weight.zero_(); first.weight[:,:-1].copy_(old.weight); first.bias.copy_(old.bias)
    net[0]=first
    o,t,_,y,_=train; x=features(o,t,aware); y=torch.tensor(y,dtype=torch.float32)
    opt=torch.optim.Adam(net.parameters(),lr=.001); rng=np.random.default_rng(seed)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/64)))):
            opt.zero_grad(); loss=(net(x[idx]).squeeze(-1)-y[idx]).square().mean(); loss.backward(); opt.step()
    return net.eval()

@torch.no_grad()
def evaluate(model,net,data,aware):
    o,t,ids,y,se=data
    p=(model.value(torch.from_numpy(o)) if net is None else net(features(o,t,aware)).squeeze(-1)).numpy()
    def metric(mask):
        err=p[mask]-y[mask]
        return dict(rmse=float(np.sqrt(np.mean(err**2))),bias=float(np.mean(err)),mc_label_se_rms=float(np.sqrt(np.mean(se[mask]**2))),n=int(mask.sum()))
    return {'all':metric(np.ones(len(t),bool)),**{str(a):metric(t==a) for a in TIMES}},p

def run_seed(ar,seed,preset,out):
    n,k,epochs=PRESETS[preset]; raw=ar.read(f'linear16_custom_{seed}','last.pt')
    model=Frozen('linear16','custom',raw,seed).model; before=state_digest(model)
    train=dataset(model,n,k,180000+seed*100); test=dataset(model,n,k,280000+seed*100)
    # Alternate independently sampled labels on exactly the same held-out states
    # to estimate label noise without reusing continuation random numbers.
    x=test[0][:,:16]; v=test[0][:,16:32]
    # Use original double-precision anchor states, not float32 reconstruction.
    xx,vv,tt,_=anchors(model,n,280000+seed*100)
    y2,se2=mc_labels(model,xx,vv,tt,k,380000+seed*100)
    rows={}; preds={}
    for arm in ('frozen','critic_only','time_aware'):
        aware=arm=='time_aware'; net=None if arm=='frozen' else probe(model,train,aware,epochs,seed)
        tr,_=evaluate(model,net,train,aware); te,p=evaluate(model,net,test,aware)
        alt,_=evaluate(model,net,(test[0],test[1],test[2],y2,se2),aware)
        rows[arm]={'train':tr,'test':te,'independent_label_repeat':alt}; preds[arm]=p
    assert state_digest(model)==before
    result=dict(seed=seed,preset=preset,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),model_unchanged=True,anchor_seeds_per_split=n,continuations_per_anchor=k,epochs=epochs,arms=rows,label_repeat_rmse=float(np.sqrt(np.mean((test[3]-y2)**2))),target='finite-horizon discounted return of frozen stochastic policy; zero tail at 100 steps',scope='independent abstract linear16; value probes not used by policy or PPO')
    write(out/f'seed_{seed}.json',result)
    with (out/f'anchors_{seed}.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['episode_cluster','time','mc_mean','mc_se','mc_repeat','mc_repeat_se',*preds]); w.writerows(zip(test[2],test[1],test[3],test[4],y2,se2,*preds.values()))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--archive'); p.add_argument('--out',required=True); p.add_argument('--preset',choices=PRESETS,default='quick'); p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]); p.add_argument('--resume',action='store_true'); p.add_argument('--status',action='store_true'); a=p.parse_args(); torch.set_num_threads(1); out=Path(a.out)
    if a.status: print((out/'progress.json').read_text()); return
    if not a.archive: p.error('--archive required')
    if len(set(a.seeds))!=len(a.seeds) or not set(a.seeds)<=set(range(42,47)): p.error('unique seeds 42–46 required')
    out.mkdir(parents=True,exist_ok=True); manifest={'archive_sha256':hashlib.sha256(Path(a.archive).read_bytes()).hexdigest(),'preset':a.preset,'seeds':a.seeds,'protocol':1}; mf=out/'manifest.json'
    if mf.exists():
        if not a.resume or json.loads(mf.read_text())!=manifest: raise ValueError('use --resume with same manifest or new output')
    elif any(out.iterdir()): raise ValueError('nonempty output')
    else: write(mf,manifest)
    m=math_audit(); write(out/'math.json',m); assert m['passed']; ar=Archive(a.archive); results=[]
    try:
        for s in a.seeds:
            path=out/f'seed_{s}.json'; reused=a.resume and path.exists() and (out/f'anchors_{s}.csv').exists()
            r=json.loads(path.read_text()) if reused else run_seed(ar,s,a.preset,out); results.append(r)
            write(out/'progress.json',{'complete':len(results),'total':len(a.seeds),'seed':s,'reused':bool(reused)}); print(f'[{len(results)}/{len(a.seeds)}] seed {s}: '+str({k:round(v['test']['all']['rmse'],4) for k,v in r['arms'].items()}),flush=True)
    finally: ar.close()
    write(out/'summary.json',results)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['seed','arm','train_rmse','test_rmse','repeat_rmse','test_bias','label_repeat_rmse']);
        for r in results:
            for k,v in r['arms'].items(): w.writerow([r['seed'],k,v['train']['all']['rmse'],v['test']['all']['rmse'],v['independent_label_repeat']['all']['rmse'],v['test']['all']['bias'],r['label_repeat_rmse']])
if __name__=='__main__': main()
