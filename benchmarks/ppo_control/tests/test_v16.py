import numpy as np
import pytest
import torch
from v16.core import Policy,gae,clipped_objective
from v16.envs import MultiLinear,make_env
from v16.eval import evaluate
from v16.audit import summarize

@pytest.mark.parametrize('n',[2,4,8,16])
def test_joint_logprob(n):
    p=Policy(4*n,n); x=torch.randn(9,4*n); a,z,lp=p.sample(x)
    assert a.shape==(9,n) and lp.shape==(9,)
    expected=(p.dist(x).log_prob(z)-torch.log1p(-a.square())).sum(-1)
    assert torch.allclose(lp,expected,atol=3e-5)
    assert torch.allclose(lp,p.logp(x,z))

def test_clipped_gradient_finite_difference():
    x=torch.tensor([.02,-.04,.06],dtype=torch.float64,requires_grad=True); old=torch.zeros_like(x); a=torch.tensor([1.,-2.,.3],dtype=x.dtype)
    f=lambda z:clipped_objective(z,old,a)[0]; f(x).backward(); eps=1e-6
    for i in range(3):
        d=torch.zeros_like(x); d[i]=eps; fd=(f(x.detach()+d)-f(x.detach()-d))/(2*eps)
        assert torch.allclose(x.grad[i],fd,atol=1e-8)

@pytest.mark.parametrize('name',['linear2','linear4','linear8','linear16','coupled4','coupled8'])
def test_seed_and_shapes(name):
    e=make_env(name,42); f=make_env(name,42); assert np.array_equal(e.reset(),f.reset()); assert e.obs().shape==(e.obs_dim,)
    for i in range(e.horizon):
        a=np.linspace(-.3,.3,e.action_dim); x,r,t,tr,info=e.step(a); y,q,_,tr2,_=f.step(a); assert np.array_equal(x,y) and r==q and tr==tr2; assert tr==(i==e.horizon-1); assert len(info['error_by_dim'])==e.action_dim
    with pytest.raises(ValueError): e.step(np.zeros(e.action_dim+1))

def test_discrete_dynamics_and_coupling():
    for c in (0,.6):
        e=MultiLinear(4,0,c); e.reset(); x=np.r_[e.x,e.v]; u=np.array([.4,0,0,0]); expected=e.A_d@x+e.B_d@u; e.step(u); assert np.allclose(np.r_[e.x,e.v],expected)
    e=MultiLinear(4,0,0); f=MultiLinear(4,0,.6)
    for x in (e,f): x.reset(); x.x[:]=x.v[:]=0; x.step([1,0,0,0])
    assert e.v[1]==0 and f.v[1]!=0

def test_dimension_normalized_reward():
    vals=[]
    for n in (1,4,16):
        e=MultiLinear(n); e.reset(); e.x[:]=.5; e.v[:]=.2; vals.append(e.step(np.full(n,.3))[1])
    assert np.allclose(vals,vals[0])

def test_analytic_reward_and_eval_dims():
    e=make_env('coupled4'); r=evaluate('coupled4',e.reference_action,episodes=3); z=evaluate('coupled4',lambda o:np.zeros(4),episodes=3)
    assert r['return_']>z['return_']; assert len(r['action_std_by_dim'])==4 and len(r['error_by_dim'])==4

def test_screen_smoke_and_missing():
    row=dict(env='linear4',backend='custom',initial=-10,best=-1,final=-2)
    assert summarize([row],[['linear4','custom',42]],'smoke')['groups'][0]['screen']=='smoke_only'
    assert summarize([row],[['linear4','custom',42]],'quick')['groups'][0]['screen']=='screen_pass'
    assert summarize([],[['linear4','custom',42]],'quick')['groups'][0]['screen']=='incomplete'

def test_official_pendulum_if_available():
    pytest.importorskip('gymnasium'); e=make_env('official_pendulum',0); o=e.reset(); assert o.shape==(3,); assert e.step(np.array([1.],np.float32))[0].shape==(3,); e.close()
