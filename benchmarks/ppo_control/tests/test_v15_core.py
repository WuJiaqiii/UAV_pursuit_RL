import numpy as np
import torch
import pytest
from v15.core import gae,Policy,clipped_objective
from v15.envs import Bench
from v15.eval import evaluate,model_action

def test_gae_hand():
    r=torch.tensor([[1.],[2.]]); z=torch.zeros_like(r); a,ret=gae(r,z,z,z,z,.9,.8); assert torch.allclose(a,torch.tensor([[2.44],[2.]])); assert torch.equal(a,ret)
def test_boundaries():
    r=torch.tensor([[1.,1.],[10.,10.]]); z=torch.zeros_like(r); nv=torch.ones_like(r)*2; term=z.clone(); trunc=z.clone(); term[0,0]=1; trunc[0,1]=1; a,_=gae(r,z,nv,term,trunc,.9,.8); assert torch.allclose(a[0],torch.tensor([1.,2.8]))
def test_log_prob():
    p=Policy(4); x=torch.randn(32,4); a,z,lp=p.sample(x); assert torch.allclose(lp,p.logp(x,z)); assert (a.abs()<=1).all(); expected=(p.dist(x).log_prob(z)-torch.log1p(-a.square())).sum(-1); assert torch.allclose(lp,expected,atol=1e-5)
def test_ratio_gradient():
    new=torch.tensor([0.,0.],requires_grad=True); loss,ratio=clipped_objective(new,torch.zeros(2),torch.tensor([1.,-1.])); loss.backward(); assert torch.equal(ratio,torch.ones(2)); assert torch.allclose(new.grad,torch.tensor([-.5,.5]))
def test_clip():
    new=torch.log(torch.tensor([1.5,.5])).requires_grad_(); loss,_=clipped_objective(new,torch.zeros(2),torch.tensor([1.,-1.])); loss.backward(); assert torch.equal(new.grad,torch.zeros(2))
def test_parallel_independence():
    r=torch.tensor([[1.,10.],[2.,20.]]); z=torch.zeros_like(r); a,_=gae(r,z,z,z,z,.9,1); assert torch.allclose(a[:,1],10*a[:,0]); assert torch.equal(r.flatten(),torch.tensor([1.,10.,2.,20.]))
@pytest.mark.parametrize('name',['linear','pendulum','setpoint','reference'])
def test_env(name):
    e=Bench(name,1); f=Bench(name,1); assert np.array_equal(e.reset(),f.reset());
    for i in range(e.horizon):
        a=e.step([2]); b=f.step([1]); assert np.array_equal(a[0],b[0]); assert a[1:]==b[1:]; assert a[3]==(i==e.horizon-1)
def test_checkpoint(tmp_path):
    p=Policy(4); x=torch.randn(8,4); torch.save(p.state_dict(),tmp_path/'x.pt'); q=Policy(4); q.load_state_dict(torch.load(tmp_path/'x.pt',weights_only=True)); assert torch.equal(p(x),q(x)); assert evaluate('linear',model_action(p,'cpu'),episodes=2)==evaluate('linear',model_action(q,'cpu'),episodes=2)
def test_reward_reference():
    e=Bench('linear'); assert evaluate('linear',e.reference_action)['return_']>evaluate('linear',lambda o:[0.])['return_']
