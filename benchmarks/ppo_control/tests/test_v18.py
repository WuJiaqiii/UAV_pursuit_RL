import numpy as np
import torch
from v16.envs import MultiLinear
from v16.core import Policy
from v18.study import step,obs,mc_labels,features,anchors,probe,dataset
from v17.audit import state_digest

def test_vector_step_matches_original():
    e=MultiLinear(16); e.reset(123); x=e.x.copy(); v=e.v.copy(); a=np.linspace(-1,1,16)
    xx,vv,r=step(x,v,a); o,rr,*_=e.step(a)
    np.testing.assert_allclose(obs(xx,vv),o,atol=1e-7); np.testing.assert_allclose(r,rr,atol=1e-12)

def test_last_step_mc():
    torch.manual_seed(1); m=Policy(64,16); x=np.zeros((2,16)); v=x.copy(); t=np.array([99,99]); y,se=mc_labels(m,x,v,t,100,1)
    assert np.all(y<=0) and np.all(y>=-.01) and np.all(se>=0)

def test_anchor_time_and_probe_isolation():
    torch.set_num_threads(1); m=Policy(64,16); before=state_digest(m); x,v,t,ids=anchors(m,2,123)
    assert t.tolist()==[0,0,25,25,50,50,75,75] and ids.tolist()==[0,1]*4
    data=dataset(m,2,3,123); probe(m,data,True,2,5); assert before==state_digest(m)
    assert features(data[0],data[1],False)[:,-1].abs().sum()==0
