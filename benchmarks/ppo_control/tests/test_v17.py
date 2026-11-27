import numpy as np
import torch
from v17.audit import math_audit,returns,stats,calibration,episode,Frozen,state_digest
from v16.core import Policy
import io

def test_math(): assert math_audit()['passed']
def test_returns(): assert np.allclose(returns([1,2],.9),[2.8,2]); assert np.allclose(returns([1,2],.9,3),[5.23,4.7])
def test_stats(): assert stats([1,2],[1,2])==dict(samples=2,bias=0.,rmse=0.,ev=1.)
def test_frozen_repeat():
    p=Policy(64,16); b=io.BytesIO(); torch.save(dict(name='linear16',seed=42,model=p.state_dict()),b); f=Frozen('linear16','custom',b.getvalue(),42); h=state_digest(f.model)
    for stochastic in (False,True):
        a=episode('linear16',f,1,stochastic); b=episode('linear16',f,1,stochastic); assert a['trace_sha256']==b['trace_sha256']; assert a['truncated']; assert len(a['mc'])==100; assert calibration([a])['all']['mc']['samples']==100
    assert state_digest(f.model)==h

def test_paired():
    from v17.archive_report import paired
    import pytest
    a={'episodes':[{'seed':3,'return_':1},{'seed':2,'return_':2}]}; b={'episodes':[{'seed':2,'return_':4},{'seed':3,'return_':0}]}
    assert paired(a,b)==dict(n=2,mean=.5,improved=1,worse=1,equal=0)
    with pytest.raises(ValueError): paired(a,{'episodes':[{'seed':3,'return_':1}]})
