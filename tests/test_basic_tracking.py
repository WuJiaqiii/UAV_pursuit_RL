"""Stage0 and optional dense progress reward, with NumPy/Torch physics parity."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from pursuit_rl.config import load_config
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.learning.buffers import TensorRolloutBuffer


def test_basic_presets_are_genuine_1khz_and_stage0_only():
    c=load_config('configs/basic/basic_stage0_1khz.yaml')
    assert c['physics']['dt']==0.001 and c['physics']['policy_interval_steps']==1
    assert c['training']['initial_stage']==c['training']['max_stage']==0
    assert c['curriculum'][0]['modes']==['straight']
    assert c['reward']['w_progress']>0 and c['training']['log_std_min']<c['training']['log_std_init']
    legacy=load_config('configs/high_frequency/ppo_1khz.yaml')
    assert legacy['reward']['w_progress']==0
    moving=load_config('configs/basic/basic_moving_1khz.yaml')
    assert moving['curriculum'][0]['speed'][1]>c['curriculum'][0]['speed'][1]


def test_dense_progress_tracks_actual_distance_reduction_numpy_and_torch():
    c=load_config('configs/basic/basic_stage0_1khz.yaml')
    x=PursuitBatch(c,n=3,seed=11,randomize=False)
    y=TorchPursuitBatch(c,n=3,seed=12,randomize=False,device='cpu',dtype=torch.float64)
    # Equalize all numeric state and derived baselines, not random samples.
    for objx,objy in ((x.pursuer,y.pursuer),(x.target,y.target),(x,y)):
        for key,value in vars(objx).items():
            if isinstance(value,np.ndarray) and isinstance(getattr(objy,key,None),torch.Tensor):
                setattr(objy,key,torch.as_tensor(value,device='cpu',dtype=getattr(objy,key).dtype).clone())
    x.target.mode[:]=0; y.target.mode[:]=0
    act=np.zeros((3,4))
    prior=np.linalg.norm(x.target.pos-x.pursuer.pos,axis=-1)
    x.step(act)
    y.step(torch.as_tensor(act,dtype=torch.float64))
    after=np.linalg.norm(x.target.pos-x.pursuer.pos,axis=-1)
    expect=c['reward']['w_progress']*(prior-after)/c['reward']['progress_scale_m']
    np.testing.assert_allclose(x.reward_components['progress'],expect,atol=1e-8)
    np.testing.assert_allclose(y.reward_components['progress'].numpy(),expect,atol=1e-5)


def test_logstd_training_floor_is_effective():
    c=load_config('configs/basic/basic_stage0_1khz.yaml')['training']
    model=AsymmetricPPO(31,43,c,device='cpu')
    with torch.no_grad():model.log_std.fill_(-3.0)
    buf=TensorRolloutBuffer(2,2,31,43,'cpu')
    buf.obs.zero_();buf.priv.zero_();buf.raw.zero_();buf.logprob.zero_()
    buf.advantage.fill_(1.);buf.returns.zero_()
    metrics=model.update(buf)
    assert model.log_std.min().item()>=c['log_std_min']-1e-6
    assert 'explained_variance' in metrics


def test_invalid_exploration_floor_is_rejected():
    from pursuit_rl.config.schema import validate
    c=load_config('configs/basic/basic_stage0_1khz.yaml')
    c['training']['log_std_min']=0.0
    with pytest.raises(ValueError,match='log_std_min'):validate(c)
