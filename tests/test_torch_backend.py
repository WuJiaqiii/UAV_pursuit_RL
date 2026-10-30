"""Torch-backend functional checks; runnable on CPU-only CI hosts."""
import numpy as np
import torch
from pursuit_rl.config import load_config
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.control.torch_guidance import TorchGuidanceController
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.learning.buffers import RolloutBuffer,TensorRolloutBuffer
from pursuit_rl.learning.checkpoints import load_model


def config():
    return load_config('configs/high_frequency/smoke_torch_1khz.yaml')


def _match_state(numpy_env,tensor_env):
    for src,dst in ((numpy_env.pursuer,tensor_env.pursuer),(numpy_env.target,tensor_env.target),(numpy_env,tensor_env)):
        for key,value in vars(src).items():
            if isinstance(value,np.ndarray) and hasattr(dst,key) and isinstance(getattr(dst,key),torch.Tensor):
                setattr(dst,key,torch.as_tensor(value,device=tensor_env.device,dtype=getattr(dst,key).dtype).clone())
    # Force straight segment (no probabilistic movement on the comparison window).
    numpy_env.target.mode[:]=0; tensor_env.target.mode[:]=0
    numpy_env.target.phase_length[:]=100;tensor_env.target.phase_length[:]=100


def test_torch_matches_numpy_physics_observations_and_reward():
    c=config();a=PursuitBatch(c,n=3,seed=37,randomize=False)
    b=TorchPursuitBatch(c,n=3,seed=42,randomize=False,device='cpu',dtype=torch.float64)
    _match_state(a,b)
    np.testing.assert_allclose(a.observe()[0],b.observe()[0].numpy(),atol=1e-6)
    for k in range(12):
        act=np.random.default_rng(k).uniform(-.5,.5,(3,4))
        (_, _),r,done,trunc,_=a.step(act)
        (_, _),rt,td,tt,_=b.step(torch.tensor(act,dtype=torch.float64))
        np.testing.assert_allclose(a.pursuer.pos,b.pursuer.pos.numpy(),rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(a.target.pos,b.target.pos.numpy(),rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(r,rt.numpy(),rtol=1e-4,atol=1e-4)
        np.testing.assert_array_equal(done,td.numpy())
        np.testing.assert_array_equal(trunc,tt.numpy())


def test_torch_guidance_matches_numpy():
    c=config();a=PursuitBatch(c,n=4,seed=82,randomize=False)
    b=TorchPursuitBatch(c,n=4,seed=83,randomize=False,device='cpu',dtype=torch.float64)
    _match_state(a,b)
    for alg in ('PN','APN','PD'):
        direct=GuidanceController(alg).act(a,a.observe()[0])
        tensor=TorchGuidanceController(alg).act(b,b.observe()[0]).numpy()
        np.testing.assert_allclose(direct,tensor,atol=1e-5)


def test_tensor_gae_matches_numpy():
    a=RolloutBuffer(11,3,31,43)
    b=TensorRolloutBuffer(11,3,31,43,'cpu')
    rng=np.random.default_rng(10)
    for key in ('reward','values','next_values'):
        x=rng.normal(size=(11,3)).astype(np.float32)
        setattr(a,key,x);setattr(b,key,torch.as_tensor(x))
    for key in ('terminated','truncated'):
        x=rng.random((11,3))<.1
        setattr(a,key,x);setattr(b,key,torch.as_tensor(x))
    a.compute_gae(.99,.95);b.compute_gae(.99,.95)
    np.testing.assert_allclose(a.advantage,b.advantage.numpy(),atol=2e-6)
    np.testing.assert_allclose(a.returns,b.returns.numpy(),atol=2e-6)


def test_torch_evaluation_with_disturbances_and_trace(tmp_path):
    c=config(); model,_=load_model('examples/v5_legacy_best.pt')
    disturb={'position_std_m':.5,'observation_delay_s':.002,'command_delay_s':.003}
    report=evaluate(model,c,stage=0,episodes=2,seed=90,disturbance=disturb,trace_dir=tmp_path)
    assert report['episodes']==2
    assert report['evaluation_protocol']['simulation_backend']=='torch'
    assert len(list(tmp_path.glob('episode_*.npz')))==2


def test_backend_preserves_pairing():
    c=config();a=evaluate(GuidanceController('PN'),c,stage=0,episodes=3,seed=99)
    b=evaluate(GuidanceController('APN'),c,stage=0,episodes=3,seed=99)
    for x,y in zip(a['episodes_detail'],b['episodes_detail']):
        np.testing.assert_allclose(x['start_rel_pos'],y['start_rel_pos'])
        np.testing.assert_allclose(x['start_target_vel'],y['start_target_vel'])
