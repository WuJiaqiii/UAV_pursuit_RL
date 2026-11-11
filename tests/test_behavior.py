import copy
import pytest
import torch
from pursuit_rl.ctbr.behavior import collect, fit, scan, acceptable
from pursuit_rl.ctbr.checkpoint import new_model
from pursuit_rl.ctbr.environment import CTBRBatch


def test_full_collection_keeps_features_and_completes(small):
    c = copy.deepcopy(small)
    model = new_model(c)
    (obs, labels), info = collect(model, c, 3, 111, stride=2)
    assert info['episodes'] == 3
    assert info['physics_steps'] == 360
    assert obs.shape[1] == model.obs_dim
    assert obs[:, 21:].abs().sum() > 0
    assert obs[:, 32:36].abs().sum() > 0
    assert labels.shape == (len(obs), 4)


def test_fit_uses_memory_and_freezes_nonactor(small):
    m = new_model(small)
    (obs, labels), _ = collect(m, small, 2, 111)
    before = copy.deepcopy(m.state_dict())
    with torch.no_grad():
        mean, _ = m.distribution_parameters(obs)
        features = m.bound_actor_input(m.actor_features(obs))
        expected = m.rho * features[:, 32:36] + (1-m.rho)*m.actor(features)
        assert torch.allclose(mean, expected)
    fit(m, (obs, labels), updates=3, batch_size=16)
    assert any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items() if k.startswith('actor.'))
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items() if not k.startswith('actor.'))


def test_probe_isolated_and_offsets_physically_applied(small):
    c = copy.deepcopy(small)
    c['task'].update(kind='cruise', speed=[1,1], match_initial_velocity=True)
    env = CTBRBatch(c)
    before = env.plant.pursuer.pos.clone()
    rows = scan(new_model(c), env, [0, 0.5], horizon=.002)
    assert torch.equal(before, env.plant.pursuer.pos)
    assert len(rows) == 2*env.n
    first = rows[0]
    shifted = rows[env.n]
    e0 = torch.tensor(first['position_error_m'])
    e1 = torch.tensor(shifted['position_error_m'])
    v = torch.tensor(first['target_velocity'])
    assert torch.dot(e1-e0, v) == pytest.approx(.5, abs=1e-5)
    assert set(first['branches']) == {'policy', 'teacher'}


def test_gate_requires_current_improvement_and_protects_retention():
    def report(error, joint=1):
        return dict(success_rate=1, joint_precision_success_rate=joint, mean_tail_position_rmse=error)
    before = {'mixed/1':report(1), 'hover/1':report(.1)}
    assert acceptable(before, {'mixed/1':report(.8), 'hover/1':report(.1)})
    assert not acceptable(before, before)
    assert not acceptable(before, {'mixed/1':report(.8), 'hover/1':report(.5)})
    assert not acceptable(before, {'mixed/1':report(.8), 'hover/1':report(.1, 0)})


def test_distill_end_to_end_exports_compatible_initialization(small, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pursuit_rl.ctbr.behavior as behavior
    from pursuit_rl.ctbr.checkpoint import save, load as restore
    source = tmp_path/'source.pt'
    model = new_model(small)
    save(source, model, small, {}, 'test')
    monkeypatch.setattr(behavior, 'load', lambda *a, **kw: copy.deepcopy(small))
    args = SimpleNamespace(command='distill', checkpoint=str(source), config='unused',
        out=str(tmp_path/'run'), device='cpu', num_envs=2, episodes=2,
        train_seed=1000, validation_seeds=[2000], validation_episodes=2,
        rounds=1, updates=1, stride=5, horizon=.002, lr=3e-5, warmup=0)
    behavior.run(args)
    result, _, saved = restore(tmp_path/'run/selected.pt')
    assert saved['progress']['critic_needs_calibration']
    assert saved['progress']['ppo_updates'] == 0
    assert torch.equal(result.log_std, model.log_std)
    assert (tmp_path/'run/round_1.json').exists()
