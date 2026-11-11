import copy
import pytest
import torch
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.diagnostics import profiles


def test_axis_metrics_exact_tail_and_selective_reset(small):
    c = copy.deepcopy(small)
    c['task'].update(distance=[0, 0], initial_tilt_deg=0)
    env = CTBRBatch(c)
    env.plant.pursuer.pos[:, 0] -= 0.25
    records = []
    for _ in range(env.max_decision_steps):
        records.extend(env.step(torch.zeros(env.n, 4))[-1])
    for r in records:
        assert r['tail_axis_rmse_m'] == pytest.approx([0.25, 0, 0], abs=1e-5)
        assert r['tail_axis_mean_error_m'] == pytest.approx([0.25, 0, 0], abs=1e-5)
        assert r['tail_velocity_projection_s'] is None
    old = env.tail_axis_p2[1].clone()
    env.reset(torch.arange(env.n) == 0)
    assert not env.tail_axis_p2[0].any()
    assert torch.equal(env.tail_axis_p2[1], old)


def test_cruise_profiles_do_not_mutate_baseline(small):
    original = copy.deepcopy(small)
    tasks = profiles(small, [0.5, 1, 2])
    assert small == original
    for speed in [0.5, 1, 2]:
        assert tasks[f'cruise_{speed:g}']['task']['speed'] == [speed, speed]
        assert tasks[f'cruise_{speed:g}']['task']['randomization'] == 0


def test_reference_joint_precision_and_failed_tail(small):
    report, _ = evaluate(None, small, 2, 123, reference=True)
    assert report['joint_precision_success_rate'] == 1
    assert len(report['mean_tail_axis_rmse_m']) == 3
    c = copy.deepcopy(small)
    c['physics']['max_speed'] = 0.000001
    c['task']['initial_velocity_noise'] = 10
    failed, _ = evaluate(None, c, 2, 123, reference=True)
    assert failed['joint_precision_success_rate'] == 0
    assert failed['mean_tail_axis_rmse_m'] is None


def test_batch_runs_all_modes_and_records_provenance(small, tmp_path):
    import json
    from pursuit_rl.ctbr.checkpoint import new_model, save
    from pursuit_rl.ctbr.diagnostics import run
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(small))
    checkpoint = tmp_path / 'source.pt'
    save(checkpoint, new_model(small), small, {}, 'test')
    rows = run({'baseline': checkpoint}, config, tmp_path / 'diagnostic',
               episodes=2, seeds=[8123], speeds=[0.5], stochastic=True)
    assert len(rows) == 6
    assert {r['model'] for r in rows} == {'teacher', 'baseline'}
    assert checkpoint.exists()
    assert (tmp_path / 'diagnostic/manifest.json').exists()
    with pytest.raises(FileExistsError):
        run({'baseline': checkpoint}, config, tmp_path / 'diagnostic')
