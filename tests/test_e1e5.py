"""E1–E5 integration, reproducibility, perception/actuation and Pareto contracts."""
import numpy as np
import pytest
from pursuit_rl.config import load_config
from pursuit_rl.control.guidance import GuidanceController
from pursuit_rl.evaluation.tuning import candidates
from pursuit_rl.evaluation.scenario_grid import performance_grid, apply_scenario_spec
from pursuit_rl.evaluation.metrics.pareto import pareto_front
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.disturbances import validate_disturbance, SensorChannel, CommandChannel
from pursuit_rl.evaluation.metrics.envelope import envelope_metrics


def test_e2_grid_is_validated():
    grid = performance_grid()
    assert len(grid) == 15
    for case in grid.values():
        cfg, idx = apply_scenario_spec(load_config(), case)
        assert cfg['curriculum'][idx]['success_radius'] == 2.


def test_integer_step_delays_and_neutral_initial_command():
    cfg = load_config()
    with pytest.raises(ValueError, match='multiple'):
        validate_disturbance({'command_delay_s': .075}, cfg['physics']['dt'])
    command = CommandChannel(cfg, {'command_delay_s': .1}, 1)
    u = np.ones((1, 4), np.float32)
    assert np.allclose(command.apply(u), 0)
    assert np.allclose(command.apply(2*u), 0)
    assert np.allclose(command.apply(3*u), u)


def test_sensor_determinism_and_guide_uses_navigation():
    cfg = load_config()
    env = PursuitBatch(cfg, n=3, seed=42, stage=0)
    obs, _ = env.observe()
    sensor1 = SensorChannel(cfg, {'position_std_m': 20.}, 123)
    sensor2 = SensorChannel(cfg, {'position_std_m': 20.}, 123)
    original = GuidanceController('PN').act(env, obs)
    measured1 = sensor1.observe(env, obs)
    affected = GuidanceController('PN').act(env, measured1)
    measured2 = sensor2.observe(env, obs)
    assert np.array_equal(measured1, measured2)
    assert not np.array_equal(original, affected)
    assert np.array_equal(env.pursuer.pos, env.pursuer.pos.copy())


def test_e4_repeatability_and_real_command_delay():
    cfg = load_config()
    cfg['curriculum'][0].update(distance=[10., 20.], speed=[.1, .4], time=1.0)
    params = dict(stage=0, episodes=2, seed=952, trace_dir=None,
                  disturbance={'observation_delay_s':.10, 'command_delay_s':.10,
                               'position_std_m':.4})
    a = evaluate(GuidanceController('PN'), cfg, **params)
    b = evaluate(GuidanceController('PN'), cfg, **params)
    assert a['episodes_detail'] == b['episodes_detail']
    assert a['evaluation_protocol']['disturbance']['command_delay_s'] == .10


def test_e5_pareto_and_no_cross_task_comparison():
    rows = [{'algorithm':'PN', 'success_rate':.6, 'cost':10},
            {'algorithm':'PPO','success_rate':.8, 'cost':11},
            {'algorithm':'APN','success_rate':.5, 'cost':12}]
    front = pareto_front(rows, cost_key='cost')
    assert [r['algorithm'] for r in front] == ['PN', 'PPO']


def test_extended_envelope_metrics():
    t = np.array([0.,.05,.10])
    z = {'t': t, 'pursuer_vel':np.array([[4.,0,0],[6.,0,0],[8.,0,0]]),
         'pursuer_rates':np.array([[.1,.2,0],[1.,0,0],[1.2,0,0]]),
         'pursuer_rot':np.repeat(np.eye(3)[None],3,axis=0),
         'load_factor':np.ones(3), 'max_speed':10., 'max_rate_xy':1.2,
         'max_rate_yaw':1., 'max_tilt_deg':65.,'max_load_factor':3.}
    result = envelope_metrics(z)
    assert result['rate_util_max'] == pytest.approx(1.)
    assert result['high_envelope_any_ratio'] >= 1/3
    assert result['load_util_max'] == pytest.approx(1/3)


def test_tuning_candidate_grid():
    assert len(candidates('PN')) == 16
    assert len(candidates('APN')) == 24
