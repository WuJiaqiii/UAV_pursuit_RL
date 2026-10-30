"""High-rate controller and low-rate trace must be a real scheduler, not a label."""
import json
import numpy as np
import pytest
from pursuit_rl.config import load_config, Config
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.artifacts.trace import load_trace
from pursuit_rl.evaluation.scenarios import configuration


class CountController:
    def __init__(self):
        self.calls = 0
    def act(self, env, obs):
        self.calls += 1
        return np.zeros((env.n, 4), np.float32)


def fast_cfg():
    c=Config.load('configs/high_frequency/smoke_1khz.yaml').to_dict()
    c['curriculum'][0]['time'] = .006
    return c


def test_frequency_is_actual_actor_call_schedule():
    cfg=fast_cfg()
    ctrl=CountController()
    report=evaluate(ctrl, cfg, stage=0, episodes=1, seed=78)
    assert ctrl.calls == 6
    assert report['evaluation_protocol']['policy_frequency_hz'] == pytest.approx(1000.)
    cfg['physics']['policy_interval_steps']=3
    ctrl=CountController()
    report=evaluate(ctrl, cfg, stage=0, episodes=1, seed=78)
    assert ctrl.calls == 2
    assert report['evaluation_protocol']['policy_frequency_hz'] == pytest.approx(1000/3)


def test_trace_stride_preserves_full_rate_episode_metrics(tmp_path):
    cfg=fast_cfg()
    cfg['curriculum'][0]['time'] = .035
    cfg['evaluation']['trace_stride']=10
    evaluate(CountController(), cfg, stage=0, episodes=1, seed=91, trace_dir=tmp_path)
    z=load_trace(tmp_path/'episode_0000.npz')
    assert z['t'][0] == pytest.approx(0)
    assert z['t'][-1] == pytest.approx(.035)
    assert len(z['t']) == 5
    record=json.loads(str(z['terminal_record_json']))
    assert 'lateral_accel_cost_full' in record
    assert 'control_variation_full' in record
    assert 'high_envelope_any_ratio_full' in record
    assert z['physics_dt'] == pytest.approx(.001)


def test_1000hz_scenario_preserves_physics_clock():
    cfg=Config.load('configs/high_frequency/ppo_1khz.yaml').to_dict()
    case=configuration(cfg, 'ood_pulse_2m')
    assert case['physics']['dt'] == pytest.approx(.001)
    assert case['physics']['substeps'] == 1
    assert case['physics']['policy_interval_steps'] == 1
    cfg['physics']['policy_interval_steps']=10
    case=configuration(cfg, 'precision_2m')
    assert case['physics']['policy_interval_steps'] == 10


def test_high_rate_discount_preserves_physical_horizon():
    old=load_config()
    new=Config.load('configs/high_frequency/ppo_1khz.yaml').to_dict()
    for key in ('gamma','gae_lambda'):
        assert new['training'][key]**50 == pytest.approx(old['training'][key], rel=1e-6)


def test_training_does_not_silently_hold_actor_commands(tmp_path):
    from pursuit_rl.training.trainer import Trainer
    cfg=fast_cfg()
    cfg['physics']['policy_interval_steps']=10
    with pytest.raises(ValueError, match='policy_interval_steps'):
        Trainer(cfg, tmp_path/'invalid')
