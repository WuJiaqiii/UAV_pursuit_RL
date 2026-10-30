"""E1/E2/E4 deterministic held-out scenario builders.

All values are validated against the original V6 config schema; the model
observation and topology contract are not modified by scenario overrides.
"""
import copy
from itertools import product
from pursuit_rl.config import validate
from pursuit_rl.evaluation.scenarios import configuration, evaluation_stage

E1_SCENARIOS = (
    'singer_fast', 'rapid_hybrid', 'high_accel_rapid',
    'ood_switchback_2m', 'ood_pulse_2m', 'overtaking_edge',
)


def apply_scenario_spec(base, spec):
    if not isinstance(spec, dict):
        raise ValueError('Scenario specification must be a map')
    cfg = configuration(base, 'standard')
    stage = evaluation_stage(cfg)
    for section, updates in spec.items():
        if section == 'stage':
            dest = cfg['curriculum'][stage]
        elif section in ('physics', 'target'):
            dest = cfg[section]
        else:
            raise ValueError(f'Unknown scenario section: {section}')
        if not isinstance(updates, dict) or (set(updates) - set(dest)):
            raise ValueError(f'Unsupported overrides in {section}')
        dest.update(copy.deepcopy(updates))
    validate(cfg)
    return cfg, stage


def performance_grid(speeds=((10,15),(15,20),(20,25)), accelerations=(3,5,7,9,10),
                     distance=(300.,400.), radius=2., time=150.):
    result = {}
    for (lo, hi), a in product(speeds, accelerations):
        if not (0 < lo <= hi <= 25 and 0 < a <= 10):
            raise ValueError('Grid outside validated target envelope')
        key = f'v{lo:02g}_{hi:02g}_a{a:02g}'
        result[key] = {
            'target': {'max_accel': float(max(7, a)), 'max_speed': 25.0,
                       'strict_speed_ceiling': True,
                       'maneuver_interval': [0.7, 1.6]},
            'stage': {'modes': ['hybrid'], 'distance': list(map(float, distance)),
                      'speed': [float(lo), float(hi)], 'max_accel': float(a),
                      'success_radius': float(radius), 'time': float(time)},
        }
    return result


def robustness_cases():
    return {
        'nominal': ({}, {}),
        'plant_lag': ({'physics': {'tau_thrust': .20, 'tau_rate': .24,
                                   'drag': .012}}, {}),
        'position_noise_1m': ({}, {'position_std_m': 1.}),
        'position_noise_2m': ({}, {'position_std_m': 2.}),
        'velocity_noise_05': ({}, {'velocity_std_mps': .5}),
        'observation_delay_50ms': ({}, {'observation_delay_s': .05}),
        'observation_delay_100ms': ({}, {'observation_delay_s': .10}),
        'observation_delay_200ms': ({}, {'observation_delay_s': .20}),
        'command_delay_50ms': ({}, {'command_delay_s': .05}),
        'command_delay_100ms': ({}, {'command_delay_s': .10}),
        'command_delay_200ms': ({}, {'command_delay_s': .20}),
        'combined': ({'physics': {'tau_thrust': .20, 'tau_rate': .24}},
                     {'position_std_m': 1., 'velocity_std_mps': .5,
                      'observation_delay_s': .10, 'command_delay_s': .10}),
    }
