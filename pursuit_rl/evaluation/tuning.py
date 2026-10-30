"""Validation-only PN/APN calibration; locked parameters are used for testing.

The tuner NEVER looks at the test seed. All candidates receive identical
validation episode seeds and scenario distribution. Current defaults are
included as a candidate. Validated constraints remain those of the plant.
"""
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from pursuit_rl.control.guidance import GuidanceParameters, GuidanceController
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.evaluation.scenarios import configuration, evaluation_stage
from pursuit_rl.artifacts.io import write_json


def candidates(algorithm, quick=False):
    default = GuidanceParameters()
    if quick:
        return [default, replace(default, speed_fraction=.99, accel_limit=15.)]
    if algorithm == 'PN':
        combinations = product((2., 3., 4., 5.), (.85, .99), (10., 20.))
        return [replace(default, N=N, speed_fraction=v, accel_limit=a)
                for N, v, a in combinations]
    combinations = product((3., 4.), (.5, 1., 1.5), (.85, .99), (10., 20.))
    return [replace(default, N=N, Na=na, speed_fraction=v, accel_limit=a)
            for N, na, v, a in combinations]


def calibrate(base, algorithm, episodes, seed, output, quick=False,
              scenarios=('singer_fast', 'rapid_hybrid')):
    if algorithm not in ('PN', 'APN'):
        raise ValueError('Calibration only accepts PN or APN')
    if episodes <= 0:
        raise ValueError('calibration episodes must be positive')
    rows = []
    for params in candidates(algorithm, quick):
        stats = []
        for scene in scenarios:
            config = configuration(base, scene)
            result = evaluate(GuidanceController(algorithm, params), config,
                              stage=evaluation_stage(config), episodes=episodes,
                              seed=seed, trace_dir=None)
            stats.append(result)
        mean_success = sum(r['success_rate'] for r in stats) / len(stats)
        # Keep time as a deterministic tie-breaker, not a replacement for success.
        time_normalized = sum(
            r['mean_capped_time'] / configuration(base, sc)['curriculum'][evaluation_stage(configuration(base, sc))]['time']
            for sc, r in zip(scenarios, stats)
        ) / len(stats)
        score = mean_success - .05 * time_normalized
        rows.append({'parameters': asdict(params), 'success_rate': mean_success,
                     'mean_capped_time_fraction': time_normalized, 'score': score})
    rows.sort(key=lambda r: (r['score'], r['success_rate']), reverse=True)
    outcome = {
        'algorithm': algorithm, 'protocol': 'validation-only-locked-parameters-v1',
        'validation_scenarios': list(scenarios), 'validation_seed': int(seed),
        'episodes_per_scenario': episodes, 'candidate_count': len(rows),
        'selected': rows[0], 'candidates': rows,
    }
    write_json(Path(output), outcome)
    return rows[0]['parameters']
