"""E1–E5 executable experiment suite using the original V6 plant and rollout.

This is deliberately a thin orchestrator. It never modifies PPO weights or
training configuration, uses held-out seeds, and exports a fully recorded
configuration/protocol for every comparison.
"""
import copy
import json
from pathlib import Path
from pursuit_rl.config import load_config
from pursuit_rl.learning.checkpoints import load_model
from pursuit_rl.learning.contracts import require_compatible_policy
from pursuit_rl.evaluation.benchmark import run_benchmark
from pursuit_rl.evaluation.scenario_grid import E1_SCENARIOS, performance_grid, robustness_cases
from pursuit_rl.evaluation.scenarios import SCENARIOS
from pursuit_rl.evaluation.tuning import calibrate
from pursuit_rl.evaluation.metrics.report import write_summary, write_latex
from pursuit_rl.artifacts.io import write_json


def _config(checkpoint, config_path=None, device=None, backend=None):
    if checkpoint:
        model, checkpoint_data = load_model(checkpoint)
        base = checkpoint_data['config']
        if config_path:
            from pursuit_rl.config import Config
            base = Config.load(config_path).to_dict()
        require_compatible_policy(checkpoint_data['config'], base)
        if device is not None:
            import torch
            base['device']='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        if backend is not None: base['simulation_backend']=backend
        elif device=='cuda':base['simulation_backend']='torch'
        return base
    base=load_config(config_path)
    if device is not None:
        import torch
        base['device']='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
    if backend is not None:base['simulation_backend']=backend
    elif device=='cuda':base['simulation_backend']='torch'
    return base


def _existing_rows(out):
    result = []
    for path in sorted(Path(out).glob('E*/**/summary.json')):
        if path.name != 'summary.json' or not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding='utf8'))
        if isinstance(doc, list) and doc and 'algorithm' in doc[0]:
            for row in doc:
                result.append({'experiment':path.parts[len(Path(out).parts)],
                               'benchmark':str(path.parent.relative_to(out)), **row})
    return result


def _visuals(out, rows):
    from pursuit_rl.evaluation.metrics.pareto import pareto_front
    from pursuit_rl.visualization.experiment import (
        paired_traces, control_panels, performance_map, pareto_plot,
    )
    root = Path(out)
    figures = root / 'figures'; figures.mkdir(exist_ok=True)
    files = []
    e2 = [r for r in rows if r['experiment']=='E2']
    if e2:
        files.append(performance_map(e2, figures / 'E2_performance_map.png'))
    # Pareto comparisons are strictly grouped by scenario and benchmark identity.
    groups = {}
    for r in rows:
        key = (r['benchmark'], r['scenario'])
        groups.setdefault(key, []).append(r)
    for (benchmark, scene), group in groups.items():
        if len(group) < 2:
            continue
        # E2 may contain many scenario cells: visualize only the first 3 of each suite.
        if scene.startswith('v') and scene not in ('v10_15_a03','v15_20_a07','v20_25_a10'):
            continue
        filename = f'{benchmark.replace("/","_")}_{scene}'
        image = pareto_plot(group, figures / f'{filename}_pareto.png', scenario=scene, front=pareto_front(group))
        if image: files.append(image)
        tracepaths = {}
        for row in group:
            path = root / benchmark / row['algorithm'] / scene / 'episode_0000.npz'
            if path.exists():
                tracepaths[row['algorithm']] = path
        if len(tracepaths) >= 2:
            try:
                files.append(paired_traces(tracepaths, figures / f'{filename}_3d.png', title=f'{benchmark}: {scene}'))
                files.append(control_panels(tracepaths, figures / f'{filename}_controls.png'))
            except ValueError as e:
                raise RuntimeError(f'Pairing validation failed for {benchmark}/{scene}: {e}') from e
    return [str(f.relative_to(root)) for f in files]


def run_suite(checkpoint, suites=('E1','E2','E3','E4','E5'), episodes=128,
              seed=930000, out='runs/e1e5', calibration_episodes=16,
              quick=False, algorithms=('PPO','PN','APN'), config_path=None,
              plots=True, device=None, backend=None):
    suites = list(suites)
    if len(suites) != len(set(suites)) or any(s not in ('E1','E2','E3','E4','E5') for s in suites):
        raise ValueError('Choose E1/E2/E3/E4/E5 at most once each')
    if episodes < 1 or calibration_episodes < 1:
        raise ValueError('Episode counts must be positive')
    algorithms = tuple(algorithms)
    if not algorithms or any(a not in ('PPO','PN','APN') for a in algorithms):
        raise ValueError('Only PPO, PN, APN supported in the research suite')
    if 'PPO' in algorithms and not checkpoint:
        raise ValueError('PPO comparison requires a real --checkpoint')
    base = _config(checkpoint, config_path, device, backend)
    root = Path(out)
    if any(s != 'E5' for s in suites) and root.exists() and any(root.iterdir()):
        raise FileExistsError('New E1–E4 runs require a new empty --out directory')
    root.mkdir(parents=True, exist_ok=True)
    protocol = {'schema':'e1e5-v1', 'suites':suites, 'checkpoint':str(checkpoint) if checkpoint else None,
                'seed':seed, 'episodes_per_scenario':episodes,
                'calibration_episodes':calibration_episodes, 'quick':quick,
                'algorithms':algorithms, 'no_train_or_checkpoint_write':True}
    write_json(root/('E5_redraw_protocol.json' if suites == ['E5'] else 'experiment_protocol.json'), protocol)
    results = {}
    def bench(where, **kwargs):
        return run_benchmark(cfg=base, algorithms=algorithms, checkpoint=checkpoint,
                             episodes=episodes, seed=seed, out=root/where,
                             plots=plots, **kwargs)
    if 'E1' in suites:
        names = E1_SCENARIOS[:2] if quick else E1_SCENARIOS
        results['E1'] = bench('E1/benchmark', scenarios=list(names))
    if 'E2' in suites:
        grid = performance_grid()
        if quick:
            grid = {k: grid[k] for k in ('v10_15_a03','v20_25_a10')}
        results['E2'] = bench('E2/benchmark', scenario_specs=grid)
    if 'E3' in suites:
        selected = ['singer_fast','rapid_hybrid'] if quick else ['singer_fast', 'rapid_hybrid',
                                                                 'ood_switchback_2m', 'overtaking_edge']
        results['E3_default'] = bench('E3/default', scenarios=selected)
        params = {}
        for algo in ('PN','APN'):
            if algo in algorithms:
                params[algo] = calibrate(base, algo, calibration_episodes,
                                         seed+10_000_000, root/'E3'/f'{algo}_calibration.json',
                                         quick=quick)
        write_json(root/'E3'/'locked_parameters.json', params)
        results['E3_calibrated'] = bench('E3/calibrated', scenarios=selected,
                                        baseline_parameters=params)
    if 'E4' in suites:
        from pursuit_rl.simulation.disturbances import validate_disturbance
        selected_cases = robustness_cases()
        if quick:
            selected_cases = {key: selected_cases[key] for key in ('nominal', 'combined')}
        # Base E4 high-maneuver target is identical across all disturbance cases.
        reference = copy.deepcopy(SCENARIOS['ood_switchback_2m'])
        for case_name, (overrides, disturbance) in selected_cases.items():
            spec = {k: copy.deepcopy(reference[k]) for k in ('target','stage')}
            spec['physics'] = copy.deepcopy(overrides.get('physics', {}))
            disturbance = validate_disturbance(disturbance, base['physics']['dt'])
            results['E4_'+case_name] = bench(f'E4/{case_name}',
                scenario_specs={f'E4_{case_name}': spec}, disturbance=disturbance)
    all_rows = _existing_rows(root)
    write_json(root/'combined_summary.json', all_rows)
    if all_rows:
        write_summary(all_rows, root/'combined_summary.csv')
        write_latex(all_rows, root/'combined_table.tex')
    if 'E5' in suites:
        figures = _visuals(root, all_rows) if plots else []
        write_json(root/'E5_visualizations.json', {'figures':figures, 'count':len(figures)})
    write_json(root/('E5_redraw_complete.json' if suites == ['E5'] else 'experiment_complete.json'), {'suites':suites,
              'rows':len(all_rows), 'figures':len(list((root/'figures').glob('*.png'))) if (root/'figures').exists() else 0})
    return {'output':str(root), 'rows':len(all_rows), 'suites':suites}
