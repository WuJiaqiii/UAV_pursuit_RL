"""E1-E5 paired 3D, control analysis, heatmaps and per-task Pareto graphics."""
import re
from pathlib import Path
import numpy as np
from pursuit_rl.artifacts.trace import load_trace
from pursuit_rl.visualization.trajectory import display_positions, equal_aspect


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def paired_traces(paths, output, title=None):
    traces = {key: load_trace(path) for key, path in paths.items()}
    if len(traces) < 2:
        raise ValueError('At least two algorithms are required for paired plots')
    # This is a strong pairing test: the target trajectory is controller-independent.
    baseline = next(iter(traces.values()))
    for algo, trace in list(traces.items())[1:]:
        k = min(len(trace['t']), len(baseline['t']))
        if (not np.allclose(trace['t'][:k], baseline['t'][:k], atol=1e-9) or
                not np.allclose(trace['target_pos'][:k], baseline['target_pos'][:k], atol=1e-6)):
            raise ValueError(f'Target trajectories are not paired for {algo}')
    plt = _plt()
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    longest = max(traces.values(), key=lambda z: len(z['t']))
    target = display_positions(longest['target_pos'])
    ax.plot(*target.T, color='black', linestyle='--', linewidth=1.6, label='Target')
    positions = [target]
    for algo, z in traces.items():
        pos = display_positions(z['pursuer_pos'])
        positions.append(pos)
        line, = ax.plot(*pos.T, label=f'{algo}: {"hit" if bool(z.get("terminal_success",False)) else "miss"}')
        ax.scatter(*pos[0], marker='o', color=line.get_color())
        ax.scatter(*pos[-1], marker='X', color=line.get_color())
    equal_aspect(ax, np.concatenate(positions))
    ax.set(xlabel='North (m)', ylabel='East (m)', zlabel='Altitude (m)',
           title=title or 'Paired target: PPO vs PN vs APN')
    ax.legend()
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165); plt.close(fig)
    return output


def control_panels(paths, output):
    plt = _plt()
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for algo, path in paths.items():
        z = load_trace(path); t = z['t']
        axes[0].plot(t, z['relative_distance'], label=algo)
        axes[1].plot(t, z['cpa_distance'], label=algo)
        axes[2].plot(t, z['load_factor'], label=algo)
        axes[3].plot(t, np.linalg.norm(z['pursuer_rates'], axis=1), label=algo)
    for ax, y in zip(axes, ('Separation (m)', 'CPA (m)', 'Specific-force load (g)', 'Body-rate magnitude (rad/s)')):
        ax.set_ylabel(y); ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel('Simulation time (s)')
    fig.tight_layout(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165); plt.close(fig)
    return output


def performance_map(rows, output):
    """Plot actual E2 success rates; missing cells are NaN, never imputed."""
    plt = _plt()
    speeds = [(10,15), (15,20), (20,25)]
    accels = [3,5,7,9,10]
    algos = sorted(set(r['algorithm'] for r in rows))
    fig, axes = plt.subplots(1, len(algos), figsize=(5.2*len(algos), 4.8), squeeze=False)
    for algo, ax in zip(algos, axes[0]):
        matrix = np.full((len(accels), len(speeds)), np.nan)
        for i, a in enumerate(accels):
            for j, (lo, hi) in enumerate(speeds):
                name = f'v{lo:02g}_{hi:02g}_a{a:02g}'
                matches = [r for r in rows if r['scenario'] == name and r['algorithm'] == algo]
                if matches:
                    matrix[i, j] = matches[0]['success_rate']
        img = ax.imshow(matrix, vmin=0, vmax=1, origin='lower', aspect='auto')
        ax.set_xticks(range(len(speeds)), [f'{lo}–{hi}' for lo, hi in speeds])
        ax.set_yticks(range(len(accels)), accels)
        ax.set(title=algo, xlabel='Target speed (m/s)', ylabel='Acceleration ceiling (m/s²)')
        for i in range(len(accels)):
            for j in range(len(speeds)):
                if np.isfinite(matrix[i,j]):
                    ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center')
    fig.colorbar(img, ax=list(axes[0]), label='Capture success probability', shrink=.8)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches='tight'); plt.close(fig)
    return output


def pareto_plot(rows, output, scenario=None, cost_key='lateral_accel_cost_mean', front=None):
    if scenario is not None:
        rows = [r for r in rows if r['scenario'] == scenario]
    valid = [r for r in rows if r.get(cost_key) is not None and r.get('success_rate') is not None]
    if not valid:
        return None
    front = [] if front is None else front
    plt = _plt(); fig, ax = plt.subplots(figsize=(7, 5))
    for row in valid:
        ax.scatter(row[cost_key], row['success_rate'])
        ax.annotate(row['algorithm'], (row[cost_key], row['success_rate']), xytext=(3,4),textcoords='offset points')
    if len(front) > 1:
        ax.plot([r[cost_key] for r in front], [r['success_rate'] for r in front], '--', label='Non-dominated')
    ax.set(xlabel=cost_key, ylabel='Capture success probability', ylim=(-.03, 1.03),
           title=f'Matched scenario Pareto — {scenario or "selected task"}')
    ax.grid(alpha=.2); ax.legend() if len(front) > 1 else None
    fig.tight_layout(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165); plt.close(fig)
    return output
