"""Common NED trajectory, command and capture-detail plots for any radius."""

from pathlib import Path
import numpy as np
from pursuit_rl.artifacts.trace import normalize_trace


def display_positions(positions):
    return np.asarray(positions) * [1.0, 1.0, -1.0]


def equal_aspect(ax, positions):
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    span = max(float(np.ptp(positions, axis=0).max()), 1.0) * 0.55
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(value - span, value + span)
    ax.set_box_aspect((1, 1, 1))


def plot_trace(trace, out, title="Pursuit trajectory"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = normalize_trace(trace)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pursuer = display_positions(trace["pursuer_pos"])
    target = display_positions(trace["target_pos"])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for positions, label in [(pursuer, "Pursuer"), (target, "Target")]:
        ax.plot(*positions.T, label=label)
        ax.scatter(*positions[0], marker="s")
    hit = all(key in trace for key in ["hit_time", "hit_target_pos", "hit_pursuer_pos"])
    if hit:
        ph = display_positions(trace["hit_pursuer_pos"])
        th = display_positions(trace["hit_target_pos"])
        ax.scatter(*ph, marker="X", s=70)
        ax.scatter(*th, marker="o", s=70)
        ax.plot(
            *np.stack((ph, th)).T,
            linestyle="--",
            label=f"First entry {float(trace['hit_time']):.2f}s",
        )
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(title)
    equal_aspect(ax, np.vstack((pursuer, target)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    dynamics = out.with_name(out.stem + "_dynamics.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    time = trace["t"]
    axes[0].plot(time, trace["relative_distance"], label="Sampled separation")
    if "capture_radius" in trace:
        axes[0].axhline(
            float(trace["capture_radius"]), linestyle="--", label="Capture radius"
        )
    if hit:
        axes[0].axvline(float(trace["hit_time"]), linestyle=":", label="First entry")
    axes[0].set_ylabel("Separation (m)")
    axes[0].legend()
    for key, label in [("pursuer_vel", "Pursuer"), ("target_vel", "Target")]:
        if key in trace:
            axes[1].plot(time, np.linalg.norm(trace[key], axis=1), label=label)
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend()
    if "action" in trace:
        for i, label in enumerate(["Collective", "Roll", "Pitch", "Yaw"]):
            axes[2].plot(time, trace["action"][:, i], label=label)
        axes[2].legend(ncol=4)
    axes[2].set_ylabel("Normalized command")
    axes[2].set_xlabel("Simulation time (s)")
    fig.tight_layout()
    fig.savefig(dynamics, dpi=160)
    plt.close(fig)
    outputs = [out, dynamics]
    if hit and "capture_radius" in trace:
        radius = float(trace["capture_radius"])
        zoom = out.with_name(out.stem + "_capture_zoom.png")
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        phi = np.linspace(0, 2 * np.pi, 26)
        theta = np.linspace(0, np.pi, 16)
        sphere = np.array(
            [
                radius * np.outer(np.cos(phi), np.sin(theta)),
                radius * np.outer(np.sin(phi), np.sin(theta)),
                radius * np.outer(np.ones_like(phi), np.cos(theta)),
            ]
        )
        ax.plot_wireframe(
            *(sphere + th[:, None, None]), rstride=3, cstride=3, alpha=0.3
        )
        ax.scatter(*th, label="Target center")
        ax.scatter(*ph, label="Pursuer at entry", marker="X")
        extent = max(2 * radius, 1.0)
        for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), th):
            setter(value - extent, value + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("North (m)")
        ax.set_ylabel("East (m)")
        ax.set_zlabel("Altitude (m)")
        ax.set_title(f"First {radius:g} m entry at {float(trace['hit_time']):.2f}s")
        ax.legend()
        fig.tight_layout()
        fig.savefig(zoom, dpi=160)
        plt.close(fig)
        outputs.append(zoom)
    return outputs
