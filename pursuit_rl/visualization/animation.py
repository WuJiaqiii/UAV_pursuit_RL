#!/usr/bin/env python3
"""Render recorded Pursuit RL episode trajectories as a time-accurate 3D GIF/MP4.

Only visualizes saved replay traces; does not modify the environment or controller.
Uses NED->North/East/Altitude coordinates. Honors saved first-entry event.

"""
from __future__ import annotations
from pursuit_rl.artifacts.trace import normalize_trace

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

TARGET_COLOR = "tab:orange"
PURSUER_COLOR = "tab:blue"


def prepare_trace(trace):
    """Validate sampled traces; append recorded first-entry event, if recorded."""
    trace = normalize_trace(trace)
    radius = float(trace["capture_radius"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("capture_radius must be positive")
    for name in ("t", "pursuer_pos", "target_pos"):
        if name not in trace:
            raise ValueError(
                f"Missing {name!r} in trace. Replay/export position history first."
            )
    t = np.asarray(trace["t"], dtype=float).reshape(-1)
    p = np.asarray(trace["pursuer_pos"], dtype=float)
    target = np.asarray(trace["target_pos"], dtype=float)
    if (
        t.size < 2
        or p.shape != (t.size, 3)
        or target.shape != (t.size, 3)
        or not (
            np.isfinite(t).all() and np.isfinite(p).all() and np.isfinite(target).all()
        )
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError(
            "Expected finite t[N] strictly increasing, pursuer_pos[N,3], target_pos[N,3]."
        )

    # NED coordinates -> right-handed display as North/East/Altitude.
    p = p * np.array([1.0, 1.0, -1.0])
    target = target * np.array([1.0, 1.0, -1.0])
    hit_t = None
    if all(k in trace for k in ("hit_time", "hit_target_pos", "hit_pursuer_pos")):
        hit_t = float(np.asarray(trace["hit_time"]).reshape(()))
        target_hit = np.asarray(trace["hit_target_pos"], dtype=float).reshape(3) * [
            1,
            1,
            -1,
        ]
        pursuer_hit = np.asarray(trace["hit_pursuer_pos"], dtype=float).reshape(3) * [
            1,
            1,
            -1,
        ]
        if not t[0] <= hit_t <= t[-1] + 1e-5:
            raise ValueError(
                f"hit_time={hit_t} lies outside recorded interval [{t[0]}, {t[-1]}]"
            )
        hit_dist = np.linalg.norm(target_hit - pursuer_hit)
        if not np.isclose(hit_dist, radius, atol=0.005):
            raise ValueError(
                f"Expected first-entry, got {hit_dist:.4f}m. Check saved metadata."
            )
        # Place the exact sub-step event at the end of the animation, rather than
        # calling the final (post-step) sample the first-entry.
        prev = t < hit_t - 1e-9
        t = np.concatenate([t[prev], [hit_t]])
        p = np.concatenate([p[prev], pursuer_hit[None]], axis=0)
        target = np.concatenate([target[prev], target_hit[None]], axis=0)
        if len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError(
                "First-entry samples not sufficiently distinct for animation."
            )
    return t, p, target, hit_t


def _path_up_to(t, xyz, now, trail_seconds=0):
    """Return recorded true geometry through `now`, plus interpolated current point."""
    first = (
        np.searchsorted(t, max(t[0], now - trail_seconds), side="left")
        if trail_seconds > 0
        else 0
    )
    last = np.searchsorted(t, now, side="right")
    before = t[first:last]
    coords = xyz[first:last]
    if trail_seconds > 0:
        start_t = max(t[0], now - trail_seconds)
        if start_t < now and (len(before) == 0 or before[0] > start_t + 1e-9):
            start_xyz = np.array([np.interp(start_t, t, xyz[:, i]) for i in range(3)])
            coords = np.concatenate([start_xyz[None], coords])
    point = np.array([np.interp(now, t, xyz[:, i]) for i in range(3)])
    if not len(before) or abs(now - before[-1]) > 1e-9:
        coords = np.vstack((coords, point[None]))
    return coords, point


def render_animation(
    trace,
    out_prefix,
    *,
    gif=True,
    mp4=False,
    fps=12,
    duration=12,
    max_frames=240,
    camera="global",
    trail_seconds=0.0,
    rotate_degrees=0.0,
    title="Pursuit RL | 3D episode replay",
    dpi=105,
):
    """Write <prefix>_trajectory.gif / .mp4. Returns list of Path outputs."""
    if not gif and not mp4:
        raise ValueError("Choose GIF or MP4 output.")
    if fps < 1 or fps > 60 or duration <= 0 or max_frames < 2 or dpi < 50:
        raise ValueError(
            "Expected fps=1..60, positive duration, max_frames>=2 and dpi>=50."
        )
    if camera not in ("global", "follow") or trail_seconds < 0:
        raise ValueError("camera must be global/follow and trail_seconds>=0.")
    radius = float(trace["capture_radius"])
    t, p, target, hit_t = prepare_trace(trace)
    n = max(2, min(int(round(duration * fps)), int(max_frames)))
    frame_times = np.linspace(t[0], t[-1], n)
    # Preserve natural colors and a fixed global spatial scale.
    both = np.vstack((p, target))
    mid = (both.min(axis=0) + both.max(axis=0)) / 2
    span = max(float(np.ptp(both, axis=0).max()), 12.0)
    pad = span * 0.56
    distances = np.linalg.norm(target - p, axis=1)

    fig = plt.figure(figsize=(11.5, 6.4), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.65, 1.0],
        left=0.05,
        right=0.97,
        bottom=0.11,
        top=0.89,
        wspace=0.22,
    )
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    axd = fig.add_subplot(gs[0, 1])
    ax3.set_proj_type("persp")
    ax3.set_xlabel("North (m)", labelpad=6)
    ax3.set_ylabel("East (m)", labelpad=6)
    ax3.set_zlabel("Altitude (m)", labelpad=6)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=22, azim=-56)
    ax3.set_title("3D trajectory", fontsize=12)

    if camera == "global":
        for setter, c in (
            (ax3.set_xlim, mid[0]),
            (ax3.set_ylim, mid[1]),
            (ax3.set_zlim, mid[2]),
        ):
            setter(c - pad, c + pad)

    (target_line,) = ax3.plot(
        [], [], [], linewidth=2, c=TARGET_COLOR, label="Target path"
    )
    (pursuer_line,) = ax3.plot(
        [], [], [], linewidth=2, c=PURSUER_COLOR, label="Pursuer path"
    )
    (target_dot,) = ax3.plot(
        [], [], [], marker="o", linestyle="None", markersize=8, c=TARGET_COLOR
    )
    (pursuer_dot,) = ax3.plot(
        [], [], [], marker="^", linestyle="None", markersize=8, c=PURSUER_COLOR
    )
    (sep_line,) = ax3.plot([], [], [], linestyle="--", linewidth=1.2, c="0.4")
    if camera == "global":
        ax3.plot(
            [target[0, 0]],
            [target[0, 1]],
            [target[0, 2]],
            marker="s",
            linestyle="None",
            markersize=5,
            alpha=0.65,
            c=TARGET_COLOR,
        )
        ax3.plot(
            [p[0, 0]],
            [p[0, 1]],
            [p[0, 2]],
            marker="s",
            linestyle="None",
            markersize=5,
            alpha=0.65,
            c=PURSUER_COLOR,
        )
    (hit_marker,) = ax3.plot(
        [], [], [], marker="*", markersize=13, linestyle="None", c=TARGET_COLOR
    )
    ax3.legend(loc="upper left", fontsize=8)

    (dist_line,) = axd.plot([], [], linewidth=2, c=PURSUER_COLOR, label="Separation")
    axd.axhline(
        radius, linestyle="--", c="tab:red", linewidth=1.5, label=f"{radius:g} m radius"
    )
    (dist_marker,) = axd.plot(
        [], [], marker="o", markersize=6, linestyle="None", c=TARGET_COLOR
    )
    cursor = axd.axvline(t[0], linestyle=":", linewidth=1.2, c="0.45")
    axd.set_xlabel("Simulation time (s)")
    axd.set_ylabel("Target–pursuer separation (m)")
    axd.set_xlim(t[0], max(t[0] + 1e-3, t[-1]))
    axd.set_ylim(0, max(8.0, float(distances.max()) * 1.08))
    axd.grid(alpha=0.25)
    axd.legend(loc="upper right", fontsize=8)
    status = axd.text(
        0.02,
        0.96,
        "",
        transform=axd.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=.5", fc="white", ec=".85", alpha=0.97),
    )
    fig.suptitle(title, fontsize=14, fontweight="semibold")

    def set_line(line, xyz):
        line.set_data(xyz[:, 0], xyz[:, 1])
        line.set_3d_properties(xyz[:, 2])

    def update(i):
        now = float(frame_times[i])
        target_path, target_now = _path_up_to(t, target, now, trail_seconds)
        pursuer_path, pursuer_now = _path_up_to(t, p, now, trail_seconds)
        set_line(target_line, target_path)
        set_line(pursuer_line, pursuer_path)
        set_line(target_dot, target_now.reshape(1, 3))
        set_line(pursuer_dot, pursuer_now.reshape(1, 3))
        set_line(sep_line, np.vstack((target_now, pursuer_now)))
        dist_now = float(np.linalg.norm(target_now - pursuer_now))
        td = t[t <= now + 1e-9]
        dd = distances[: len(td)]
        if len(td) == 0 or td[-1] < now - 1e-9:
            td = np.append(td, now)
            dd = np.append(dd, dist_now)
        dist_line.set_data(td, dd)
        dist_marker.set_data([now], [dist_now])
        cursor.set_xdata([now, now])
        caught = hit_t is not None and now >= hit_t - 1e-8
        state = (
            f"FIRST {radius:g} m ENTRY @ {hit_t:.2f} s"
            if caught
            else "Pursuit in progress"
        )
        if hit_t is None and dist_now < radius:
            state = f"Within {radius:g} m (no verified entry event)"
        if caught:
            set_line(hit_marker, target_now.reshape(1, 3))
        else:
            hit_marker.set_data([], [])
            hit_marker.set_3d_properties([])
        status.set_text(f"t = {now:.2f} s\nd = {dist_now:.2f} m\n{state}")
        if camera == "follow":
            # Adapt the view to the live separation: captures begin far apart
            # but need a much tighter local view at capture.
            center = (target_now + pursuer_now) / 2
            zoom = max(12.0, dist_now * 0.62 + 18.0)
            for setter, c in (
                (ax3.set_xlim, center[0]),
                (ax3.set_ylim, center[1]),
                (ax3.set_zlim, center[2]),
            ):
                setter(c - zoom, c + zoom)
        if rotate_degrees:
            ax3.view_init(elev=22, azim=-56 + rotate_degrees * (i / max(n - 1, 1)))
        return (
            target_line,
            pursuer_line,
            target_dot,
            pursuer_dot,
            hit_marker,
            sep_line,
            dist_line,
            dist_marker,
            cursor,
            status,
        )

    animation = FuncAnimation(
        fig, update, frames=n, interval=1000 / fps, repeat=False, blit=False
    )
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        if gif:
            gif_path = Path(str(out_prefix) + "_trajectory.gif")
            animation.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=dpi)
            created.append(gif_path)
        if mp4:
            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "ffmpeg is not available. Install it (Ubuntu: sudo apt install ffmpeg)."
                )
            mp4_path = Path(str(out_prefix) + "_trajectory.mp4")
            animation.save(
                str(mp4_path), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi
            )
            created.append(mp4_path)
    finally:
        plt.close(fig)
    for path in created:
        print(
            f"Saved {path}  ({n} frames, {fps} fps; playback {n/fps:.2f} s, simulation {t[-1]-t[0]:.2f} s)"
        )
    return created
