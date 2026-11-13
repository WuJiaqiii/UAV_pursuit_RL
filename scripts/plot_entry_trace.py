"""Plot one episode from evaluate --trace-dir output."""

import argparse
from pathlib import Path

import numpy as np


def load_episode(path, slot, radius_override=None):
    with np.load(path, allow_pickle=False) as trace:
        if not 0 <= slot < trace["pursuer"].shape[1]:
            raise ValueError("Slot is outside the recorded batch")
        active = trace["active"][:, slot]
        time = trace["time"][active]
        pursuer = trace["pursuer"][active, slot]
        target = trace["target"][active, slot]
        rotation = trace["rotation"][active, slot]
        body_rate = trace["body_rate"][active, slot]
        radius = radius_override
        if radius is None and "radius_m" in trace:
            radius = float(trace["radius_m"])
        terminal = None
        if "terminal_slot" in trace:
            matches = np.flatnonzero(trace["terminal_slot"] == slot)
            if len(matches) == 1:
                index = matches[0]
                terminal = (
                    float(trace["terminal_time"][index]),
                    str(trace["terminal_reason"][index]),
                    trace["terminal_pursuer"][index],
                    trace["terminal_target"][index],
                )
    if radius is None or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Provide a positive --radius for traces without radius metadata")
    if not len(time):
        raise ValueError("No active samples recorded for this slot")
    return time, pursuer, target, rotation, body_rate, radius, terminal


def plot(path, slot, output, radius_override=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    time, pursuer, target, rotation, body_rate, radius, terminal = load_episode(
        path, slot, radius_override
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    xy, relative, distance_ax, attitude = axes.flat
    flight_time = time
    if terminal:
        end_time, reason, final_pursuer, final_target = terminal
        flight_time = np.r_[time, end_time]
        pursuer = np.vstack((pursuer, final_pursuer))
        target = np.vstack((target, final_target))
    offset = pursuer - target
    distances = np.linalg.norm(offset, axis=1)

    xy.plot(pursuer[:, 0], pursuer[:, 1], label="Pursuer")
    xy.plot(target[:, 0], target[:, 1], label="Target")
    xy.set(title="World trajectory", xlabel="North (m)", ylabel="East (m)")
    xy.set_aspect("equal", adjustable="datalim")
    xy.legend()

    relative.plot(offset[:, 0], offset[:, 1], label="Relative trajectory")
    relative.add_patch(Circle((0, 0), radius, fill=False, linestyle="--",
                              color="gray", label="Sphere projection"))
    relative.scatter(*offset[0, :2], marker="o", label="Start")
    if terminal:
        relative.scatter(*offset[-1, :2], marker="x", label="Terminal")
    relative.set(title="Target-relative top view", xlabel="North offset (m)", ylabel="East offset (m)")
    relative.set_aspect("equal", adjustable="datalim")
    relative.legend()

    distance_ax.plot(flight_time, distances, label="3D distance")
    distance_ax.axhline(radius, color="gray", linestyle="--", label=f"{radius:g} m radius")
    if terminal:
        distance_ax.scatter(end_time, distances[-1], marker="x", label=reason)
    distance_ax.set(title="Entry condition", xlabel="Time (s)", ylabel="Distance (m)")
    distance_ax.legend()

    tilt = np.degrees(np.arccos(np.clip(rotation[:, 2, 2], -1, 1)))
    attitude.plot(time, tilt, color="tab:orange", label="Tilt")
    rate_ax = attitude.twinx()
    rate_ax.plot(time, np.linalg.norm(body_rate, axis=1), color="tab:blue", label="Body rate norm")
    attitude.set(title="Attitude and actual body rate", xlabel="Time (s)", ylabel="Tilt (deg)")
    rate_ax.set_ylabel("Body rate norm (rad/s)")
    attitude.legend(loc="upper left")
    rate_ax.legend(loc="upper right")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    status = f"{reason} at {end_time:.3f} s" if terminal else "sampled frames"
    fig.suptitle(f"{path.stem} | slot {slot} | {status}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=float)
    args = parser.parse_args()
    plot(args.trace, args.slot, args.out, args.radius)


if __name__ == "__main__":
    main()
