"""Evaluate single-aircraft navigation on randomized, low-speed survey references."""

import argparse
import copy
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pursuit_rl.ctbr.checkpoint import load, write_json
from pursuit_rl.ctbr.config import validate
from pursuit_rl.ctbr.control import teacher
from pursuit_rl.ctbr.environment import CTBRBatch


@dataclass(frozen=True)
class Leg:
    seconds: int
    speed: tuple[float, float]
    turn_rate: float
    intervals: tuple[float, float]


LEGS = {
    "01_random_turns": Leg(12, (0.7, 0.7), 0.35, (1.8, 3.2)),
    "02_random_speed_turns": Leg(16, (0.4, 1.0), 0.50, (1.5, 2.8)),
    "03_frequent_changes": Leg(20, (0.4, 1.0), 0.50, (1.0, 2.0)),
}
SEEDS = (4490042, 4590042)
SCENES = 16
REQUIRED = 15
RADIUS = 0.45
TAIL_RMSE_LIMIT = 0.32


def random_route(leg, seed, dt, initial_positions, initial_velocities):
    """Draw independent command knots and integrate a smooth horizontal route."""
    rng = np.random.default_rng(seed)
    time = np.arange(round(leg.seconds / dt) + 1) * dt
    count = len(initial_positions)
    speed = np.empty((len(time), count))
    yaw_rate = np.empty_like(speed)
    for slot in range(count):
        knot_time = [0.0]
        while knot_time[-1] < leg.seconds:
            knot_time.append(knot_time[-1] + rng.uniform(*leg.intervals))
        knots = np.asarray(knot_time)
        index = np.minimum(np.searchsorted(knots, time, side="right") - 1, len(knots) - 2)
        u = (time - knots[index]) / (knots[index + 1] - knots[index])
        blend = (1 - np.cos(np.pi * u)) / 2
        speed_knots = rng.uniform(*leg.speed, size=len(knots))
        speed_knots[0] = 0.7
        rate_knots = rng.uniform(-leg.turn_rate, leg.turn_rate, size=len(knots))
        rate_knots[0] = 0
        speed[:, slot] = (1 - blend) * speed_knots[index] + blend * speed_knots[index + 1]
        yaw_rate[:, slot] = (1 - blend) * rate_knots[index] + blend * rate_knots[index + 1]

    start_yaw = np.arctan2(initial_velocities[:, 1], initial_velocities[:, 0])
    yaw = np.vstack((start_yaw, start_yaw + np.cumsum(
        0.5 * (yaw_rate[:-1] + yaw_rate[1:]) * dt, axis=0
    )))
    velocity = np.stack((speed * np.cos(yaw), speed * np.sin(yaw), np.zeros_like(speed)), axis=-1)
    position = np.empty_like(velocity)
    position[0] = initial_positions
    position[1:] = initial_positions + np.cumsum(
        0.5 * (velocity[:-1] + velocity[1:]) * dt, axis=0
    )
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = np.diff(velocity, axis=0) / dt
    return position, velocity, acceleration


class SurveyBatch(CTBRBatch):
    """Use the CTBR simulator with a virtual navigation reference."""

    def __init__(self, config, leg, seed, count):
        super().__init__(config, n=count, seed=seed)
        self.slots = torch.arange(count, device=self.device)
        point = self.plant.target
        route = random_route(
            leg, seed + 760000, self.dt, point.pos.cpu().numpy(), point.vel.cpu().numpy()
        )
        self.route = tuple(torch.as_tensor(arr, device=self.device) for arr in route)
        h = hashlib.sha256()
        for arr in route:
            h.update(arr.tobytes())
        self.route_digest = h.hexdigest()
        self.maximum_reference_speed_mps = float(np.linalg.norm(route[1], axis=-1).max())
        self.maximum_reference_acceleration_mps2 = float(np.linalg.norm(route[2], axis=-1).max())

    def advance_target(self, active=None):
        if active is None:
            active = ~self.finished
        step = (self.steps + 1).clamp_max(self.max_steps)
        point = self.plant.target
        for key, table in zip(("pos", "vel", "acc"), self.route):
            state = torch.where(active[:, None], table[step, self.slots], getattr(point, key))
            setattr(point, key, state)


def configure(base, name, leg):
    c = copy.deepcopy(base)
    c["task"].update(
        name=name, objective="sustained", kind="cruise", duration=leg.seconds,
        tail_seconds=4, radius=RADIUS, velocity_tolerance=0.6,
        minimum_tail_fraction=0.90, maximum_tail_rmse=TAIL_RMSE_LIMIT,
        distance=[0.5, 1.25], speed=[0.7, 0.7], max_accel=0,
        initial_tilt_deg=0, initial_velocity_noise=0, match_initial_velocity=True,
        randomization=0,
    )
    return validate(c)


@torch.no_grad()
def flight(c, leg, seed, count, model=None):
    env = SurveyBatch(c, leg, seed, count)
    obs, _ = env.observe()
    initial_digest = hashlib.sha256(obs.cpu().numpy().tobytes()).hexdigest()
    frames, records, times = [], [], []
    stride = round(0.1 / env.decision_dt)

    def snapshot(tick):
        times.append(tick * env.decision_dt)
        frames.append(torch.cat(
            (env.plant.pursuer.pos, env.plant.target.pos, env.plant.target.vel), dim=1
        ).cpu().numpy())

    snapshot(0)
    for tick in range(1, env.max_decision_steps + 1):
        commands = teacher(env) if model is None else model.predict_tensor(obs)
        (obs, _), _, _, _, completed = env.step(commands)
        records.extend(completed)
        done = len(records) == count
        if tick % stride == 0 or done:
            snapshot(tick)
        if done:
            break
    if len(records) != count or {r["slot"] for r in records} != set(range(count)):
        raise RuntimeError("Incomplete flight records")

    track = np.stack(frames)
    time = np.asarray(times)
    rows = []
    for record in sorted(records, key=lambda r: r["slot"]):
        slot = record["slot"]
        mask = time <= record["time"] + 1e-5
        error = track[mask, slot, :3] - track[mask, slot, 3:6]
        velocity = track[mask, slot, 6:8]
        tangent = velocity / np.linalg.norm(velocity, axis=1)[:, None]
        cross_track = error[:, 1] * tangent[:, 0] - error[:, 0] * tangent[:, 1]
        qualified = record["success"] and record["tail_position_rmse"] <= TAIL_RMSE_LIMIT
        rows.append(dict(
            slot=slot, qualified=qualified, terminal_reason=record["reason"],
            terminal_time_s=record["time"], position_rmse_m=record["position_rmse"],
            tail_position_rmse_m=record["tail_position_rmse"] if record["tail_complete"] else None,
            tail_inside_fraction=record["tail_within_fraction"] if record["tail_complete"] else None,
            cross_track_rmse_sampled_m=float(np.sqrt(np.mean(cross_track**2))),
            height_rmse_sampled_m=float(np.sqrt(np.mean(error[:, 2]**2))),
            action_saturation_fraction=record["action_saturation_fraction"],
        ))
    return dict(
        seed=seed, scenes=count, qualified=sum(r["qualified"] for r in rows),
        route_digest=env.route_digest, initial_state_digest=initial_digest,
        max_reference_speed_mps=env.maximum_reference_speed_mps,
        max_reference_acceleration_mps2=env.maximum_reference_acceleration_mps2,
        episodes=rows,
    ), time, track


def draw_pair(out, time, geometry, actor, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    eligible = [i for i, row in enumerate(rows) if row["qualified"]]
    ranked = sorted(eligible or range(len(rows)), key=lambda i: rows[i]["position_rmse_m"])
    slot = ranked[len(ranked) // 2] if len(eligible) == len(rows) else max(
        (i for i in range(len(rows)) if i not in eligible),
        key=lambda i: rows[i]["position_rmse_m"],
    )
    ref = actor[:, slot, 3:6]
    geo = geometry[:, slot, :3]
    learned = actor[:, slot, :3]
    origin = ref[0]
    err = [np.linalg.norm(path - ref, axis=1) for path in (geo, learned)]
    fig, (map_ax, error_ax) = plt.subplots(1, 2, figsize=(8.3, 3.9), layout="constrained")
    map_ax.plot(ref[:, 1] - origin[1], ref[:, 0] - origin[0], "--", color="#999999",
                linewidth=1.4, label="Survey reference")
    map_ax.set(xlabel="East from start (m)", ylabel="North from start (m)", aspect="equal")
    for arr in (geo, learned):
        map_ax.update_datalim(np.column_stack((arr[:, 1] - origin[1], arr[:, 0] - origin[0])))
    map_ax.autoscale_view()
    error_ax.set(xlabel="Time (s)", ylabel="3D position error (m)",
                 xlim=(0, time[-1]), ylim=(0, max(RADIUS, max(e.max() for e in err)) * 1.10))
    error_ax.axhline(RADIUS, color="#999999", linestyle="--", linewidth=1)
    paths, dots, errors = [], [], []
    for name, color in (("Geometric", "#008a85"), ("Actor", "#df6823")):
        path, = map_ax.plot([], [], color=color, linewidth=1.8, label=name)
        dot, = map_ax.plot([], [], "o", color=color, markersize=4)
        line, = error_ax.plot([], [], color=color, linewidth=1.5)
        paths.append(path)
        dots.append(dot)
        errors.append(line)
    map_ax.legend(fontsize=8, loc="best")
    for ax in (map_ax, error_ax):
        ax.grid(alpha=0.23)
    title = fig.suptitle("")

    def update(frame):
        for path, dot, line, arr, e in zip(paths, dots, errors, (geo, learned), err):
            end = frame + 1
            path.set_data(arr[:end, 1] - origin[1], arr[:end, 0] - origin[0])
            dot.set_data([arr[frame, 1] - origin[1]], [arr[frame, 0] - origin[0]])
            line.set_data(time[:end], e[:end])
        title.set_text(f"{out.stem.replace('_', ' ')} | scene {slot:02d} | {time[frame]:.1f} s")

    update(len(time) - 1)
    fig.savefig(out.with_suffix(".png"), dpi=145)
    frames = sorted({*range(0, len(time), 2), len(time) - 1})
    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(out.with_suffix(".gif"), writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    return slot


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legs={name: asdict(leg) for name, leg in LEGS.items()},
        seeds=args.seeds, device=args.device, render_gif=not args.no_gif,
        scenes=args.scenes, minimum_qualified=REQUIRED if args.scenes == SCENES else args.scenes,
    )
    receipt = out / "manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Run settings changed; choose a new --out.")
    write_json(receipt, manifest)
    model, base, _ = load(args.checkpoint, args.device)
    results = []
    for name, leg in LEGS.items():
        reports = []
        c = configure(base, name, leg)
        for seed in args.seeds:
            directory = out / name / str(seed)
            path = directory / "report.json"
            if path.exists():
                reports.append(json.loads(path.read_text()))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            geometry, gt, gp = flight(c, leg, seed, args.scenes)
            actor, at, ap = flight(c, leg, seed, args.scenes, model)
            for key in ("route_digest", "initial_state_digest"):
                if geometry[key] != actor[key]:
                    raise RuntimeError(f"Unpaired {key}: {name}, {seed}")
            common = min(len(gt), len(at))
            if not np.array_equal(gt[:common], at[:common]):
                raise RuntimeError(f"Unpaired sampling times: {name}, {seed}")
            np.savez_compressed(directory / "trajectories.npz",
                                geometry_time=gt, geometry=gp, actor_time=at, actor=ap)
            slot = None
            if not args.no_gif and seed == args.seeds[0]:
                slot = draw_pair(out / name, gt[:common], gp[:common], ap[:common], actor["episodes"])
            row = dict(geometry=geometry, actor=actor, animated_scene=slot)
            write_json(path, row)
            reports.append(row)
            print(name, seed, "geometry", geometry["qualified"], "actor", actor["qualified"], flush=True)
        result = dict(phase=name, controllers={})
        for key in ("geometry", "actor"):
            group = [r[key] for r in reports]
            episodes = [e for report in group for e in report["episodes"]]
            result["controllers"][key] = dict(
                qualified=sum(r["qualified"] for r in group),
                episodes=len(episodes),
                by_seed=[r["qualified"] for r in group],
                passed=all(r["qualified"] >= manifest["minimum_qualified"] for r in group),
                mean_tail_position_rmse_m=statistics.mean(
                    e["tail_position_rmse_m"] for e in episodes if e["tail_position_rmse_m"] is not None
                ) if any(e["tail_position_rmse_m"] is not None for e in episodes) else None,
                mean_cross_track_rmse_m=statistics.mean(e["cross_track_rmse_sampled_m"] for e in episodes),
                mean_height_rmse_m=statistics.mean(e["height_rmse_sampled_m"] for e in episodes),
                safety_terminations=sum(e["terminal_reason"] not in ("success", "tracking_failed")
                                        for e in episodes),
            )
        results.append(result)
        write_json(out / "summary.json", results)
    return 0 if all(r["controllers"]["actor"]["passed"] for r in results) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenes", type=int, choices=(2, SCENES), default=SCENES)
    parser.add_argument("--seeds", type=int, nargs=2, default=SEEDS)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != 2:
        parser.error("--seeds must contain two distinct values")
    torch.set_num_threads(1)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
