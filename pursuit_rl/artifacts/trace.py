"""Versioned per-episode traces; legacy field names are accepted only on read."""

import json
from pathlib import Path
import numpy as np

TRACE_SCHEMA = "pursuit-trace-v1"
ALIASES = {
    "t": ("time",),
    "pursuer_pos": ("pos", "p_pos"),
    "pursuer_vel": ("vel", "p_vel"),
    "target_pos": ("t_pos",),
    "target_vel": ("t_vel",),
    "relative_distance": ("distance",),
}


def normalize_trace(trace):
    data = {k: np.asarray(v) for k, v in trace.items()}
    if "trace_schema" in data and str(data["trace_schema"]) != TRACE_SCHEMA:
        raise ValueError("Unsupported trace schema")
    for name, aliases in ALIASES.items():
        if name not in data:
            for alias in aliases:
                if alias in data:
                    data[name] = data[alias]
                    break
    if "t" not in data:
        raise ValueError("Trace requires timestamps t (seconds)")
    t = data["t"]
    if t.ndim != 1 or len(t) == 0 or not np.isfinite(t).all():
        raise ValueError("Expected one nonempty episode with finite 1D timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Trace timestamps must increase strictly")
    for name in ("pursuer_pos", "pursuer_vel", "target_pos", "target_vel"):
        if name in data and (
            data[name].shape != (len(t), 3) or not np.isfinite(data[name]).all()
        ):
            raise ValueError(f"{name} must have shape (T, 3) and finite values")
    if "relative_distance" not in data and all(
        k in data for k in ("pursuer_pos", "target_pos")
    ):
        data["relative_distance"] = np.linalg.norm(
            data["target_pos"] - data["pursuer_pos"], axis=1
        )
    if "relative_distance" in data:
        d = data["relative_distance"]
        if d.shape != t.shape or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError(
                "relative_distance must be a finite nonnegative (T,) array"
            )
    if "coordinate_frame" in data and str(data["coordinate_frame"]) != "NED/FRD":
        raise ValueError("Expected NED/FRD coordinate frame")
    if "action" in data and (
        data["action"].shape != (len(t), 4) or not np.isfinite(data["action"]).all()
    ):
        raise ValueError("action must have finite shape (T,4)")
    return data


def load_trace(path):
    with np.load(path, allow_pickle=False) as z:
        return normalize_trace({k: z[k] for k in z.files})


def save_episode(path, sequence, record, cfg, stage, seed):
    """Save t=0 through first terminal step, plus authoritative event metadata."""
    if not sequence:
        raise ValueError("Cannot save empty trajectory")
    arrays = {k: np.stack([s[k] for s in sequence]) for k in sequence[0]}
    reach = record["reach_time"]
    arrays.update(
        {
            "trace_schema": np.asarray(TRACE_SCHEMA),
            "coordinate_frame": np.asarray("NED/FRD"),
            "capture_radius": np.asarray(cfg["curriculum"][stage]["success_radius"]),
            "time_limit": np.asarray(cfg["curriculum"][stage]["time"]),
            "cpa_horizon": np.asarray(cfg["reward"]["cpa_horizon"]),
            "max_speed": np.asarray(cfg["physics"]["max_speed"]),
            "max_tilt_deg": np.asarray(cfg["physics"]["max_tilt_deg"]),
            "max_rate_xy": np.asarray(cfg["physics"]["rate_xy"]),
            "max_rate_yaw": np.asarray(cfg["physics"]["rate_yaw"]),
            "max_load_factor": np.asarray(
                min(
                    cfg["physics"]["thrust_max"],
                    cfg["physics"]["hover_thrust"] + cfg["physics"]["thrust_delta"],
                )
                / cfg["physics"]["gravity"]
            ),
            "episode": np.asarray(record["episode"]),
            "physics_dt": np.asarray(cfg["physics"]["dt"]),
            "policy_interval_steps": np.asarray(cfg["physics"].get("policy_interval_steps", 1)),
            "trace_stride": np.asarray(cfg.get("evaluation", {}).get("trace_stride", 1)),
            "batch_seed": np.asarray(seed),
            "terminal_success": np.asarray(record["success"]),
            "terminal_reason": np.asarray(record["reason"]),
            "terminal_reach_time": np.asarray(reach if reach is not None else np.nan),
            "terminal_record_json": np.asarray(json.dumps(record)),
        }
    )
    if record["success"]:
        # Exact last 1 ms environment event position: never interpolate
        # across the decimated trace interval.
        if record.get("capture_pursuer_pos") is not None:
            arrays["hit_pursuer_pos"] = np.asarray(record["capture_pursuer_pos"])
            arrays["hit_target_pos"] = np.asarray(record["capture_target_pos"])
        else:
            dt = arrays["t"][-1] - arrays["t"][-2]
            frac = np.clip((reach - arrays["t"][-2]) / dt, 0.0, 1.0)
            for name, key in (("pursuer_pos", "hit_pursuer_pos"), ("target_pos", "hit_target_pos")):
                arrays[key] = (1 - frac) * arrays[name][-2] + frac * arrays[name][-1]
        arrays["hit_time"] = np.asarray(reach)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class TraceRecorder:
    """Own per-slot sequences and stop recording at each first terminal event."""

    def __init__(self, env, folder):
        self.stride = int(env.cfg.get("evaluation", {}).get("trace_stride", 1))
        self.tick = 0
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        snapshot = env.snapshot()
        self.sequences = [
            [{k: np.asarray(v[i]).copy() for k, v in snapshot.items()}]
            for i in range(env.n)
        ]

    def record(self, env, finished, done=None):
        self.tick += 1
        if self.tick % self.stride != 0 and not (done is not None and np.any(done & ~finished)):
            return
        snapshot = env.snapshot()
        for i, sequence in enumerate(self.sequences):
            if not finished[i] and (self.tick % self.stride == 0 or (done is not None and done[i])):
                sequence.append(
                    {k: np.asarray(v[i]).copy() for k, v in snapshot.items()}
                )

    def save(self, local_index, record, cfg, stage, seed):
        save_episode(
            self.folder / f"episode_{record['episode']:04d}.npz",
            self.sequences[local_index],
            record,
            cfg,
            stage,
            seed,
        )
