"""Frozen paired diagnostics; never trains or modifies input checkpoints."""
import argparse
import copy
import hashlib
from pathlib import Path
from importlib.metadata import version
import torch
from .checkpoint import load as load_model, write_json, source_fingerprint
from .config import load, validate, interface, fingerprint
from .evaluation import evaluate


def profiles(base, speeds):
    result = {"mixed": copy.deepcopy(base)}
    for speed in speeds:
        if speed <= 0:
            raise ValueError("Cruise speeds must be positive")
        c = copy.deepcopy(base)
        c["task"].update(name=f"cruise_{speed:g}", kind="cruise", speed=[speed, speed],
                         max_accel=0, match_initial_velocity=True,
                         initial_tilt_deg=0, initial_velocity_noise=0, randomization=0)
        result[c["task"]["name"]] = validate(c)
    return result


def run(checkpoints, config, out, episodes=64, seeds=(310042, 410042),
        speeds=(0.5, 1.0, 2.0), device="cpu", stochastic=False):
    if episodes < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive episodes and distinct nonempty seeds required")
    if not checkpoints or "teacher" in checkpoints:
        raise ValueError("Provide model names; teacher is reserved")
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new empty diagnostic output directory")
    c = load(config, device=device)
    tasks = profiles(c, speeds)
    models, provenance = {}, {}
    for name, path in checkpoints.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Model names must be single path components")
        model, old, saved = load_model(path, device)
        if interface(old) != interface(c):
            raise ValueError(f"Incompatible interface for {name}; migrate explicitly first")
        models[name] = model
        provenance[name] = dict(path=str(Path(path).resolve()),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), origin=saved["origin"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", dict(source_fingerprint=source_fingerprint(),
        torch_version=torch.__version__, package_version=version("torch"), device=device,
        episodes=episodes, seeds=list(seeds), stochastic=stochastic,
        checkpoints=provenance, configs=tasks,
        fingerprints={k: fingerprint(v) for k, v in tasks.items()}))
    rows = []
    for task, profile in tasks.items():
        for seed in seeds:
            digest = None
            for name, model in [("teacher", None), *models.items()]:
                modes = [False, True] if stochastic and model is not None else [False]
                for noisy in modes:
                    report, _ = evaluate(model, profile, episodes, seed,
                                         stochastic=noisy, reference=model is None)
                    if digest is not None and digest != report["initial_state_digest"]:
                        raise RuntimeError("Paired initial states differ")
                    digest = report["initial_state_digest"]
                    mode = "stochastic" if noisy else "deterministic"
                    write_json(root / task / str(seed) / f"{name}_{mode}.json", report)
                    rows.append(dict(task=task, seed=seed, model=name, mode=mode,
                        success_rate=report["success_rate"],
                        joint_precision_success_rate=report["joint_precision_success_rate"],
                        tail_rmse=report["mean_tail_position_rmse"],
                        tail_axis_rmse_m=report["mean_tail_axis_rmse_m"],
                        tail_p95_m=report["tail_position_rmse_p95_m"],
                        full_tail_episodes=report["full_tail_episodes"],
                        tail_abs_along_offset_median_m=report.get("tail_abs_along_offset_median_m"),
                        tail_abs_along_offset_p90_m=report.get("tail_abs_along_offset_p90_m")))
                    write_json(root / "summary.json", rows)
                    print(rows[-1], flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeatable")
    p.add_argument("--config", default="configs/ctbr_v701/near_mixed.json")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seeds", nargs="+", type=int, default=[310042, 410042])
    p.add_argument("--speeds", nargs="+", type=float, default=[0.5, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--stochastic", action="store_true")
    a = p.parse_args()
    checkpoints = {}
    for item in a.checkpoint:
        name, sep, path = item.partition("=")
        if not sep or not path or name in checkpoints:
            p.error("Checkpoints require unique NAME=PATH entries")
        checkpoints[name] = path
    torch.set_num_threads(a.cpu_threads)
    run(checkpoints, a.config, a.out, a.episodes, a.seeds, a.speeds, a.device, a.stochastic)


if __name__ == "__main__":
    main()
