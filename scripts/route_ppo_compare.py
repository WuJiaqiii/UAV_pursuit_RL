#!/usr/bin/env python3
"""Run paired PPO comparisons for a single aircraft on virtual survey routes."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


from scripts.survey_route_profiles import ARC, STATIC, PHASES, route_task


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/experiments/route_ppo_compare.json"
<<<<<<< HEAD
ARMS = ("critic_only", "ppo", "ppo_bc")
EVENT_CODES = dict(success=1, speed=2, outside=3, tilt=4, tracking_failed=5,
                   invalid_target=6, timeout=7)
=======
ROUTE_SOURCE_FINGERPRINT = "deaec637e145d3cc87fd94140a2356488bd2995d1a27c1ef33d75d3a59d30229"
ARMS = ("critic_only", "ppo", "ppo_bc")
EVENT_CODES = dict(success=1, speed=2, outside=3, tilt=4, tracking_failed=5,
                   invalid_target=6, timeout=7)


class IncompatibleRouteInput(ValueError):
    """The checkout or checkpoint cannot run this survey comparison."""


def check_route_source(root):
    # Match the v4 checkpoint fingerprint before importing Torch or its loader.
    digest = hashlib.sha256()
    paths = [root / "main.py", *(root / "pursuit_rl").rglob("*.py"),
             *(root / "tests").rglob("*.py")]
    try:
        for path in sorted(paths):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    except OSError as error:
        raise IncompatibleRouteInput(f"Cannot read survey source in {root}: {error}") from error
    found = digest.hexdigest()
    if found != ROUTE_SOURCE_FINGERPRINT:
        raise IncompatibleRouteInput(
            f"The code in {root} is not the v4 survey implementation required by mixed_bc.pt. "
            f"Source fingerprint: {found[:12]} (expected {ROUTE_SOURCE_FINGERPRINT[:12]}). "
            "Run the command in a clean worktree of origin/feat/route-ppo-ablation; "
            "copying the experiment scripts alone does not install the required implementation."
        )
    return found


>>>>>>> origin/feat/route-ppo-ablation
def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seeds(values, label):
    if (not isinstance(values, list) or not values
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must be an array of unique nonnegative 32-bit seeds")


def positive(group, keys):
    for key in keys:
        if type(group[key]) is not int or group[key] < 1:
            raise ValueError(f"{key} must be a positive integer")


def load_plan(path, mode):
    data = json.loads(path.read_text(encoding="utf-8"))
    warmup, training, run, ev = (data[key] for key in ("warmup", "training", mode, "evaluation"))
    positive(warmup, ("episodes", "critic_updates"))
    positive(training, ("envs", "rollout_steps", "batch_size", "epochs"))
    positive(run, ("updates", "evaluate_every"))
    positive(ev, ("episodes_per_seed",))
    for key in ("actor_lr", "critic_lr", "bc_coefficient"):
        if type(training[key]) not in (float, int) or not 0 < training[key] < math.inf:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < ev["minimum_joint_rate"] <= 1 or not 0 <= ev["max_tail_rmse_increase_m"] < math.inf:
        raise ValueError("Invalid precision gate settings")
    seeds(run["seeds"], "training seeds")
    seeds(ev["validation_seeds"], "validation seeds")
    seeds(ev["test_seeds"], "test seeds")
    if set(ev["validation_seeds"]) & set(ev["test_seeds"]):
        raise ValueError("Validation and final test seeds must be distinct")
    return dict(mode=mode, warmup=warmup, training=training, evaluation=ev, **run)


def route_config(original, name, device, seed, plan):
    c = copy.deepcopy(original)
    c.update(device=device, seed=seed, num_envs=plan["training"]["envs"])
    c["task"].update(route_task(name))
    c["physics"]["max_speed"] = min(10.0, c["physics"]["max_speed"])
    c["training"].update({key: plan["training"][key] for key in (
        "rollout_steps", "batch_size", "epochs", "actor_lr", "critic_lr",
    )})
    return c


def aggregate(reports):
    def mean(field):
        values = [row[field] for row in reports]
        return statistics.mean(values) if all(value is not None for value in values) else None

    height = [row["mean_tail_axis_rmse_m"] for row in reports]
    return {
        "episodes": sum(row["episodes"] for row in reports),
        "successes": sum(row["successes"] for row in reports),
        "precision_successes": sum(round(row["joint_precision_success_rate"] * row["episodes"]) for row in reports),
        "full_flights": sum(row["full_tail_episodes"] for row in reports),
        "tail_rmse_m": mean("mean_tail_position_rmse"),
        "tail_height_rmse_m": statistics.mean(x[2] for x in height) if all(height) else None,
    }


def eligible(current, baseline, ev):
    return all(
        row["full_flights"] == row["episodes"]
        and row["precision_successes"] / row["episodes"] >= ev["minimum_joint_rate"]
        and row["tail_rmse_m"] is not None and baseline[name]["tail_rmse_m"] is not None
        and row["tail_rmse_m"] <= baseline[name]["tail_rmse_m"] + ev["max_tail_rmse_increase_m"]
        for name, row in current.items()
    )


def rank(summary):
    row = summary[ARC]
    return (row["precision_successes"], -row["tail_rmse_m"])


def run(args, plan):
<<<<<<< HEAD
    checkpoint = args.checkpoint.resolve(strict=True)
=======
    runtime_fingerprint = check_route_source(ROOT)
    if not args.checkpoint.is_file():
        raise IncompatibleRouteInput(f"Checkpoint file not found: {args.checkpoint}")
    checkpoint = args.checkpoint.resolve()
>>>>>>> origin/feat/route-ppo-ablation
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    commit = head.stdout.strip() if head.returncode == 0 else None
    import torch
<<<<<<< HEAD
    from pursuit_rl.ctbr.checkpoint import load, new_model, save, source_fingerprint
=======
    from pursuit_rl.ctbr import checkpoint as checkpoint_io

    expected_loader = ROOT / "pursuit_rl/ctbr/checkpoint.py"
    actual_loader = Path(checkpoint_io.__file__).resolve()
    if actual_loader != expected_loader:
        raise IncompatibleRouteInput(
            f"Python imported checkpoint.py from {actual_loader}, expected {expected_loader}. "
            "Check PYTHONPATH and run from the survey worktree."
        )
    from pursuit_rl.ctbr.checkpoint import load, new_model, save
>>>>>>> origin/feat/route-ppo-ablation
    from pursuit_rl.ctbr.config import validate
    from pursuit_rl.ctbr.environment import CTBRBatch
    from pursuit_rl.ctbr.evaluation import evaluate
    from pursuit_rl.ctbr.value_learning import fit_critic, value_errors
    from pursuit_rl.learning.buffers import TensorRolloutBuffer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
<<<<<<< HEAD
    teacher, original, saved = load(checkpoint, args.device)
    if (saved["origin"] != "mixed_route_bc" or original["task"]["name"] != ARC
            or original["task"]["kind"] != "circle" or original["task"]["objective"] != "sustained"):
        raise ValueError("Expected the mixed_route_bc checkpoint from the preplanned circular survey")
    runtime_fingerprint = source_fingerprint()
    if saved["source_fingerprint"] != runtime_fingerprint:
        raise ValueError(
            "Incompatible checkpoint source. Compare the trainer and test files in this "
            "checkout with the temp revision 07c262e (including LF line endings)."
=======
    try:
        teacher, original, saved = load(checkpoint, args.device)
    except ValueError as error:
        raise IncompatibleRouteInput(f"Cannot load {checkpoint}: {error}") from error
    if (saved["origin"] != "mixed_route_bc" or original["task"]["name"] != ARC
            or original["task"]["kind"] != "circle" or original["task"]["objective"] != "sustained"):
        raise IncompatibleRouteInput("Expected the mixed_route_bc checkpoint from the preplanned circular survey")
    if saved["source_fingerprint"] != runtime_fingerprint:
        raise IncompatibleRouteInput(
            f"{checkpoint} was saved by a different source version. "
            "Use mixed_route_v2/mixed_bc.pt from the v4 survey experiment."
>>>>>>> origin/feat/route-ppo-ablation
        )
    teacher.eval().requires_grad_(False)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", {
        **plan, "git_commit": commit, "source_fingerprint": runtime_fingerprint,
        "device": args.device, "routes": PHASES,
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    })
    settings, ev = plan["training"], plan["evaluation"]
    captures = {}
    summaries = []
    selected = {}
    validation = ev["validation_seeds"]
    completed = 0
    total = 0 if args.preflight else len(ARMS) * len(plan["seeds"]) * plan["updates"]

    def assess(model, routes, seed_list, folder):
        folder.mkdir(parents=True, exist_ok=True)
        result = {}
        for name in routes:
            c = validate(route_config(original, name, args.device, original["seed"], plan))
            reports = []
            for seed in seed_list:
                report, _ = evaluate(model, c, ev["episodes_per_seed"], seed)
                digest = report["initial_state_digest"]
                if captures.setdefault((name, seed), digest) != digest:
                    raise RuntimeError(f"Evaluation initial states differ: {name}, seed {seed}")
                write_json(folder / f"{name}_{seed}.json", report)
                reports.append(report)
            result[name] = aggregate(reports)
        write_json(folder / "summary.json", result)
        return result

    def log_summary(arm, seed, update, phase, result, candidate):
        for name, stats in result.items():
            summaries.append(dict(phase=phase, arm=arm, seed=seed, update=update,
                                  route=name, eligible=candidate, **stats))
        temporary = out / "summary.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        temporary.replace(out / "summary.csv")

    def progress(phase, arm=None, seed=None, update=None):
        data = dict(phase=phase, arm=arm, seed=seed, update=update,
                    completed_updates=completed, planned_updates=total)
        write_json(out / "progress.json", data)
        print(f"[{completed}/{total} PPO updates] {phase} {arm or ''} {seed or ''}", flush=True)

    @torch.no_grad()
    def collect(model, environment, obs, priv, labels):
        buf = TensorRolloutBuffer(settings["rollout_steps"], environment.n, environment.actor_dim,
                                  environment.critic_dim, environment.device)
        buf.decision_dt = environment.decision_dt
        buf.physics = copy.deepcopy(environment.config["physics"])
        buf.action_semantics = "ctbr-held-v3"
        physics_ticks = 0
        episode_start = environment.steps == 0
        for index in range(settings["rollout_steps"]):
            action, raw, lp, value = model.sample_tensor(obs, priv)
            buf.obs[index], buf.priv[index], buf.raw[index] = obs, priv, raw
            buf.logprob[index], buf.values[index], buf.episode_start[index] = lp, value, episode_start
            if labels:
                buf.teacher[index] = teacher.predict_tensor(obs)
            (next_obs, next_priv), reward, terminated, truncated, finished = environment.step(action, raw)
            physics_ticks += environment.last_physics_steps
            buf.reward[index], buf.terminated[index], buf.truncated[index] = reward, terminated, truncated
            buf.next_values[index] = model.values_tensor(next_priv)
            buf.elapsed[index] = environment.steps.float() * environment.dt
            for name, component in environment.reward_components.items():
                if name not in buf.reward_components:
                    buf.reward_components[name] = torch.zeros_like(buf.reward)
                buf.reward_components[name][index] = component
            for record in finished:
                buf.event_code[index, record["slot"]] = EVENT_CODES[record["reason"]]
            buf.diagnostic_steps = index + 1
            episode_start = terminated | truncated
            obs, priv = environment.reset(episode_start) if bool(episode_start.any()) else (next_obs, next_priv)
        buf.compute_gae(environment.gamma, environment.gae_lambda)
        return buf, obs, priv, physics_ticks

    @torch.no_grad()
    def logprob_check(model, buf):
        def errors(obs, raw, old):
            mean, std = model.distribution_parameters(obs)
            return (model.logprob(mean, std, raw) - old).abs()

        def stats(chunks):
            e = torch.cat(chunks)
            if not bool(torch.isfinite(e).all()):
                raise FloatingPointError("Nonfinite pre-update policy log probabilities")
            return dict(p50=e.median().item(), p99=e.quantile(0.99).item(), max=e.max().item())

        same = [errors(o, a, lp) for o, a, lp in zip(buf.obs, buf.raw, buf.logprob)]
        size = settings["batch_size"]
        mixed = [errors(o, a, lp) for o, a, lp in zip(buf.obs.flatten(0, 1).split(size),
                 buf.raw.flatten(0, 1).split(size), buf.logprob.flatten().split(size))]
        return dict(rollout_batch=stats(same), update_batch=stats(mixed))

    start = None
    try:
        progress("baseline")
        baseline = assess(teacher, (STATIC, ARC), validation, out / "baseline" / "validation")
        log_summary("mixed_bc", None, 0, "validation", baseline, None)
        if not eligible(baseline, baseline, ev):
            raise RuntimeError("The BC checkpoint does not reproduce the precision gate on both routes")
        if args.preflight:
            progress("completed")
            return

        for seed in plan["seeds"]:
            start = seed
            c = validate(route_config(original, ARC, args.device, seed, plan))
            torch.manual_seed(seed)
            common = new_model(c)
            common.load_state_dict(teacher.state_dict())
            progress("critic_warmup", seed=seed)
            _, data = evaluate(common, c, plan["warmup"]["episodes"], 810000 + seed,
                               stochastic=True, collect_mc=True)
            _, held = evaluate(common, c, plan["warmup"]["episodes"], 910000 + seed,
                               stochastic=True, collect_mc=True)
            pre = value_errors(common, held, settings["batch_size"])
            fit = fit_critic(common, data, plan["warmup"]["critic_updates"], settings["batch_size"])
            post = value_errors(common, held, settings["batch_size"])
            write_json(out / f"critic_warmup_s{seed}.json", dict(before=pre, after=post, fit=fit))
            del data, held
            state = copy.deepcopy(common.state_dict())
            opt = copy.deepcopy(common.critic_optimizer.state_dict())
            del common

            for arm in ARMS:
                model = new_model(c)
                model.load_state_dict(state)
                model.critic_optimizer.load_state_dict(copy.deepcopy(opt))
                if arm == "critic_only":
                    for group in model.actor_optimizer.param_groups:
                        group["lr"] = 0.0
                same_actor_keys = [key for key in state if key == "log_std" or key.startswith(("actor.", "predictor."))]
                directory = out / f"{arm}_s{seed}"
                directory.mkdir()
                torch.manual_seed(seed)
                environment = CTBRBatch(c, seed=seed)
                obs, priv = environment.observe()
                selected[arm, seed] = (None, 0)
                best_rank = rank(baseline)
                coefficient = settings["bc_coefficient"] if arm == "ppo_bc" else 0.0
                total_ticks = 0
                progress("ppo", arm, seed, 0)
                with (directory / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
                    for update in range(1, plan["updates"] + 1):
                        buf, obs, priv, ticks = collect(model, environment, obs, priv, coefficient > 0)
                        receipt = logprob_check(model, buf)
                        stats = model.update(buf, aux_pd_coef=coefficient)
                        if arm == "critic_only":
                            now = model.state_dict()
                            if any(not torch.equal(now[key], state[key]) for key in same_actor_keys):
                                raise RuntimeError("Frozen Actor or its prediction state changed")
                        total_ticks += ticks
                        model.critic_policy_match = arm == "critic_only"
                        metrics.write(json.dumps(dict(update=update, physics_steps=total_ticks,
                                                     logprob=receipt, stats=stats), allow_nan=False) + "\n")
                        metrics.flush()
                        completed += 1
                        progress("ppo", arm, seed, update)
                        if update % plan["evaluate_every"] and update != plan["updates"]:
                            continue
                        progress("validation", arm, seed, update)
                        result = assess(model, (STATIC, ARC), validation, directory / f"update_{update:04d}")
                        candidate = eligible(result, baseline, ev)
                        log_summary(arm, seed, update, "validation", result, candidate)
                        if candidate and rank(result) > best_rank:
                            best_rank = rank(result)
                            path = directory / "best_validation.pt"
                            save(path, model, c, {"ppo_updates": update, "physics_steps": total_ticks,
                                                 "critic_needs_calibration": not model.critic_policy_match}, arm)
                            selected[arm, seed] = (path, update)
                save(directory / "final.pt", model, c, {"ppo_updates": update, "physics_steps": total_ticks,
                     "critic_needs_calibration": not model.critic_policy_match}, arm)
        write_json(out / "selection.json", [dict(arm=arm, seed=seed, update=update,
                   checkpoint=str(path or checkpoint)) for (arm, seed), (path, update) in selected.items()])

        if args.final_test:
            progress("final_test")
            final_seeds = ev["test_seeds"]
            log_summary("mixed_bc", None, 0, "test", assess(teacher, PHASES, final_seeds,
                        out / "baseline" / "test"), None)
            for (arm, seed), (path, update) in selected.items():
                model = load(path, args.device)[0] if path else teacher
                result = assess(model, PHASES, final_seeds, out / f"{arm}_s{seed}" / "test")
                log_summary(arm, seed, update, "test", result, None)
    except (Exception, KeyboardInterrupt):
        progress("interrupted" if sys.exc_info()[0] is KeyboardInterrupt else "failed", seed=start)
        raise
    progress("completed")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_waypoint_curriculum ppo-compare", description=__doc__,
    )
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--mode", choices=("pilot", "confirm"), default="pilot")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
<<<<<<< HEAD
    parser.add_argument("--preflight", action="store_true", help="Verify the checkpoint and baseline without training")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args(argv)
=======
    parser.add_argument("--check-source", action="store_true", help="Check checkout compatibility without Torch")
    parser.add_argument("--preflight", action="store_true", help="Verify the checkpoint and baseline without training")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args(argv)
    if args.check_source:
        if args.dry_run or args.preflight or args.final_test:
            parser.error("--check-source runs on its own")
        try:
            fingerprint = check_route_source(ROOT)
        except IncompatibleRouteInput as error:
            parser.error(str(error))
        print(f"Survey v4 source verified: {ROOT} ({fingerprint[:12]})")
        return
>>>>>>> origin/feat/route-ppo-ablation
    try:
        plan = load_plan(args.plan, args.mode)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    if args.final_test and args.mode != "confirm":
        parser.error("--final-test requires --mode confirm")
    if args.preflight and args.final_test:
        parser.error("--preflight and --final-test are mutually exclusive")
    if args.dry_run:
        print(json.dumps({**plan, "arms": ARMS, "routes": PHASES,
                          "planned_updates": 0 if args.preflight else len(ARMS) * len(plan["seeds"]) * plan["updates"]}, indent=2))
        return
    if args.checkpoint is None or args.out is None:
        parser.error("--checkpoint and --out are required for training")
<<<<<<< HEAD
    run(args, plan)
=======
    try:
        run(args, plan)
    except IncompatibleRouteInput as error:
        parser.error(str(error))
>>>>>>> origin/feat/route-ppo-ablation


if __name__ == "__main__":
    main()
