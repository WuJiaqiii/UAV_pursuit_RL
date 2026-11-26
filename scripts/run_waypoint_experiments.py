"""Compare PPO and PD assistance on a stationary virtual waypoint in simulation."""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from pursuit_rl.learning.buffers import TensorRolloutBuffer
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.simulation.torch_device import resolve_torch_device
from pursuit_rl.waypoint import OBS_DIM, SCHEMA, Outcome, WaypointConfig, WaypointEnv, waypoint_pd


ARMS = ("zero", "pd", "ppo", "bc_ppo", "aux_ppo")
EVAL_SEED = 9_000_000


def _zero(env):
    return env.body.pos.new_zeros((env.n, 4))


@torch.no_grad()
def evaluate(config, controller, episodes, device):
    env = WaypointEnv(config, episodes, EVAL_SEED, device)
    initial_waypoints = env.waypoint.cpu().tolist()
    for _ in range(math.ceil(env.deadline / config.decision_ticks)):
        env.step(controller(env))
        if bool((env.outcome != Outcome.ACTIVE).all()):
            break
    counts = torch.bincount(env.outcome, minlength=int(max(Outcome)) + 1).tolist()
    if counts[Outcome.ACTIVE]:
        raise RuntimeError("Evaluation contains a nonterminal episode")
    successes = counts[Outcome.GOAL_REACHED]
    return {
        "evaluation_seed": EVAL_SEED,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_minimum_distance_m": env.minimum_distance.mean().item(),
        "max_ground_speed_mps": env.peak_speed.max().item(),
        "outcomes": {event.name.lower(): counts[event] for event in Outcome if event},
        "cases": [
            {"waypoint_ned_m": goal, "outcome": Outcome(code).name.lower(),
             "minimum_distance_m": minimum, "max_ground_speed_mps": peak,
             "elapsed_s": tick * env.dt}
            for goal, code, minimum, peak, tick in zip(
                initial_waypoints, env.outcome.tolist(), env.minimum_distance.tolist(),
                env.peak_speed.tolist(), env.ticks.tolist(),
            )
        ],
    }


def create_model(seed, device, batch_size):
    torch.manual_seed(seed)
    config = dict(
        action_dim=4, hidden_actor=(128, 128), hidden_critic=(128, 128),
        actor_lr=1e-4, critic_lr=3e-4, batch_size=batch_size, epochs=3,
        clip=0.15, target_kl=0.015, ent_coef=0.0, max_grad_norm=0.5,
        log_std_init=math.log(0.12), log_std_min=math.log(0.04),
        log_std_max=math.log(0.60), separate_critic_updates=True,
    )
    model = AsymmetricPPO(OBS_DIM, OBS_DIM, config, device=device)
    for layer in model.actor:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            torch.nn.init.zeros_(layer.bias)
    torch.nn.init.orthogonal_(model.actor[-1].weight, gain=0.02)
    return model


def pretrain(model, config, args, seed):
    env = WaypointEnv(config, args.envs, seed + 1_000_000, model.device)
    observations, actions = [], []
    for _ in range(args.bc_steps):
        observations.append(env.observe())
        action = waypoint_pd(env)
        actions.append(action)
        _, _, done, _ = env.step(action)
        env.reset(done)
    observations = torch.cat(observations)
    actions = torch.cat(actions)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    loss = None
    for _ in range(args.bc_epochs):
        order = torch.randperm(len(actions), device=model.device)
        for ids in order.split(args.batch_size):
            loss = F.smooth_l1_loss(torch.tanh(model.actor(observations[ids])), actions[ids])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
    return {"demonstration_decisions": len(actions), "last_bc_minibatch_loss": loss.item()}


def collect(env, model, steps, label_with_pd):
    buf = TensorRolloutBuffer(steps, env.n, OBS_DIM, OBS_DIM, model.device)
    for i in range(steps):
        obs = env.observe()
        buf.episode_start[i] = env.ticks == 0
        action, raw, logprob, values = model.sample_tensor(obs, obs)
        if label_with_pd:
            buf.teacher[i] = waypoint_pd(env)
        last_obs, reward, done, elapsed = env.step(action)
        buf.obs[i], buf.priv[i] = obs, obs
        buf.raw[i], buf.logprob[i], buf.values[i] = raw, logprob, values
        buf.reward[i], buf.terminated[i], buf.elapsed[i] = reward, done, elapsed
        buf.event_code[i] = env.outcome
        buf.next_values[i] = model.values_tensor(last_obs)
        env.reset(done)
    # At the finite task deadline the time-to-go is zero and the task ends.
    # At a rollout boundary an unfinished environment retains its bootstrap.
    buf.compute_gae(gamma=0.995, lam=0.97)
    return buf


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(directory, name, model, config, seed):
    torch.save({"schema": SCHEMA, "waypoint_config": asdict(config),
                "seed": seed, "model_config": model.config,
                "state_dict": model.state_dict()}, directory / f"{name}.pt")


def run_suite(args):
    device = resolve_torch_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    config = WaypointConfig(
        radius_m=args.radius,
        horizontal_distance_m=tuple(args.distance),
        timeout_s=args.episode_seconds,
    )
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"Output directory is already in use: {root}")
    # Exclusive creation prevents concurrent launchers from writing the same suite.
    with (root / "plan.json").open("x", encoding="utf-8") as stream:
        plan = vars(args) | {"out": str(root), "device": str(device), "schema": SCHEMA,
                             "waypoint": asdict(config), "evaluation_seed": EVAL_SEED}
        json.dump(plan, stream, indent=2, allow_nan=False)
    total_updates = args.updates * len(args.seeds) * len(set(args.arms) - {"zero", "pd"})
    completed_updates = 0
    cases = [(arm, None if arm in ("zero", "pd") else seed)
             for arm in args.arms for seed in ([None] if arm in ("zero", "pd") else args.seeds)]
    summary = []

    def record(arm, phase, seed, controller, directory):
        report = evaluate(config, controller, args.episodes, device)
        write_json(directory / f"eval_{phase}.json", report)
        summary.append({
            "arm": arm, "phase": phase, "training_seed": seed,
            "episodes": report["episodes"], "successes": report["successes"],
            "success_rate": report["success_rate"],
            "mean_minimum_distance_m": report["mean_minimum_distance_m"],
            "max_ground_speed_mps": report["max_ground_speed_mps"],
            **report["outcomes"],
        })
        temporary = root / "summary.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        temporary.replace(root / "summary.csv")
        print(f"{arm} {phase}: {report['successes']}/{report['episodes']} waypoints reached", flush=True)

    def progress(index, arm, seed, phase):
        write_json(root / "progress.json", dict(
            completed_runs=index - 1, total_runs=len(cases), active_arm=arm,
            active_seed=seed, phase=phase, completed_ppo_updates=completed_updates,
            planned_ppo_updates=total_updates,
        ))

    for index, (arm, seed) in enumerate(cases, 1):
        directory = root / (arm if seed is None else f"{arm}_seed{seed}")
        directory.mkdir()
        print(f"[{index}/{len(cases)}] {directory.name}", flush=True)
        progress(index, arm, seed, "evaluation" if seed is None else "setup")
        if seed is None:
            record(arm, "baseline", seed, _zero if arm == "zero" else waypoint_pd, directory)
            continue
        model = create_model(seed, device, args.batch_size)

        def controller(current_env):
            return model.predict_tensor(current_env.observe())

        if arm == "bc_ppo":
            progress(index, arm, seed, "behavior_cloning")
            write_json(directory / "bc.json", pretrain(model, config, args, seed))
            save_checkpoint(directory, "bc", model, config, seed)
            record(arm, "after_bc", seed, controller, directory)
        else:
            record(arm, "initial", seed, controller, directory)

        torch.manual_seed(seed + 2_000_000)
        env = WaypointEnv(config, args.envs, seed, device)
        aux_coef = args.aux_coef if arm == "aux_ppo" else 0.0
        with (directory / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for update in range(1, args.updates + 1):
                buf = collect(env, model, args.rollout_steps, aux_coef > 0)
                stats = model.update(buf, aux_pd_coef=aux_coef)
                stats["update"] = update
                stats["episode_count"] = int(buf.terminated.sum().item())
                stats["goal_reached_count"] = int((buf.event_code == Outcome.GOAL_REACHED).sum().item())
                stream.write(json.dumps(stats, allow_nan=False) + "\n")
                stream.flush()
                completed_updates += 1
                progress(index, arm, seed, "ppo")
                if update == 1 or update == args.updates or update % 10 == 0:
                    print(f"  PPO updates {completed_updates}/{total_updates} overall", flush=True)
        progress(index, arm, seed, "evaluation")
        save_checkpoint(directory, "final", model, config, seed)
        record(arm, "final", seed, controller, directory)
    progress(len(cases) + 1, None, None, "completed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--distance", type=float, nargs=2, default=(2.0, 5.0), help="Initial horizontal range in metres")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bc-steps", type=int, default=256)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--aux-coef", type=float, default=0.1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.envs, args.episodes, args.rollout_steps, args.updates, args.batch_size,
           args.bc_steps, args.bc_epochs, args.cpu_threads) < 1:
        parser.error("All budgets and counts must be positive")
    if not math.isfinite(args.aux_coef) or args.aux_coef <= 0:
        parser.error("--aux-coef must be finite and positive")
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Arms and seeds must be unique")
    if not all(0 <= seed < 1_000_000 for seed in args.seeds):
        parser.error("Training seeds must be between 0 and 999999")
    try:
        return run_suite(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
