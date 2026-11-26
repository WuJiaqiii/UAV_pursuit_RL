"""Compare PPO and PD assistance on a stationary virtual waypoint in simulation."""

import csv
from dataclasses import asdict
import json
import math

import torch
from torch.nn import functional as F

from pursuit_rl.learning.buffers import TensorRolloutBuffer
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.simulation.torch_device import resolve_torch_device
from pursuit_rl.waypoint import OBS_DIM, SCHEMA, Outcome, WaypointEnv, waypoint_pd
from pursuit_rl.waypoint_plan import BASELINES, EVAL_SEED


def _zero(env):
    return env.body.pos.new_zeros((env.n, 4))


@torch.no_grad()
def evaluate(config, controller, episodes, device, evaluation_seed=EVAL_SEED):
    env = WaypointEnv(config, episodes, evaluation_seed, device)
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
        "evaluation_seed": evaluation_seed,
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


def create_model(seed, device, ppo):
    torch.manual_seed(seed)
    config = dict(
        action_dim=4, hidden_actor=(128, 128), hidden_critic=(128, 128),
        actor_lr=ppo.actor_lr, critic_lr=ppo.critic_lr,
        batch_size=ppo.batch_size, epochs=ppo.epochs,
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


def pretrain(model, plan, seed):
    ppo, teacher = plan.ppo, plan.teacher
    env = WaypointEnv(plan.task, ppo.envs, seed + 1_000_000, model.device)
    observations, actions = [], []
    obs = env.observe()
    for _ in range(teacher.bc_steps):
        observations.append(obs)
        action = waypoint_pd(env)
        actions.append(action)
        _, _, done, _ = env.step(action)
        obs = env.reset(done)
    observations = torch.cat(observations)
    actions = torch.cat(actions)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
    loss = None
    for _ in range(teacher.bc_epochs):
        order = torch.randperm(len(actions), device=model.device)
        for ids in order.split(ppo.batch_size):
            loss = F.smooth_l1_loss(torch.tanh(model.actor(observations[ids])), actions[ids])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
    return {"demonstration_decisions": len(actions), "last_bc_minibatch_loss": loss.item()}


def collect(env, model, ppo, label_with_pd):
    buf = TensorRolloutBuffer(ppo.rollout_steps, env.n, OBS_DIM, OBS_DIM, model.device)
    obs = env.observe()
    for i in range(ppo.rollout_steps):
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
        obs = env.reset(done)
    # At the finite task deadline the time-to-go is zero and the task ends.
    # At a rollout boundary an unfinished environment retains its bootstrap.
    buf.compute_gae(gamma=ppo.gamma, lam=ppo.gae_lambda)
    return buf


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(directory, name, model, config, seed):
    torch.save({"schema": SCHEMA, "waypoint_config": asdict(config),
                "seed": seed, "model_config": model.config,
                "state_dict": model.state_dict()}, directory / f"{name}.pt")


def run_suite(plan, root, stage=None):
    config, experiment, ppo = plan.task, plan.experiment, plan.ppo
    device = resolve_torch_device(experiment.device)
    torch.set_num_threads(experiment.cpu_threads)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"Output directory is already in use: {root}")
    # Exclusive creation prevents concurrent launchers from writing the same suite.
    with (root / "plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan.document(), stream, indent=2, allow_nan=False)
        stream.write("\n")
    cases = [
        (arm, seed) for arm in experiment.arms
        for seed in ((None,) if arm in BASELINES else experiment.seeds)
    ]
    total_updates = ppo.updates * sum(seed is not None for _, seed in cases)
    completed_updates = 0
    summary = []

    def record(arm, phase, seed, controller, directory):
        report = evaluate(
            config, controller, experiment.episodes, device, experiment.evaluation_seed,
        )
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
            active_seed=seed, phase=phase, stage=stage, device=str(device),
            completed_ppo_updates=completed_updates,
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
        model = create_model(seed, device, ppo)

        def controller(current_env):
            return model.predict_tensor(current_env.observe())

        if arm == "bc_ppo":
            progress(index, arm, seed, "behavior_cloning")
            write_json(directory / "bc.json", pretrain(model, plan, seed))
            save_checkpoint(directory, "bc", model, config, seed)
            record(arm, "after_bc", seed, controller, directory)
        else:
            record(arm, "initial", seed, controller, directory)

        torch.manual_seed(seed + 2_000_000)
        env = WaypointEnv(config, ppo.envs, seed, device)
        aux_coef = plan.teacher.aux_coef if arm == "aux_ppo" else 0.0
        with (directory / "metrics.jsonl").open("x", encoding="utf-8") as stream:
            for update in range(1, ppo.updates + 1):
                buf = collect(env, model, ppo, aux_coef > 0)
                stats = model.update(buf, aux_pd_coef=aux_coef)
                stats["update"] = update
                stats["episode_count"] = int(buf.terminated.sum().item())
                stats["goal_reached_count"] = int((buf.event_code == Outcome.GOAL_REACHED).sum().item())
                stream.write(json.dumps(stats, allow_nan=False) + "\n")
                stream.flush()
                completed_updates += 1
                progress(index, arm, seed, "ppo")
                if update == 1 or update == ppo.updates or update % 10 == 0:
                    print(f"  PPO updates {completed_updates}/{total_updates} overall", flush=True)
        progress(index, arm, seed, "evaluation")
        save_checkpoint(directory, "final", model, config, seed)
        record(arm, "final", seed, controller, directory)
    progress(len(cases) + 1, None, None, "completed")
    return summary
