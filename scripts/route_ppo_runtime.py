"""PyTorch adapter for the v4 single-aircraft survey simulator."""

import copy
from pathlib import Path

import torch

from pursuit_rl.ctbr import checkpoint as checkpoint_io
from pursuit_rl.ctbr.config import validate
from pursuit_rl.ctbr.environment import CTBRBatch
from pursuit_rl.ctbr.evaluation import evaluate
from pursuit_rl.ctbr.value_learning import fit_critic, value_errors
from pursuit_rl.learning.buffers import TensorRolloutBuffer
from scripts.route_ppo_plan import IncompatibleRouteInput, ROOT
from scripts.survey_route_profiles import ARC, route_task


EVENT_CODES = dict(success=1, speed=2, outside=3, tilt=4, tracking_failed=5,
                   invalid_target=6, timeout=7)


def load_teacher(path, device, source):
    expected_loader = ROOT / "pursuit_rl/ctbr/checkpoint.py"
    actual_loader = Path(checkpoint_io.__file__).resolve()
    if actual_loader != expected_loader:
        raise IncompatibleRouteInput(
            f"Python imported {actual_loader}, expected {expected_loader}. Check PYTHONPATH."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise IncompatibleRouteInput("CUDA is unavailable in this Python environment")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model, config, saved = checkpoint_io.load(path, device)
    except ValueError as error:
        raise IncompatibleRouteInput(f"Cannot load {path}: {error}") from error
    task = config["task"]
    if (saved["origin"], task["name"], task["kind"], task["objective"]) != (
        "mixed_route_bc", ARC, "circle", "sustained"
    ) or saved["source_fingerprint"] != source:
        raise IncompatibleRouteInput("Use mixed_route_v2/mixed_bc.pt from this v4 survey source")
    return model.eval().requires_grad_(False), config


def route_config(original, name, device, seed, plan):
    c = copy.deepcopy(original)
    c.update(device=device, seed=seed, num_envs=plan["training"]["envs"])
    c["task"].update(route_task(name))
    c["physics"]["max_speed"] = min(10.0, c["physics"]["max_speed"])
    c["training"].update({key: plan["training"][key] for key in (
        "rollout_steps", "batch_size", "epochs", "actor_lr", "critic_lr",
    )})
    return validate(c)


def warm_critic(original, teacher, plan, device, seed):
    config = route_config(original, ARC, device, seed, plan)
    torch.manual_seed(seed)
    model = checkpoint_io.new_model(config)
    model.load_state_dict(teacher.state_dict())
    warm = plan["warmup"]
    batch = plan["training"]["batch_size"]
    _, train_mc = evaluate(model, config, warm["episodes"], 810000 + seed,
                           stochastic=True, collect_mc=True)
    _, check_mc = evaluate(model, config, warm["episodes"], 910000 + seed,
                           stochastic=True, collect_mc=True)
    before = value_errors(model, check_mc, batch)
    fit = fit_critic(model, train_mc, warm["critic_updates"], batch)
    return config, model.state_dict(), model.critic_optimizer.state_dict(), dict(
        before=before, after=value_errors(model, check_mc, batch), fit=fit,
    )


def evaluate_policy(model, config, episodes, seed):
    training = model.training
    model.eval()
    try:
        return evaluate(model, config, episodes, seed)[0]
    finally:
        model.train(training)


@torch.no_grad()
def collect(model, environment, obs, priv, settings, teacher):
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
        if teacher is not None:
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
def logprob_check(model, buf, size):
    def errors(obs, raw, old):
        mean, std = model.distribution_parameters(obs)
        return (model.logprob(mean, std, raw) - old).abs()

    def stats(chunks):
        e = torch.cat(chunks)
        if not bool(torch.isfinite(e).all()):
            raise FloatingPointError("Nonfinite pre-update policy log probabilities")
        return dict(p50=e.median().item(), p99=e.quantile(0.99).item(), max=e.max().item())

    same = [errors(o, a, lp) for o, a, lp in zip(buf.obs, buf.raw, buf.logprob)]
    mixed = [errors(o, a, lp) for o, a, lp in zip(buf.obs.flatten(0, 1).split(size),
             buf.raw.flatten(0, 1).split(size), buf.logprob.flatten().split(size))]
    return dict(rollout_batch=stats(same), update_batch=stats(mixed))


def train_updates(config, state, optimizer, teacher, plan, arm):
    model = checkpoint_io.new_model(config)
    model.load_state_dict(state)
    model.critic_optimizer.load_state_dict(copy.deepcopy(optimizer))
    frozen = arm == "critic_only"
    if frozen:
        for group in model.actor_optimizer.param_groups:
            group["lr"] = 0.0
    actor_keys = tuple(
        key for key in state if key == "log_std" or key.startswith(("actor.", "predictor."))
    )
    coefficient = plan["training"]["bc_coefficient"] if arm == "ppo_bc" else 0.0
    seed = config["seed"]
    torch.manual_seed(seed)
    environment = CTBRBatch(config, seed=seed)
    obs, priv = environment.observe()
    physics_steps = 0
    for update in range(1, plan["updates"] + 1):
        buf, obs, priv, ticks = collect(
            model, environment, obs, priv, plan["training"], teacher if coefficient else None,
        )
        probability = logprob_check(model, buf, plan["training"]["batch_size"])
        stats = model.update(buf, aux_pd_coef=coefficient)
        del buf
        if frozen:
            now = model.state_dict()
            if any(not torch.equal(now[key], state[key]) for key in actor_keys):
                raise RuntimeError("Frozen Actor, action standard deviation or predictor changed")
        physics_steps += ticks
        model.critic_policy_match = frozen
        diagnostics = stats.pop("diagnostics", {})
        yield model, dict(update=update, physics_steps=physics_steps, logprob=probability, stats=stats,
                          anomaly_reasons=diagnostics.get("anomaly_reasons", []))


def save_candidate(path, model, config, receipt, arm):
    checkpoint_io.save(path, model, config, {
        "ppo_updates": receipt["update"],
        "physics_steps": receipt["physics_steps"],
        "critic_needs_calibration": not model.critic_policy_match,
    }, arm)


def load_candidate(path, device):
    return checkpoint_io.load(path, device)[0]
