"""Geometric BC/DAgger, MC critic warm-up, teacher-free on-policy PPO."""

import copy
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from pursuit_rl.learning.buffers import TensorRolloutBuffer
from .config import fingerprint, interface
from .checkpoint import new_model, save, load, write_json, source_fingerprint
from .environment import CTBRBatch
from .control import teacher
from .evaluation import evaluate, score, improved
from .errors import ExperimentGateError


def emit(root, event, **fields):
    row = dict(event=event, **fields)
    with (root / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in row.items() if k not in ("diagnostics", "parent")},
            allow_nan=False,
        ),
        flush=True,
    )


@torch.no_grad()
def imitation_data(c, model=None, seed=10000):
    env = CTBRBatch(c, seed=seed)
    obs, priv = env.observe()
    states = []
    labels = []
    tick = 0
    count = 0
    segment_ticks = round(c["training"]["bc_segment_seconds"] / c["physics"]["dt"])
    while count < c["training"]["bc_samples"]:
        target = teacher(env)
        if tick % c["training"]["bc_stride"] == 0:
            clean = obs.clone()
            clean[:, 21:] = 0
            states.append(clean)
            labels.append(target.clone())
            count += env.n
        action = (
            target
            if model is None
            else model.sample_tensor(obs, priv, deterministic=True)[0]
        )
        raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        (obs, priv), _, done, _, _ = env.step(action, raw)
        if bool(done.any()):
            obs, priv = env.reset(done)
        tick += 1
        if tick % segment_ticks == 0:
            obs, priv = env.reset()
    return (
        torch.cat(states)[: c["training"]["bc_samples"]],
        torch.cat(labels)[: c["training"]["bc_samples"]],
        tick * env.n,
    )


def fit_actor(model, data, updates, batch_size):
    """Condition supervised regression, then fold normalization into the first layer.

    Only the teacher's sufficient current kinematics are used. Actuator state,
    time, boundaries, goals and raw-action memory are excluded to prevent
    shortcuts. PPO still receives the complete original observation.
    """
    x, y = data
    first = model.actor[0]
    x = x.clone()
    x[:, 21:] = 0
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(0.01)
    scale[21:] = 1
    with torch.no_grad():
        first.weight[:, 21:] = 0
        original = first.weight.clone()
        first.weight.mul_(scale)
        first.bias.add_(original @ center)
    normalized = (x - center) / scale
    output_scale = y.std(0, unbiased=False).clamp_min(0.02)
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    loss = None
    try:
        for _ in range(updates):
            ids = torch.randint(len(x), (min(batch_size, len(x)),), device=model.device)
            loss = (
                ((torch.tanh(model.actor(normalized[ids])) - y[ids]) / output_scale)
                .square()
                .mean()
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite imitation loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.actor.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
    finally:
        with torch.no_grad():
            first.weight.div_(scale)
            first.bias.sub_(first.weight @ center)
    return float(loss.detach()) if loss is not None else None


def fit_critic(model, data, updates, batch_size):
    x, y = data
    with torch.no_grad():
        before = float((model.values_tensor(x) - y).square().mean().sqrt())
    # Episode starts are only one of ~1000 MC samples in a 20 s flight.
    # Oversample them so acquisition-state value is not sacrificed to steady states.
    starts = (x[:, 31] >= x[:, 31].max() - 1e-8).nonzero(as_tuple=True)[0]
    size = min(batch_size, len(x))
    n_start = max(1, size // 8)
    for _ in range(updates):
        ids = torch.cat(
            (
                torch.randint(len(x), (size - n_start,), device=model.device),
                starts[torch.randint(len(starts), (n_start,), device=model.device)],
            )
        )
        loss = F.huber_loss(model.critic(x[ids]).squeeze(-1), y[ids])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite critic fit")
        model.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.critic.parameters(), 0.5, error_if_nonfinite=True
        )
        model.critic_optimizer.step()
    with torch.no_grad():
        after = float((model.values_tensor(x) - y).square().mean().sqrt())
    return dict(
        samples=len(y),
        updates=updates,
        fit_rmse_before=before,
        fit_rmse_after=after,
        initial_state_samples=len(starts),
        initial_state_batch_fraction=n_start / size,
    )


def initialization_validation(model, c, root, progress, label):
    report, _ = evaluate(
        model,
        c,
        c["training"]["initialization_episodes"],
        c["training"]["validation_seed"],
    )
    progress["validation_physics_steps"] += sum(
        round(e["time"] / c["physics"]["dt"]) for e in report["episodes_detail"]
    )
    write_json(root / f"initialization_{label}.json", report)
    emit(
        root,
        "initialization_validation",
        label=label,
        success_rate=report["success_rate"],
        tail_rmse=report["mean_tail_position_rmse"],
        failures=report["failures"],
    )
    return report


def train(c, out, resume=None, init_from=None):
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    root = Path(out).resolve()
    tr = c["training"]
    if resume and Path(resume).resolve().parent.parent != root:
        raise ValueError("Resume inside the checkpoint original run directory")
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError("New run requires empty output")
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    if not resume:
        write_json(root / "config.json", c)
    torch.manual_seed(c["seed"])
    progress = dict(
        physics_steps=0,
        decision_transitions=0,
        ppo_updates=0,
        bc_updates=0,
        accepted_bc_updates=0,
        rejected_dagger_rounds=0,
        imitation_physics_steps=0,
        mc_physics_steps=0,
        validation_physics_steps=0,
        completed_episodes=0,
        successful_episodes=0,
        parent_physics_steps=0,
        cumulative_ppo_physics_steps=0,
        parent=None,
        best_score=[-1.0, -1.0, -1e9],
        best_origin=None,
    )
    if resume:
        model, old, saved = load(resume, c["device"], optimizers=True)
        comparison = copy.deepcopy(c)
        comparison["training"]["physics_steps"] = old["training"]["physics_steps"]
        if fingerprint(comparison) != fingerprint(old):
            raise ValueError("Resume allows only total budget/device changes")
        if saved["source_fingerprint"] != source_fingerprint():
            raise ValueError(
                "Resume source code changed; start a new run with explicit init-from"
            )
        if tr["physics_steps"] < saved["progress"]["physics_steps"]:
            raise ValueError("Resume budget below completed work")
        progress.update(saved["progress"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and str(model.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if (
            tr["physics_steps"] == progress["physics_steps"]
            and (root / "training_summary.json").exists()
        ):
            return json.loads((root / "training_summary.json").read_text())
        emit(
            root,
            "resume",
            physics_steps=progress["physics_steps"],
            note="Unfinished environments restart; completed updates/optimizers retained",
        )
    else:
        model = new_model(c)
        if init_from:
            parent, old, saved = load(init_from, c["device"])
            if interface(old) != interface(c):
                raise ValueError(
                    "Transfer would change action, observation, dynamics, reward or exploration contract"
                )
            model.actor.load_state_dict(parent.actor.state_dict())
            model.critic.load_state_dict(parent.critic.state_dict())
            progress["parent_physics_steps"] = saved["progress"][
                "cumulative_ppo_physics_steps"
            ]
            progress["parent"] = dict(
                path=str(Path(init_from).resolve()),
                sha256=hashlib.sha256(Path(init_from).read_bytes()).hexdigest(),
                origin=saved["origin"],
                ppo_updates=saved["progress"]["ppo_updates"],
                critic="warm weights, fresh optimizers + new MC calibration",
            )
            origin = "transfer_initial"
            del parent
            if tr["transfer_dagger_updates"]:
                prior = initialization_validation(
                    model, c, root, progress, "transfer_before"
                )
                actor_before = {
                    k: v.detach().clone() for k, v in model.actor.state_dict().items()
                }
                x, y, interactions = imitation_data(c, model, c["seed"] + 30000)
                loss = fit_actor(
                    model, (x, y), tr["transfer_dagger_updates"], tr["batch_size"]
                )
                progress["bc_updates"] += tr["transfer_dagger_updates"]
                progress["imitation_physics_steps"] += interactions
                candidate = initialization_validation(
                    model, c, root, progress, "transfer_after"
                )
                accepted = improved(candidate, score(prior))
                if accepted:
                    origin = "transfer_dagger_initial"
                    progress["accepted_bc_updates"] += tr["transfer_dagger_updates"]
                else:
                    model.actor.load_state_dict(actor_before)
                    progress["rejected_dagger_rounds"] += 1
                emit(
                    root,
                    "transfer_dagger",
                    updates=tr["transfer_dagger_updates"],
                    samples=len(x),
                    loss=loss,
                    accepted=accepted,
                    loss_space="standardized_action_mse",
                )
                del x, y
        else:
            origin = "bc_initial" if tr["bc_updates"] else "random_initial"
            # The teacher uses only current kinematics (first 21 features).
            with torch.no_grad():
                model.actor[0].weight[:, 21:] = 0
            if tr["bc_updates"]:
                x, y, interactions = imitation_data(c, seed=c["seed"] + 10000)
                loss = fit_actor(model, (x, y), tr["bc_updates"], tr["batch_size"])
                progress["bc_updates"] += tr["bc_updates"]
                progress["accepted_bc_updates"] += tr["bc_updates"]
                progress["imitation_physics_steps"] += interactions
                emit(
                    root,
                    "bc",
                    loss=loss,
                    samples=len(x),
                    loss_space="standardized_action_mse",
                )
                prior = initialization_validation(model, c, root, progress, "bc")
                for i in range(tr["dagger_rounds"]):
                    actor_before = {
                        k: v.detach().clone()
                        for k, v in model.actor.state_dict().items()
                    }
                    dx, dy, interactions = imitation_data(
                        c, model, c["seed"] + 20000 + i
                    )
                    x = torch.cat((x, dx))
                    y = torch.cat((y, dy))
                    loss = fit_actor(
                        model, (x, y), tr["dagger_updates"], tr["batch_size"]
                    )
                    progress["bc_updates"] += tr["dagger_updates"]
                    progress["imitation_physics_steps"] += interactions
                    candidate = initialization_validation(
                        model, c, root, progress, f"dagger_{i+1}"
                    )
                    accepted = improved(candidate, score(prior))
                    if accepted:
                        prior = candidate
                        origin = "bc_dagger_initial"
                        progress["accepted_bc_updates"] += tr["dagger_updates"]
                    else:
                        model.actor.load_state_dict(actor_before)
                        progress["rejected_dagger_rounds"] += 1
                    emit(
                        root,
                        "dagger",
                        round=i + 1,
                        loss=loss,
                        samples=len(x),
                        accepted=accepted,
                        loss_space="standardized_action_mse",
                    )
                del x, y
        calibration, data = evaluate(
            model,
            c,
            tr["mc_episodes"],
            810000 + c["seed"],
            stochastic=True,
            collect_mc=True,
        )
        progress["mc_physics_steps"] += sum(
            round(e["time"] / c["physics"]["dt"])
            for e in calibration["episodes_detail"]
        )
        fit = fit_critic(model, data, tr["critic_updates"], tr["batch_size"])
        del data
        heldout, _ = evaluate(
            model, c, tr["mc_episodes"], 910000 + c["seed"], stochastic=True
        )
        write_json(root / "critic_warmup_heldout.json", heldout)
        progress["mc_physics_steps"] += sum(
            round(e["time"] / c["physics"]["dt"]) for e in heldout["episodes_detail"]
        )
        emit(
            root,
            "critic_warmup",
            fit=fit,
            mc_rmse_before_fit=calibration["critic_mc_rmse"],
            heldout_rmse=heldout["critic_mc_rmse"],
            heldout_bias=heldout["critic_bias"],
        )
        initial, _ = evaluate(
            model, c, tr["validation_episodes"], tr["validation_seed"]
        )
        write_json(root / "initial_validation.json", initial)
        progress["validation_physics_steps"] += sum(
            round(e["time"] / c["physics"]["dt"]) for e in initial["episodes_detail"]
        )
        progress["best_score"] = list(score(initial))
        progress["best_origin"] = origin
        progress["cumulative_ppo_physics_steps"] = progress["parent_physics_steps"]
        for name in ("initial", "best", "latest"):
            save(ckpt / f"{name}.pt", model, c, progress, origin)
        emit(
            root,
            "initial_validation",
            origin=origin,
            success_rate=initial["success_rate"],
            tail_rmse=initial["mean_tail_position_rmse"],
        )
    if (
        not init_from
        and progress["ppo_updates"] == 0
        and progress["parent"] is None
        and tr["bc_updates"]
    ):
        initial = json.loads((root / "initial_validation.json").read_text())
    if (
        not init_from
        and progress["ppo_updates"] == 0
        and progress["parent"] is None
        and tr["bc_updates"]
        and initial["success_rate"] < tr["initialization_success_floor"]
    ):
        details = dict(
            event="initialization_gate_failed",
            phase=c["task"]["name"],
            success_rate=initial["success_rate"],
            minimum_success=tr["initialization_success_floor"],
            failures=initial["failures"],
            report=str(root / "initial_validation.json"),
            checkpoint=str(ckpt / "initial.pt"),
            message="Initialization failed full-episode validation; PPO was not started. Fix initialization or use a new explicitly configured experiment.",
        )
        write_json(root / "gate_failure.json", details)
        raise ExperimentGateError(details)
    write_json(root / "config.json", c)
    env = CTBRBatch(c, seed=c["seed"] + 7919 * progress["ppo_updates"])
    obs, priv = env.observe()
    episode_start = torch.ones(env.n, dtype=torch.bool, device=env.device)
    next_validation = (progress["physics_steps"] // tr["validation_every"] + 1) * tr[
        "validation_every"
    ]
    while progress["physics_steps"] < tr["physics_steps"]:
        ticks = min(
            tr["rollout_steps"],
            math.ceil((tr["physics_steps"] - progress["physics_steps"]) / env.n),
        )
        buf = TensorRolloutBuffer(
            ticks,
            env.n,
            env.actor_dim,
            env.critic_dim,
            env.device,
            action_dim=4,
            diagnostics={
                "value_abs_threshold": 100.0,
                "target_abs_threshold": 100.0,
                "advantage_abs_threshold": 100.0,
            },
        )
        buf.physics = copy.deepcopy(c["physics"])
        buf.action_semantics = "ctbr-v2"
        records = []
        for tick in range(ticks):
            try:
                action, raw, lp, value = model.sample_tensor(obs, priv)
                (nobs, npriv), reward, term, trunc, completed = env.step(action, raw)
            except Exception as exc:
                if not (root / "first_anomaly.pt").exists():
                    snapshot = buf.diagnostic_snapshot()
                    snapshot["tensors"] = {
                        k: v[:tick] for k, v in snapshot["tensors"].items()
                    }
                    torch.save(
                        dict(
                            phase="collection",
                            error=repr(exc),
                            config=c,
                            progress=progress,
                            model=model.state_dict(),
                            obs=obs.cpu(),
                            priv=priv.cpu(),
                            buffer=snapshot,
                        ),
                        root / "first_anomaly.pt",
                    )
                raise
            for name, data in [
                ("obs", obs),
                ("priv", priv),
                ("raw", raw),
                ("logprob", lp),
                ("values", value),
                ("reward", reward),
                ("terminated", term),
                ("truncated", trunc),
                ("next_values", model.values_tensor(npriv)),
                ("episode_start", episode_start),
                ("elapsed", env.steps.float() * env.dt),
            ]:
                getattr(buf, name)[tick] = data
            buf.actuator_state[tick, :, 0] = env.plant.pursuer.thrust
            buf.actuator_state[tick, :, 1:] = env.plant.pursuer.rate
            for key, data in env.reward_components.items():
                if key not in buf.reward_components:
                    buf.reward_components[key] = torch.zeros_like(buf.reward)
                buf.reward_components[key][tick] = data
            for rec in completed:
                buf.event_code[tick, rec["slot"]] = {
                    "success": 1,
                    "speed": 2,
                    "outside": 3,
                    "tilt": 4,
                    "tracking_failed": 5,
                    "invalid_target": 6,
                }[rec["reason"]]
            buf.diagnostic_steps = tick + 1
            records.extend(completed)
            episode_start = term | trunc
            obs, priv = env.reset(episode_start) if completed else (nobs, npriv)
        buf.compute_gae(env.gamma, env.gae_lambda)
        before = dict(
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            actor_optimizer=copy.deepcopy(model.actor_optimizer.state_dict()),
            critic_optimizer=copy.deepcopy(model.critic_optimizer.state_dict()),
            torch_rng=torch.get_rng_state(),
            progress=copy.deepcopy(progress),
        )
        try:
            stats = model.update(buf, aux_pd_coef=0.0)
        except Exception as exc:
            if not (root / "first_anomaly.pt").exists():
                torch.save(
                    dict(
                        before=before,
                        buffer=buf.diagnostic_snapshot(),
                        error=repr(exc),
                        config=c,
                    ),
                    root / "first_anomaly.pt",
                )
            raise
        progress["physics_steps"] += ticks * env.n
        progress["decision_transitions"] += ticks * env.n
        progress["ppo_updates"] += 1
        progress["cumulative_ppo_physics_steps"] = (
            progress["parent_physics_steps"] + progress["physics_steps"]
        )
        progress["completed_episodes"] += len(records)
        progress["successful_episodes"] += sum(r["success"] for r in records)
        diagnostics = stats.get("diagnostics")
        if (
            diagnostics
            and diagnostics.get("anomaly_reasons")
            and not (root / "first_anomaly.pt").exists()
        ):
            torch.save(
                dict(
                    before=before,
                    buffer=buf.diagnostic_snapshot(),
                    diagnostics=diagnostics,
                    config=c,
                ),
                root / "first_anomaly.pt",
            )
        emit(
            root,
            "ppo_update",
            physics_steps=progress["physics_steps"],
            ppo_updates=progress["ppo_updates"],
            nominal_seconds_per_env=progress["physics_steps"] * env.dt / env.n,
            completed_episodes=len(records),
            successes=sum(r["success"] for r in records),
            completed_success_rate=(
                sum(r["success"] for r in records) / len(records) if records else None
            ),
            metrics={k: v for k, v in stats.items() if k != "diagnostics"},
            diagnostics=diagnostics,
        )
        if (
            progress["physics_steps"] >= next_validation
            or progress["physics_steps"] >= tr["physics_steps"]
        ):
            report, _ = evaluate(
                model, c, tr["validation_episodes"], tr["validation_seed"]
            )
            write_json(root / f"validation_{progress['physics_steps']}.json", report)
            progress["validation_physics_steps"] += sum(
                round(e["time"] / env.dt) for e in report["episodes_detail"]
            )
            cal, _ = evaluate(
                model,
                c,
                tr["mc_episodes"],
                810000 + c["seed"] + progress["ppo_updates"],
                stochastic=True,
            )
            write_json(root / f"calibration_{progress['physics_steps']}.json", cal)
            progress["mc_physics_steps"] += sum(
                round(e["time"] / env.dt) for e in cal["episodes_detail"]
            )
            emit(
                root,
                "validation",
                success_rate=report["success_rate"],
                tail_rmse=report["mean_tail_position_rmse"],
                stochastic_mc_rmse=cal["critic_mc_rmse"],
                stochastic_bias=cal["critic_bias"],
            )
            if improved(report, progress["best_score"]):
                progress["best_score"] = list(score(report))
                progress["best_origin"] = "ppo"
                save(ckpt / "best.pt", model, c, progress, "ppo")
            next_validation += tr["validation_every"]
        save(ckpt / "latest.pt", model, c, progress, "ppo")
        if progress["ppo_updates"] % tr["save_every_updates"] == 0:
            save(
                ckpt / f"update_{progress['ppo_updates']:05d}.pt",
                model,
                c,
                progress,
                "ppo",
            )
    result = dict(
        schema="pursuit-direct-ctbr-training-v2",
        **progress,
        total_local_interactions=sum(
            progress[k]
            for k in (
                "physics_steps",
                "imitation_physics_steps",
                "mc_physics_steps",
                "validation_physics_steps",
            )
        ),
        unfinished_episodes=int((env.steps > 0).sum()),
        maximum_unfinished_elapsed_s=float(env.steps.max()) * env.dt,
        source_fingerprint=source_fingerprint(),
        note="Initial/best/latest must be compared; completed training episodes exclude unfinished trajectories.",
    )
    write_json(root / "training_summary.json", result)
    return result
