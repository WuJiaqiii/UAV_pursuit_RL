"""BC initialization, complete-episode critic anchoring, then on-policy PPO.

Teacher actions are training labels only. PPO and learned-policy evaluation use
only network actions through the declared decoder, without target-tracking PD.
"""

import copy
import json
import hashlib
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from pursuit_rl.config.rescue import rescue_contract, rescue_fingerprint
from pursuit_rl.config.challenge import assert_transfer_compatible
from pursuit_rl.artifacts.provenance import rescue_source_fingerprint
from pursuit_rl.learning.rescue_checkpoint import (
    new_rescue_model,
    save_rescue_checkpoint,
    load_rescue_checkpoint,
)
from pursuit_rl.learning.buffers import TensorRolloutBuffer
from pursuit_rl.evaluation.rescue import evaluate_rescue
from pursuit_rl.simulation.rescue_environment import RescueBatch
from pursuit_rl.simulation.rescue_control import reference_action


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )


def emit(out, event, **data):
    record = dict(event=event, **data)
    with (out / "metrics.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    public = {
        k: v
        for k, v in record.items()
        if k not in ("episodes_detail", "diagnostics", "parent_checkpoint")
    }
    print(json.dumps(public, ensure_ascii=False, allow_nan=False), flush=True)


@torch.no_grad()
def collect_imitation_data(cfg, profile, samples, seed, model=None):
    env = RescueBatch(cfg, profile, seed=seed)
    obs, priv = env.observe()
    states = []
    labels = []
    count = 0
    tick = 0
    episodes = []
    stride = max(1, math.ceil(profile["bc_stride_physics_steps"] / env.interval))
    while count < samples:
        teacher = reference_action(env.env, profile)
        if tick % stride == 0:
            # Raw-command memory is observed for AR likelihoods, but cannot
            # become a shortcut that copies the teacher's preceding action.
            clean = obs.clone()
            clean[:, env.memory_start :] = 0.0
            states.append(clean)
            labels.append(teacher.clone())
            count += env.n
        if model is None:
            action = teacher
            raw = torch.atanh(action.clamp(-0.999999, 0.999999))
        else:
            action, raw, _, _ = model.sample_tensor(obs, priv, deterministic=True)
        (obs, priv), _, done, _, records = env.step(action, raw)
        episodes.extend(records)
        if bool(done.any()):
            obs, priv = env.reset(done)
        tick += 1
    return (
        torch.cat(states)[:samples],
        torch.cat(labels)[:samples],
        {
            "samples": samples,
            "physics_steps": tick * env.n * env.interval,
            "completed_episodes": len(episodes),
            "successes": sum(r["success"] for r in episodes),
        },
    )


def fit_imitation(model, states, labels, profile, updates):
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=profile["bc_lr"])
    losses = []
    for step in range(updates):
        ids = torch.randint(
            len(states),
            (min(profile["bc_batch_size"], len(states)),),
            device=model.device,
        )
        prediction = torch.tanh(model.actor(states[ids]))
        loss = F.mse_loss(prediction, labels[ids])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite BC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.actor.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "updates": updates,
        "final_training_loss": sum(losses[-100:]) / max(1, len(losses[-100:])),
    }


def fit_critic_mc(model, data, updates, batch_size):
    states, targets = data
    with torch.no_grad():
        before = float((model.values_tensor(states) - targets).square().mean().sqrt())
    for _ in range(updates):
        ids = torch.randint(
            len(states), (min(batch_size, len(states)),), device=model.device
        )
        loss = F.huber_loss(model.critic(states[ids]).squeeze(-1), targets[ids])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite MC critic loss")
        model.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.critic.parameters(), 0.5, error_if_nonfinite=True
        )
        model.critic_optimizer.step()
    with torch.no_grad():
        after = float((model.values_tensor(states) - targets).square().mean().sqrt())
    return {
        "samples": len(targets),
        "updates": updates,
        "fit_rmse_before": before,
        "fit_rmse_after": after,
    }


def score(report):
    return (report["success_rate"], -(report["mean_capture_time_success"] or 1.0e6))


def capture_before_update(model):
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": copy.deepcopy(model.config),
        "actor_optimizer": copy.deepcopy(model.actor_optimizer.state_dict()),
        "critic_optimizer": copy.deepcopy(model.critic_optimizer.state_dict()),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if str(model.device).startswith("cuda")
            else None
        ),
    }


def save_anomaly(
    out, model, cfg, profile, progress, buffer, before, error=None, metrics=None
):
    path = out / "first_anomaly.pt"
    if path.exists():
        return
    snapshot = buffer.diagnostic_snapshot()
    snapshot["tensors"] = {
        k: v[: buffer.diagnostic_steps] for k, v in snapshot["tensors"].items()
    }
    torch.save(
        {
            "schema": "pursuit-rescue-anomaly-v1",
            "checkpoint_contract": rescue_contract(cfg, profile),
            "base_config": cfg,
            "rescue_profile": profile,
            "progress": copy.deepcopy(progress),
            "before_update": before,
            "buffer": snapshot,
            "metrics": metrics,
            "diagnostics": copy.deepcopy(getattr(model, "last_update_diagnostics", {})),
            "exception": repr(error) if error else None,
        },
        path,
    )


def train_rescue(cfg, profile, out, resume=None, init_from=None):
    if resume and init_from:
        raise ValueError("Use resume OR init_from, not both")
    if profile.get("require_init_checkpoint", False) and not resume and not init_from:
        raise ValueError("This curriculum profile requires --init-from")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.manual_seed(cfg["seed"])
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    code_fingerprint = rescue_source_fingerprint()
    progress = {
        "physics_steps": 0,
        "decision_transitions": 0,
        "ppo_updates": 0,
        "critic_reanchors": 0,
        "imitation_physics_steps": 0,
        "best_score": [-1.0, -1.0e6],
        "best_origin": None,
        "bc_updates": 0,
        "critic_ready": False,
        "code_fingerprint": code_fingerprint,
    }
    if resume:
        if Path(resume).resolve().parent.parent != out:
            raise ValueError("Resume in the original output directory")
        model, saved_cfg, saved_profile, saved = load_rescue_checkpoint(
            resume,
            device=cfg["device"],
            expected_contract=rescue_contract(cfg, profile),
            optimizers=True,
        )
        comparison_profile = copy.deepcopy(profile)
        comparison_profile["budget_physics_steps"] = saved_profile[
            "budget_physics_steps"
        ]
        if profile["budget_physics_steps"] < saved["progress"]["physics_steps"]:
            raise ValueError("Resume budget is smaller than completed work")
        if not saved["progress"].get("critic_ready", False):
            raise ValueError(
                "Initialization is incomplete; resume from latest.pt after critic warm-up"
            )
        if rescue_fingerprint(saved_cfg, saved_profile) != rescue_fingerprint(
            cfg, comparison_profile
        ):
            raise ValueError("Resume requires identical profile/configuration")
        progress.update(saved["progress"])
        if progress.get("code_fingerprint") != code_fingerprint:
            emit(
                out,
                "resume_source_changed",
                previous=progress.get("code_fingerprint"),
                current=code_fingerprint,
            )
        progress["code_fingerprint"] = code_fingerprint
        progress["cumulative_ppo_physics_steps"] = (
            progress.get("parent_ppo_physics_steps", 0) + progress["physics_steps"]
        )
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None and str(model.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        emit(
            out,
            "resume",
            physics_steps=progress["physics_steps"],
            ppo_updates=progress["ppo_updates"],
            note="Incomplete episodes restart; completed optimizer updates and counters are restored",
        )
    else:
        if (out / "metrics.jsonl").exists():
            raise ValueError(
                "Output already has a run; use --resume or a new directory"
            )
        write_json(out / "base_config.json", cfg)
        write_json(out / "rescue_profile.json", profile)
        write_json(out / "control_contract.json", rescue_contract(cfg, profile))
        model = new_rescue_model(cfg, profile)
        if init_from:
            parent, parent_cfg, parent_profile, parent_saved = load_rescue_checkpoint(
                init_from, device=cfg["device"]
            )
            assert_transfer_compatible(parent_cfg, parent_profile, cfg, profile)
            model.actor.load_state_dict(parent.actor.state_dict())
            progress["parent_checkpoint"] = {
                "path": str(Path(init_from).resolve()),
                "sha256": hashlib.sha256(Path(init_from).read_bytes()).hexdigest(),
                "origin": parent_saved["origin"],
                "progress": {
                    k: v
                    for k, v in parent_saved["progress"].items()
                    if k
                    in (
                        "physics_steps",
                        "ppo_updates",
                        "cumulative_ppo_physics_steps",
                        "critic_ready",
                    )
                },
                "training_contract": parent_saved["contract"],
                "critic_and_optimizers_reset": True,
                "exploration_reset_to_profile": True,
            }
            progress["parent_ppo_physics_steps"] = parent_saved["progress"].get(
                "cumulative_ppo_physics_steps",
                parent_saved["progress"]["physics_steps"],
            )
            emit(
                out,
                "transfer_initialization",
                parent=progress["parent_checkpoint"],
                note="Actor copied under the same control interface; critic and optimizers are fresh",
            )
            del parent
        # Only imitation initialization masks command memory. Transfer without BC
        # retains the actor exactly, including learned memory weights.
        if not init_from or profile["bc_updates"]:
            with torch.no_grad():
                model.actor[0].weight[:, 32:] = 0.0
        emit(
            out,
            "start",
            fingerprint=rescue_fingerprint(cfg, profile),
            action_mode=profile["action_mode"],
            physics_hz=1 / cfg["physics"]["dt"],
            decision_hz=1 / (cfg["physics"]["dt"] * profile["decision_interval_steps"]),
        )
        if profile["bc_updates"]:
            states, labels, collection = collect_imitation_data(
                cfg, profile, profile["bc_samples"], cfg["seed"] + 10000
            )
            progress["imitation_physics_steps"] += collection["physics_steps"]
            fit = fit_imitation(model, states, labels, profile, profile["bc_updates"])
            progress["bc_updates"] += fit["updates"]
            emit(out, "behavior_cloning", collection=collection, fit=fit)
            for round_index in range(profile["dagger_rounds"]):
                extra_x, extra_y, collection = collect_imitation_data(
                    cfg,
                    profile,
                    profile["dagger_samples"],
                    cfg["seed"] + 20000 + round_index,
                    model=model,
                )
                progress["imitation_physics_steps"] += collection["physics_steps"]
                states = torch.cat((states, extra_x))
                labels = torch.cat((labels, extra_y))
                fit = fit_imitation(
                    model, states, labels, profile, profile["dagger_updates"]
                )
                progress["bc_updates"] += fit["updates"]
                emit(
                    out, "dagger", round=round_index + 1, collection=collection, fit=fit
                )
            del states, labels
        if init_from:
            origin = (
                "transfer_bc_initial" if progress["bc_updates"] else "transfer_initial"
            )
        else:
            origin = "bc_initial" if progress["bc_updates"] else "random_initial"
        initial, _ = evaluate_rescue(
            model,
            cfg,
            profile,
            profile["validation_episodes"],
            profile["validation_seed"],
        )
        write_json(out / "initial_validation.json", initial)
        emit(
            out,
            "initial_validation",
            origin=origin,
            success_rate=initial["success_rate"],
            failures=initial["failures"],
        )
        progress["best_score"] = list(score(initial))
        progress["best_origin"] = origin
        progress["cumulative_ppo_physics_steps"] = (
            progress.get("parent_ppo_physics_steps", 0) + progress["physics_steps"]
        )
        save_rescue_checkpoint(
            checkpoint_dir / "bc.pt", model, cfg, profile, progress, origin
        )
        if (
            profile["bc_updates"]
            and initial["success_rate"] < profile["bc_min_success"]
        ):
            raise RuntimeError(
                "Initialization validation is below bc_min_success; initialization saved. Do not spend the PPO budget yet."
            )
        calibration, data = evaluate_rescue(
            model,
            cfg,
            profile,
            profile["critic_mc_episodes"],
            810000 + cfg["seed"],
            stochastic=True,
            collect_mc=True,
        )
        fit = fit_critic_mc(
            model, data, profile["critic_pretrain_updates"], profile["bc_batch_size"]
        )
        del data
        emit(
            out,
            "critic_pretrain",
            initial_value_rmse=calibration["critic_rmse"],
            policy_success_rate=calibration["success_rate"],
            fit=fit,
        )
        progress["critic_ready"] = True
        save_rescue_checkpoint(
            checkpoint_dir / "bc.pt", model, cfg, profile, progress, origin
        )
        save_rescue_checkpoint(
            checkpoint_dir / "best.pt", model, cfg, profile, progress, origin
        )
        save_rescue_checkpoint(
            checkpoint_dir / "latest.pt", model, cfg, profile, progress, origin
        )
    write_json(out / "rescue_profile_latest.json", profile)
    env = RescueBatch(cfg, profile, seed=cfg["seed"] + progress["ppo_updates"] * 7919)
    obs, priv = env.observe()
    episode_start = torch.ones(env.n, dtype=torch.bool, device=env.device)
    next_validation = (
        progress["physics_steps"] // profile["validation_every_physics_steps"] + 1
    ) * profile["validation_every_physics_steps"]
    wall = time.monotonic()
    while progress["physics_steps"] < profile["budget_physics_steps"]:
        remaining = profile["budget_physics_steps"] - progress["physics_steps"]
        steps = min(
            profile["rollout_decisions"], math.ceil(remaining / (env.n * env.interval))
        )
        buffer = TensorRolloutBuffer(
            steps,
            env.n,
            env.actor_dim,
            env.critic_dim,
            env.device,
            action_dim=env.action_dim,
            diagnostics={
                "enabled": True,
                "save_first_anomaly": True,
                "value_abs_threshold": 30.0,
                "target_abs_threshold": 30.0,
                "advantage_abs_threshold": 30.0,
                "critic_loss_threshold": 10.0,
            },
        )
        buffer.physics = copy.deepcopy(cfg["physics"])
        buffer.physics["dt"] = env.decision_dt
        buffer.action_semantics = profile["action_mode"]
        completed = []
        for tick in range(steps):
            action, raw, logprob, value = model.sample_tensor(obs, priv)
            (next_obs, next_priv), reward, term, trunc, records = env.step(action, raw)
            buffer.obs[tick] = obs
            buffer.priv[tick] = priv
            buffer.raw[tick] = raw
            buffer.logprob[tick] = logprob
            buffer.values[tick] = value
            buffer.reward[tick] = reward
            buffer.terminated[tick] = term
            buffer.truncated[tick] = trunc
            buffer.next_values[tick] = model.values_tensor(next_priv)
            buffer.episode_start[tick] = episode_start
            buffer.elapsed[tick] = env.env.elapsed
            buffer.actuator_state[tick, :, 0] = (
                next_obs[:, 24] * cfg["physics"]["hover_thrust"]
            )
            buffer.actuator_state[tick, :, 1:] = next_obs[:, 21:24] * max(
                cfg["physics"]["rate_xy"], cfg["physics"]["rate_yaw"]
            )
            for rec in records:
                buffer.event_code[tick, rec["slot"]] = {
                    "success": 1,
                    "speed": 2,
                    "outside": 3,
                    "tilt": 4,
                    "timeout": 5,
                }[rec["reason"]]
            buffer.diagnostic_steps = tick + 1
            completed.extend(records)
            episode_start = term | trunc
            obs, priv = env.reset(episode_start) if records else (next_obs, next_priv)
        buffer.compute_gae(env.gamma, env.gae_lambda)
        before = (
            capture_before_update(model)
            if not (out / "first_anomaly.pt").exists()
            else None
        )
        if before is not None:
            before["progress"] = copy.deepcopy(progress)
        try:
            stats = model.update(buffer, aux_pd_coef=0.0)
        except Exception as error:
            save_anomaly(
                out, model, cfg, profile, progress, buffer, before, error=error
            )
            raise
        diagnostics = stats.pop("diagnostics", None)
        progress["physics_steps"] += steps * env.n * env.interval
        progress["decision_transitions"] += steps * env.n
        progress["ppo_updates"] += 1
        progress["cumulative_ppo_physics_steps"] = (
            progress.get("parent_ppo_physics_steps", 0) + progress["physics_steps"]
        )
        emit(
            out,
            "ppo_update",
            **progress,
            metrics=stats,
            completed_episodes=len(completed),
            rollout_successes=sum(r["success"] for r in completed),
            active_physics_steps=int(env.active_physics_steps),
            diagnostics=diagnostics,
            seconds_since_ppo_start=time.monotonic() - wall,
        )
        if (
            diagnostics
            and diagnostics.get("anomaly_reasons")
            and not (out / "first_anomaly.pt").exists()
        ):
            save_anomaly(
                out, model, cfg, profile, progress, buffer, before, metrics=stats
            )
        if (
            progress["physics_steps"] >= next_validation
            or progress["physics_steps"] >= profile["budget_physics_steps"]
        ):
            validation, _ = evaluate_rescue(
                model,
                cfg,
                profile,
                profile["validation_episodes"],
                profile["validation_seed"],
            )
            write_json(out / f"validation_{progress['physics_steps']}.json", validation)
            calibration, data = evaluate_rescue(
                model,
                cfg,
                profile,
                profile["critic_mc_episodes"],
                810000 + cfg["seed"] + progress["ppo_updates"],
                stochastic=True,
                collect_mc=True,
            )
            fit = None
            if (
                profile["critic_reanchor_rmse"] > 0
                and calibration["critic_rmse"] > profile["critic_reanchor_rmse"]
            ):
                fit = fit_critic_mc(
                    model,
                    data,
                    profile["critic_reanchor_updates"],
                    profile["bc_batch_size"],
                )
                progress["critic_reanchors"] += 1
            del data
            emit(
                out,
                "validation",
                physics_steps=progress["physics_steps"],
                ppo_updates=progress["ppo_updates"],
                success_rate=validation["success_rate"],
                failures=validation["failures"],
                critic_rmse_before_refit=calibration["critic_rmse"],
                critic_policy_success_rate=calibration["success_rate"],
                critic_refit=fit,
            )
            if score(validation) > tuple(progress["best_score"]):
                progress["best_score"] = list(score(validation))
                progress["best_origin"] = "ppo"
                save_rescue_checkpoint(
                    checkpoint_dir / "best.pt", model, cfg, profile, progress, "ppo"
                )
            next_validation += profile["validation_every_physics_steps"]
        save_rescue_checkpoint(
            checkpoint_dir / "latest.pt", model, cfg, profile, progress, "ppo"
        )
    result = {
        "schema": "pursuit-rescue-training-summary-v1",
        **progress,
        "note": "Best may remain the initial policy. Compare bc.pt (the initialization) against PPO on held-out episodes; parent training is recorded separately.",
    }
    write_json(out / "training_summary.json", result)
    return result
