"""Production-oriented training loop, evaluation, checkpointing and curriculum."""

from pathlib import Path
import json
import copy
import random
import time
import platform
import os
import math
import numpy as np
import torch
from pursuit_rl.config import config_hash, dump_config, validate, as_dict
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.observations import observation_dimensions
from pursuit_rl.learning.ppo import AsymmetricPPO
from pursuit_rl.learning.buffers import (
    RolloutBuffer,
    TensorRolloutBuffer,
    diagnostic_options,
    distribution_summary,
)
from pursuit_rl.evaluation.rollout import evaluate
from pursuit_rl.artifacts.provenance import source_fingerprint
from pursuit_rl import __version__

from pursuit_rl.learning.checkpoints import (
    SCHEMA,
    save_checkpoint,
    initialize_weights,
    resume_training,
)
from pursuit_rl.artifacts.io import write_json
from pursuit_rl.artifacts.logging import ExpLogger


def promotion_state(stage, stage_steps, streak, success_rate, training):
    if stage >= training["max_stage"] or stage_steps < training["min_stage_steps"]:
        return streak, False
    streak = streak + 1 if success_rate >= training["promotion_success"] else 0
    return streak, streak >= training["promotion_consecutive"]


class Trainer:
    def __init__(
        self, cfg, out, resume=None, init_from=None, invocation=None, diagnostics=None
    ):
        if resume and init_from:
            raise ValueError("resume and init_from are mutually exclusive")
        cfg = copy.deepcopy(as_dict(cfg))
        validate(cfg)
        self.out = Path(out)
        if self.out.exists() and any(self.out.iterdir()) and not resume:
            raise FileExistsError(
                "New training requires an empty output directory; use --resume to continue"
            )
        if resume and (self.out / "run.json").exists():
            old_run = json.loads((self.out / "run.json").read_text(encoding="utf8"))
            if old_run["config_hash"] != config_hash(cfg):
                raise ValueError("Resume output belongs to a different configuration")
            if Path(resume).resolve().parent != (self.out / "checkpoints").resolve():
                raise ValueError(
                    "Resume into an existing run must use its own checkpoint"
                )
        elif resume and self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError("Resume destination must be empty or a matching run")
        self.exp_logger = ExpLogger(
            self.out,
            "train",
            cfg,
            invocation=invocation
            or {
                "api": "Trainer",
                "out": str(out),
                "resume": str(resume) if resume else None,
                "init_from": str(init_from) if init_from else None,
            },
            tensorboard=True,
        )
        try:
            # A sidecar keeps diagnostic controls out of the experiment config
            # hash, so existing checkpoints/configs retain strict compatibility.
            diagnostic_path = os.environ.get("PURSUIT_DIAGNOSTICS_CONFIG")
            if diagnostics is None and diagnostic_path:
                diagnostics = json.loads(
                    Path(diagnostic_path).expanduser().read_text(encoding="utf8")
                )
                self.exp_logger.record_inputs({"diagnostics": diagnostic_path})
            self.diagnostics = diagnostic_options(diagnostics)
            self.exp_logger.metadata(ppo_diagnostics=self.diagnostics)
            inputs = {
                key: value
                for key, value in {
                    "resume": resume,
                    "init_from": init_from,
                    "config": (invocation or {}).get("config")
                    or (
                        (invocation or {}).get("effective_config")
                        if (invocation or {}).get("effective_config")
                        != "packaged defaults"
                        else None
                    ),
                }.items()
                if value
            }
            self.exp_logger.record_inputs(inputs)
            self._initialize(cfg, out, resume, init_from)
        except BaseException as error:
            self.exp_logger.fail(error)
            self.exp_logger.close()
            raise

    def _initialize(self, cfg, out, resume, init_from):
        self.config = cfg
        # PPO storage/GAE assumes one new action is sampled per environment step.
        # Holding commands is supported in evaluation only.
        if cfg["physics"].get("policy_interval_steps", 1) != 1:
            raise ValueError(
                "Training requires physics.policy_interval_steps=1; frequency hold is evaluation-only"
            )
        self.training_config = cfg["training"]
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt = self.out / "checkpoints"
        self.ckpt.mkdir(exist_ok=True)
        torch.set_num_threads(cfg["torch_threads"])
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        self.actor_dim, self.critic_dim = observation_dimensions(cfg)
        self.model = AsymmetricPPO(
            self.actor_dim,
            self.critic_dim,
            self.training_config,
            device=cfg["device"],
        )
        self.exp_logger.metadata(
            simulation_backend=cfg["simulation_backend"],
            device=str(self.model.device),
            cuda_runtime=torch.version.cuda,
            gpu_name=(
                torch.cuda.get_device_name(self.model.device)
                if self.model.device.type == "cuda"
                else None
            ),
        )
        self.tensor_backend = cfg.get("simulation_backend", "numpy") == "torch"
        self.teacher = None
        if self.training_config["aux_pd_coef"] > 0:
            from pursuit_rl.simulation.pd_teacher import pd_teacher

            self.teacher = pd_teacher
        if self.tensor_backend:
            from pursuit_rl.simulation.torch_backend import TorchPursuitBatch

            self.env = TorchPursuitBatch(
                cfg,
                seed=cfg["seed"],
                stage=self.training_config["initial_stage"],
                device=self.model.device,
            )
        else:
            self.env = PursuitBatch(
                cfg, seed=cfg["seed"], stage=self.training_config["initial_stage"]
            )
        self.obs, self.priv = self.env.observe()
        self.episode_count = 0
        self.steps = 0
        self.stage = self.training_config["initial_stage"]
        self.stage_since = 0
        self.promotion_streak = 0
        self.best_score = (-1.0, float("-inf"))
        self.best_stage_score = {}
        self.parent_checkpoint = None
        self.source_sha256 = source_fingerprint()
        self.first_anomaly_saved = any(
            (self.out / "diagnostics").glob("first_anomaly_*.pt")
        )
        if resume:
            resume_training(self, resume)
        elif init_from:
            initialize_weights(self, init_from)
            with torch.no_grad():
                self.model.log_std.clamp_(
                    min=self.training_config.get("log_std_min", -4.0)
                )
        dump_config(cfg, self.out / "config.yaml")
        metadata = {
            "schema": SCHEMA,
            "run_id": self.exp_logger.session_id,
            "created_at": self.exp_logger.state["created_at"],
            "config_hash": config_hash(cfg),
            "seed": cfg["seed"],
            "init_from": init_from,
            "package_version": __version__,
            "source_sha256": self.source_sha256,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": cfg["device"],
            "parent_checkpoint": self.parent_checkpoint,
        }
        if not (self.out / "run.json").exists():
            write_json(self.out / "run.json", metadata)
        if resume:
            self.exp_logger.event("resume", checkpoint=str(resume), steps=self.steps)

    def auxiliary_coefficient(self):
        """Fixed teacher weight by default; optional time-based decay for ablation."""
        tc = self.training_config
        coef = float(tc["aux_pd_coef"])
        if tc["aux_pd_schedule"] == "linear":
            coef *= max(0.0, 1.0 - self.steps / tc["aux_pd_decay_steps"])
        return coef

    def collect(self):
        if self.tensor_backend:
            return self.collect_tensor()
        n = self.env.n
        t = self.training_config["n_steps"]
        buffer = RolloutBuffer(
            t, n, self.actor_dim, self.critic_dim, diagnostics=self.diagnostics
        )
        records = []
        reward_totals = {}
        saturation = np.zeros(4)
        count = 0
        for k in range(t):
            episode_start = self.env.steps == 0
            if self.teacher is not None:
                buffer.teacher[k] = self.teacher(self.env)
            actions, raw, logprob, vals = self.model.sample(self.obs, self.priv)
            (next_obs, next_priv), reward, term, trunc, info = self.env.step(actions)
            for name, values in self.env.reward_components.items():
                reward_totals[name] = reward_totals.get(name, 0.0) + float(
                    np.mean(values)
                )
            next_values = self.model.values(
                next_priv
            )  # before reset, timeout bootstrap
            buffer.obs[k] = self.obs
            buffer.priv[k] = self.priv
            buffer.raw[k] = raw
            buffer.logprob[k] = logprob
            buffer.values[k] = vals
            buffer.reward[k] = reward
            buffer.next_values[k] = next_values
            buffer.terminated[k] = term
            buffer.truncated[k] = trunc
            buffer.record_diagnostics(k, self.env, episode_start, info)
            saturation += (np.abs(actions) >= 0.98).sum(axis=0)
            count += n
            for rec in info:
                if rec:
                    records.append(rec)
                    self.episode_count += 1
                    self.exp_logger.record(
                        "episodes.jsonl",
                        {
                            "training_episode": self.episode_count,
                            "episode_id": f"{self.exp_logger.session_id}:{self.episode_count}",
                            "phase": "collected",
                            "steps": self.steps + (k + 1) * n,
                            **rec,
                        },
                    )
            done = term | trunc
            if done.any():
                self.env.reset(done)
                next_obs, next_priv = self.env.observe()
            self.obs, self.priv = next_obs, next_priv
        self.rollout_reward_means = {
            name: value / t for name, value in reward_totals.items()
        }
        buffer.compute_gae(
            self.training_config["gamma"], self.training_config["gae_lambda"]
        )
        return buffer, records, (saturation / count).tolist()

    @torch.no_grad()
    def collect_tensor(self):
        """Keep observations, policy samples, rewards, GAE and actions on GPU.

        Transfer only per-rollout summaries for JSONL logging. Reconstructing
        full episode records is intentionally restricted to CPU evaluation.
        """
        n = self.env.n
        t = self.training_config["n_steps"]
        buf = TensorRolloutBuffer(
            t,
            n,
            self.actor_dim,
            self.critic_dim,
            self.model.device,
            diagnostics=self.diagnostics,
        )
        comp_totals = {}
        saturation = torch.zeros(4, device=self.model.device)
        events = torch.empty((t, n, 6), device=self.model.device, dtype=torch.float32)
        for k in range(t):
            episode_start = self.env.steps == 0
            if self.teacher is not None:
                buf.teacher[k] = self.teacher(self.env)
            action, raw, lp, values = self.model.sample_tensor(self.obs, self.priv)
            (next_obs, next_priv), reward, term, trunc, _ = self.env.step(action)
            for name, val in self.env.reward_components.items():
                comp_totals[name] = (
                    comp_totals.get(name, torch.zeros((), device=self.model.device))
                    + val.mean()
                )
            next_values = self.model.values_tensor(next_priv)
            buf.obs[k] = self.obs
            buf.priv[k] = self.priv
            buf.raw[k] = raw
            buf.logprob[k] = lp
            buf.values[k] = values
            buf.reward[k] = reward
            buf.next_values[k] = next_values
            buf.terminated[k] = term
            buf.truncated[k] = trunc
            buf.record_diagnostics(k, self.env, episode_start)
            saturation += (action.abs() >= 0.98).sum(0)
            done = term | trunc
            ev = self.env.last_event
            events[k] = torch.stack(
                (
                    done.to(torch.float32),
                    ev["hit"].to(torch.float32),
                    (ev["speed"] > self.config["physics"]["max_speed"]).to(
                        torch.float32
                    ),
                    ev["outside"].to(torch.float32),
                    ev["failed"].to(torch.float32),
                    self.env.elapsed,
                ),
                dim=1,
            )
            # Dense masked reset is fully on-device; never sync each tick.
            # A single bool transfer is cheaper than 40+ masked reset kernels
            # on every 1 ms tick when no episode has completed.
            if bool(done.any().item()):
                self.env.reset(done)
                next_obs, next_priv = self.env.observe()
            self.obs, self.priv = next_obs, next_priv
        buf.compute_gae(
            self.training_config["gamma"], self.training_config["gae_lambda"]
        )
        summary = torch.stack([comp_totals[name] for name in sorted(comp_totals)])
        values = summary.cpu().tolist()
        self.rollout_reward_means = {
            name: float(val) / t for name, val in zip(sorted(comp_totals), values)
        }
        sat = (saturation / (n * t)).cpu().tolist()
        # Aggregate completion events in one transfer per rollout, not 1 kHz.
        host = events.cpu().numpy()
        coords = np.argwhere(host[:, :, 0] > 0)
        records = []
        for k, i in coords:
            e = host[k, i]
            reason = (
                "success"
                if e[1] > 0
                else (
                    "speed"
                    if e[2] > 0
                    else "outside" if e[3] > 0 else "tilt" if e[4] > 0 else "timeout"
                )
            )
            self.episode_count += 1
            record = {
                "success": bool(e[1] > 0),
                "reason": reason,
                "stage": self.stage,
                "time": float(e[5]),
            }
            records.append(record)
            self.exp_logger.record(
                "episodes.jsonl",
                {
                    "training_episode": self.episode_count,
                    "episode_id": f"{self.exp_logger.session_id}:{self.episode_count}",
                    "phase": "collected",
                    "steps": self.steps + (int(k) + 1) * n,
                    **record,
                },
            )
        return buf, records, sat

    @staticmethod
    def _diagnostic_host(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, dict):
            return {key: Trainer._diagnostic_host(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(Trainer._diagnostic_host(item) for item in value)
        return copy.deepcopy(value)

    def _capture_update_state(self):
        if (
            not self.diagnostics["enabled"]
            or not self.diagnostics["save_first_anomaly"]
            or self.first_anomaly_saved
        ):
            return None
        return self._diagnostic_host(
            {
                "actor_critic": self.model.state_dict(),
                "actor_optimizer": self.model.actor_optimizer.state_dict(),
                "critic_optimizer": self.model.critic_optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if self.model.device.type == "cuda"
                    else None
                ),
                "python_rng": random.getstate(),
            }
        )

    def _save_first_anomaly(self, buffer, before, reasons, metrics=None, error=None):
        if before is None or self.first_anomaly_saved:
            return None
        directory = self.out / "diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"first_anomaly_{self.exp_logger.session_id}_{self.steps:012d}"
        path = directory / (stem + ".pt")
        summary = {
            "schema": "ppo-first-anomaly-v1",
            "reasons": reasons,
            "steps_before_update": self.steps,
            "stage": self.stage,
            "aux_pd_coef": self.auxiliary_coefficient(),
            "error": str(error) if error else None,
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device": str(self.model.device),
            },
            "diagnostic_options": self.diagnostics,
            "source_sha256": self.source_sha256,
            "metrics": metrics or getattr(self.model, "last_update_diagnostics", {}),
        }
        payload = {
            **summary,
            "config": self.config,
            "pre_update": before,
            "post_update": self._diagnostic_host(self.model.state_dict()),
            "rollout": buffer.diagnostic_snapshot(
                include_teacher=self.teacher is not None
            ),
            "environment_after_collection": self._diagnostic_host(self.env.snapshot()),
            "purpose": "Replay the frozen rollout optimizer update; not a resumable training checkpoint",
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self.first_anomaly_saved = True
        write_json(directory / (stem + ".json"), summary)
        self.exp_logger.event(
            "first_numerical_anomaly",
            path=str(path.resolve()),
            reasons=reasons,
            steps_before_update=self.steps,
        )
        self.exp_logger.logger.warning(
            "PPO numerical anomaly %s; saved %s", reasons, path
        )
        return path

    @torch.no_grad()
    def calibrate_critic(self, episodes=None):
        """Complete stochastic episodes under a frozen policy, without training RNG changes.

        This is opt-in because complete 1 kHz episodes can be expensive. Episode
        cutoffs are scored as the finite evaluation task, with zero tail value.
        """
        wanted = (
            self.diagnostics["calibration_episodes"] if episodes is None else episodes
        )
        if type(wanted) is not int or wanted < 1:
            raise ValueError("Critic calibration requires a positive episode count")
        seed = self.diagnostics["calibration_seed"]
        dt, gamma = self.config["physics"]["dt"], self.training_config["gamma"]
        predictions, returns, successes, timeouts = [], [], 0, 0
        started = time.monotonic()
        devices = (
            list(range(torch.cuda.device_count()))
            if self.model.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for start in range(0, wanted, self.env.n):
                count = min(self.env.n, wanted - start)
                if self.tensor_backend:
                    from pursuit_rl.simulation.torch_backend import TorchPursuitBatch

                    env = TorchPursuitBatch(
                        self.config,
                        n=count,
                        seed=seed + start,
                        stage=self.stage,
                        device=self.model.device,
                    )
                    finished = torch.zeros(
                        count, dtype=torch.bool, device=self.model.device
                    )
                    total = torch.zeros(
                        count, dtype=torch.float64, device=self.model.device
                    )
                    batch_success = torch.zeros(
                        (), dtype=torch.long, device=self.model.device
                    )
                    batch_timeout = torch.zeros(
                        (), dtype=torch.long, device=self.model.device
                    )
                else:
                    env = PursuitBatch(
                        self.config, n=count, seed=seed + start, stage=self.stage
                    )
                    finished = np.zeros(count, bool)
                    total = np.zeros(count, np.float64)
                obs, priv = env.observe()
                predicted = (
                    self.model.values_tensor(priv).cpu().numpy()
                    if self.tensor_backend
                    else self.model.values(priv)
                )
                discount = 1.0
                for tick in range(int(math.ceil(env.stage_config["time"] / dt)) + 2):
                    sample = (
                        self.model.sample_tensor(obs, priv)
                        if self.tensor_backend
                        else self.model.sample(obs, priv)
                    )
                    (obs, priv), reward, term, trunc, info = env.step(sample[0])
                    active = ~finished
                    if self.tensor_backend:
                        total += discount * torch.where(active, reward, 0.0)
                        new = (term | trunc) & active
                        batch_success += (new & env.last_event["hit"]).sum()
                        batch_timeout += (new & trunc & ~term).sum()
                    else:
                        total += discount * np.where(active, reward, 0.0)
                        new = (term | trunc) & active
                        successes += sum(
                            bool(new[i]) and rec.get("success", False)
                            for i, rec in enumerate(info)
                        )
                        timeouts += int((new & trunc & ~term).sum())
                    finished = finished | term | trunc
                    discount *= gamma
                    if tick % 64 == 0 and bool(finished.all()):
                        break
                if not bool(finished.all()):
                    raise RuntimeError("Critic calibration did not finish all episodes")
                predictions.extend(predicted.tolist())
                returns.extend(
                    total.cpu().tolist() if self.tensor_backend else total.tolist()
                )
                if self.tensor_backend:
                    successes += int(batch_success.item())
                    timeouts += int(batch_timeout.item())
        prediction = np.asarray(predictions, np.float64)
        actual = np.asarray(returns, np.float64)
        error = prediction - actual
        report = {
            "schema": "ppo-critic-calibration-v1",
            "steps": self.steps,
            "stage": self.stage,
            "episodes": wanted,
            "seed": seed,
            "policy": "frozen_stochastic",
            "tail_at_task_cutoff": 0.0,
            "gamma": gamma,
            "dt": dt,
            "successes": successes,
            "timeouts": timeouts,
            "predicted_values": distribution_summary(prediction),
            "monte_carlo_returns": distribution_summary(actual),
            "prediction_errors": distribution_summary(error),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "elapsed_s": time.monotonic() - started,
        }
        self.exp_logger.record("critic_calibration.jsonl", report)
        return report

    def run(self):
        with self.exp_logger:
            result = self._run()
            self.exp_logger.complete(result)
            return result

    def _run(self):
        next_eval = (
            (self.steps // self.training_config["eval_every"]) + 1
        ) * self.training_config["eval_every"]
        wall = time.monotonic()
        while self.steps < self.training_config["total_steps"]:
            buf, episodes, sat = self.collect()
            coef = self.auxiliary_coefficient()
            before = self._capture_update_state()
            try:
                metrics = (
                    self.model.update(buf, aux_pd_coef=coef)
                    if coef > 0
                    else self.model.update(buf)
                )
            except BaseException as error:
                try:
                    self._save_first_anomaly(
                        buf, before, ["update_exception"], error=error
                    )
                except Exception:
                    self.exp_logger.logger.exception(
                        "Could not save first numerical anomaly"
                    )
                raise
            numerical = metrics.get("diagnostics")
            if numerical is not None:
                self.exp_logger.record(
                    "diagnostics.jsonl",
                    {
                        "steps_before_update": self.steps,
                        "stage": self.stage,
                        **numerical,
                    },
                )
                reasons = numerical["anomaly_reasons"]
                if reasons:
                    self._save_first_anomaly(buf, before, reasons, metrics=metrics)
            del before
            gained = self.env.n * self.training_config["n_steps"]
            previous_steps = self.steps
            self.steps += gained
            self.stage_since += gained
            for mark in self.training_config.get("snapshot_steps", []):
                if previous_steps < mark <= self.steps:
                    save_checkpoint(self, f"snapshot_{mark:08d}")
            metric = {
                "steps": self.steps,
                "simulated_seconds_per_env": self.steps
                * self.config["physics"]["dt"]
                / self.env.n,
                "policy_update_hz": 1.0 / self.config["physics"]["dt"],
                "stage": self.stage,
                "train_episodes": len(episodes),
                "train_failure_reasons": {
                    reason: sum(1 for e in episodes if e.get("reason") == reason)
                    for reason in ("success", "speed", "tilt", "outside", "timeout")
                },
                "simulation_backend": self.config["simulation_backend"],
                "environment_steps_per_second": round(
                    self.steps / max(time.monotonic() - wall, 1e-6), 1
                ),
                "train_success": (
                    float(np.mean([e["success"] for e in episodes]))
                    if episodes
                    else None
                ),
                "saturation": sat,
                "elapsed_s": round(time.monotonic() - wall, 1),
                "reward_components": self.rollout_reward_means,
                "training_episodes_total": self.episode_count,
                **metrics,
            }
            self.exp_logger.record("train.jsonl", metric, tensorboard="train")
            cadence = self.diagnostics["calibration_every_updates"]
            rollout_index = self.steps // gained
            if self.diagnostics["enabled"] and cadence and rollout_index % cadence == 0:
                self.calibrate_critic()
            if (
                self.steps >= next_eval
                or self.steps >= self.training_config["total_steps"]
            ):
                # Persist the just-trained optimizer/policy BEFORE validation.
                # A GPU evaluator exception must not discard millions of steps
                # when it happens at the first scheduled evaluation.
                save_checkpoint(self, "latest")
                # Evaluations are separate from the training environment.
                local = evaluate(
                    self.model,
                    self.config,
                    stage=self.stage,
                    episodes=self.training_config["eval_episodes"],
                    seed=self.training_config["eval_seed"] + self.stage * 100000,
                )
                if self.training_config["challenge_eval_enabled"]:
                    global_test = evaluate(
                        self.model,
                        self.config,
                        stage=self.training_config["max_stage"],
                        episodes=self.training_config["eval_episodes"],
                        seed=self.training_config["eval_seed"] + 900000,
                    )
                else:
                    # Basic Stage0 optimization: one evaluation, not an independent
                    # 'challenge' estimate. Mark this explicitly in saved logs.
                    global_test = local
                rank = (global_test["success_rate"], -global_test["mean_capped_time"])
                local_rank = (local["success_rate"], -local["mean_capped_time"])
                if (
                    self.stage == self.training_config["max_stage"]
                    and rank > self.best_score
                ):
                    self.best_score = rank
                    save_checkpoint(self, "best")
                if local_rank > self.best_stage_score.get(
                    self.stage, (-1.0, float("-inf"))
                ):
                    self.best_stage_score[self.stage] = local_rank
                    save_checkpoint(self, f"best_stage{self.stage}")
                result = {
                    "steps": self.steps,
                    "stage": self.stage,
                    "stage_success": local["success_rate"],
                    "stage_capped_time": local["mean_capped_time"],
                    "challenge_success": global_test["success_rate"],
                    "challenge_independent": self.training_config[
                        "challenge_eval_enabled"
                    ],
                    "challenge_capped_time": global_test["mean_capped_time"],
                }
                self.exp_logger.record(
                    "validation.jsonl", result, tensorboard="validation"
                )
                self.exp_logger.record(
                    "validation_details.jsonl",
                    {
                        "steps": self.steps,
                        "stage": self.stage,
                        "stage_report": local,
                        "challenge_report": (
                            global_test
                            if self.training_config["challenge_eval_enabled"]
                            else None
                        ),
                        "challenge_independent": self.training_config[
                            "challenge_eval_enabled"
                        ],
                    },
                )
                save_checkpoint(self, "latest")
                streak, promote = promotion_state(
                    self.stage,
                    self.stage_since,
                    self.promotion_streak,
                    local["success_rate"],
                    self.training_config,
                )
                self.promotion_streak = streak
                if promote:
                    self.stage += 1
                    self.stage_since = 0
                    self.promotion_streak = 0
                    self.env.set_stage(self.stage)
                    self.obs, self.priv = self.env.observe()
                    self.exp_logger.event(
                        "promotion", steps=self.steps, stage=self.stage
                    )
                save_checkpoint(self, "latest")
                while next_eval <= self.steps:
                    next_eval += self.training_config["eval_every"]
                self.exp_logger.logger.info(
                    f"steps={self.steps:,} stage={self.stage} challenge={global_test['success_rate']:.1%} time={global_test['mean_capped_time']:.1f}s KL={metrics['approx_kl']:.4f}",
                )
        save_checkpoint(self, "latest")
        return {"steps": self.steps, "stage": self.stage, "best_score": self.best_score}
