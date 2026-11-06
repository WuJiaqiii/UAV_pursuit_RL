"""Asymmetric PPO policy and optimization, independent of rollout storage."""

import math
import torch
from pursuit_rl.simulation.torch_device import resolve_torch_device
from torch import nn
import torch.nn.functional as F
from .valuenorm import ValueNorm
from .prediction import TrajectoryPredictor


def mlp(input_dim, hidden, output_dim):
    dims = [input_dim, *hidden, output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class AsymmetricPPO(nn.Module):
    def __init__(
        self, obs_dim, critic_dim, config, device="cpu", create_optimizers=True
    ):
        super().__init__()
        self.config = config
        self.action_dim = int(config.get("action_dim", 4))
        self.rho = float(config.get("exploration_rho", 0.0))
        self.memory_start = int(config.get("exploration_memory_start", 0))
        self.obs_dim = obs_dim
        self.base_obs_dim = config.get("base_obs_dim", obs_dim)
        self.position_error_cap_m = float(config.get("position_error_cap_m", 0.0))
        self.prediction_config = config.get("prediction", {"enabled": False})
        self.predictor = (
            TrajectoryPredictor(self.prediction_config)
            if self.prediction_config["enabled"]
            else None
        )
        feature_dim = self.base_obs_dim + (
            3 * len(self.prediction_config["horizons_s"])
            if self.predictor is not None
            else 0
        )
        if not (0 <= self.rho < 1):
            raise ValueError("exploration_rho must be in [0, 1)")
        if self.rho and self.memory_start + self.action_dim > obs_dim:
            raise ValueError("Exploration memory is missing from observations")
        self.actor = mlp(feature_dim, config["hidden_actor"], self.action_dim)
        self.critic = mlp(
            feature_dim + critic_dim - obs_dim, config["hidden_critic"], 1
        )
        self.value_norm = ValueNorm(
            config.get("value_norm", False),
            config.get("norm_beta", 0.995),
            config.get("norm_min_variance", 0.01),
        )
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), config["log_std_init"])
        )
        self.device = resolve_torch_device(device)
        self.to(self.device)
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.predictor_optimizer = None
        if create_optimizers:
            self.actor_optimizer = torch.optim.Adam(
                list(self.actor.parameters()) + [self.log_std], lr=config["actor_lr"]
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=config["critic_lr"]
            )
            if self.predictor is not None and self.prediction_config["mode"] == "lstm":
                self.predictor_optimizer = torch.optim.Adam(
                    self.predictor.parameters(), lr=self.prediction_config["lr"]
                )

    @torch.no_grad()
    def actor_features(self, obs):
        if self.predictor is None:
            return obs
        history = obs[..., self.base_obs_dim :].reshape(
            -1, self.prediction_config["history_steps"], 10
        )
        prediction = self.predictor(history).flatten(1)
        return torch.cat((obs[..., : self.base_obs_dim], prediction), -1)

    @torch.no_grad()
    def critic_features(self, critic_obs):
        if self.predictor is None:
            return critic_obs
        return torch.cat(
            (
                self.actor_features(critic_obs[..., : self.obs_dim]),
                critic_obs[..., self.obs_dim :],
            ),
            -1,
        )

    def critic_values(self, critic_obs):
        normalized = self.critic(self.critic_features(critic_obs)).squeeze(-1)
        return self.value_norm.denormalize(normalized)

    def bound_actor_input(self, features):
        """Bound only the Actor's position vector, preserving real critic state."""
        if self.position_error_cap_m <= 0:
            return features
        features = features.clone()
        position = features[..., :3]
        length = torch.linalg.vector_norm(position, dim=-1, keepdim=True)
        features[..., :3] = position * (
            self.position_error_cap_m / 30.0 / length.clamp_min(1e-9)
        ).clamp(max=1)
        return features

    @torch.no_grad()
    def update_value_statistics(self, returns):
        """Rebase the output head when units change, preserving raw V predictions.

        ValueNorm alone does not preserve outputs. This explicit PopArt-style
        affine correction also keeps old-value clipping in consistent units.
        Clear only the rebased head's Adam state; hidden-layer state is retained.
        """
        old_mean, old_std = self.value_norm.statistics()
        self.value_norm.update(returns)
        mean, std = self.value_norm.statistics()
        if not self.value_norm.enabled:
            return
        head = self.critic[-1]
        ratio = (old_std / std).to(head.weight)
        head.weight.mul_(ratio)
        head.bias.mul_(ratio).add_(((old_mean - mean) / std).to(head.bias))
        if self.critic_optimizer is not None:
            self.critic_optimizer.state.pop(head.weight, None)
            self.critic_optimizer.state.pop(head.bias, None)

    @staticmethod
    def logprob(mean, std, raw):
        normal = torch.distributions.Normal(mean, std)
        # Stable log derivative of tanh; crucial near action saturation.
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return (normal.log_prob(raw) - correction).sum(dim=-1)

    def distribution_parameters(self, obs, prepared=False):
        obs = obs if prepared else self.actor_features(obs)
        obs = self.bound_actor_input(obs)
        target_mean = self.actor(obs)
        if self.rho:
            previous = obs[..., self.memory_start : self.memory_start + self.action_dim]
            mean = self.rho * previous + (1 - self.rho) * target_mean
            std = self.log_std.exp().expand_as(mean) * math.sqrt(
                1 - self.rho * self.rho
            )
        else:
            mean = target_mean
            std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, obs, critic_obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            c = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
            mean, std = self.distribution_parameters(o)
            raw = mean if deterministic else mean + std * torch.randn_like(mean)
            actions = torch.tanh(raw)
            logprob = self.logprob(mean, std, raw)
            values = self.critic_values(c)
        return (
            actions.cpu().numpy(),
            raw.cpu().numpy(),
            logprob.cpu().numpy(),
            values.cpu().numpy(),
        )

    def sample_tensor(self, obs, critic_obs, deterministic=False):
        """Zero-copy device-resident policy sample for tensor environments."""
        with torch.no_grad():
            # Compare against the real weight device, not a possibly unindexed
            # torch.device("cuda") descriptor (which != torch.device("cuda:0")).
            policy_device = next(self.actor.parameters()).device
            critic_device = next(self.critic.parameters()).device
            if not isinstance(obs, torch.Tensor) or obs.device != policy_device:
                raise ValueError(
                    "sample_tensor requires observations on policy device: "
                    f"obs={getattr(obs, 'device', type(obs).__name__)}, "
                    f"policy={policy_device}"
                )
            if (
                not isinstance(critic_obs, torch.Tensor)
                or critic_obs.device != critic_device
            ):
                raise ValueError(
                    "sample_tensor requires critic observations on policy device: "
                    f"critic_obs={getattr(critic_obs, 'device', type(critic_obs).__name__)}, "
                    f"critic={critic_device}"
                )
            mean, std = self.distribution_parameters(obs)
            raw = mean if deterministic else mean + std * torch.randn_like(mean)
            action = torch.tanh(raw)
            lp = self.logprob(mean, std, raw)
            values = self.critic_values(critic_obs)
            return action, raw, lp, values

    def predict_tensor(self, obs):
        with torch.no_grad():
            return torch.tanh(self.distribution_parameters(obs)[0])

    def values_tensor(self, critic_obs):
        with torch.no_grad():
            return self.critic_values(critic_obs)

    def predict(self, obs):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return torch.tanh(self.distribution_parameters(o)[0]).cpu().numpy()

    def values(self, crit):
        with torch.no_grad():
            return (
                self.critic_values(
                    torch.as_tensor(crit, dtype=torch.float32, device=self.device)
                )
                .cpu()
                .numpy()
            )

    def act(self, env, obs):
        return self.predict(obs)

    def update(self, buffer, aux_pd_coef=0.0):
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise RuntimeError(
                "Inference-only model cannot update; initialize a Trainer"
            )
        c = self.config
        options = getattr(buffer, "diagnostic_options", {})
        diagnose = options.get("enabled", False)
        guard = diagnose and options.get("fail_on_nonfinite", True)
        self.last_update_diagnostics = {"phase": "before_update"}
        rollout_diagnostics = buffer.diagnostic_summary() if diagnose else None
        if diagnose:
            self.last_update_diagnostics["rollout"] = rollout_diagnostics
            bad = [
                name
                for name in (
                    "obs",
                    "priv",
                    "raw",
                    "logprob",
                    "reward",
                    "values",
                    "next_values",
                    "bootstrap_values",
                    "delta",
                    "advantage",
                    "returns",
                )
                if rollout_diagnostics[name]["nonfinite_count"]
                or rollout_diagnostics[name]["nonfinite_statistics"]
            ]
            if bad and guard:
                raise FloatingPointError(
                    "Nonfinite rollout fields before PPO update: " + ", ".join(bad)
                )
        # Enforce the configured lower bound even if KL stopping skips updates.
        with torch.no_grad():
            self.log_std.clamp_(
                min=c.get("log_std_min", -4.0), max=c.get("log_std_max", 1.0)
            )
        device = self.device
        o = torch.as_tensor(buffer.obs.reshape(-1, buffer.obs.shape[-1]), device=device)
        pr = torch.as_tensor(
            buffer.priv.reshape(-1, buffer.priv.shape[-1]), device=device
        )
        raw = torch.as_tensor(buffer.raw.reshape(-1, self.action_dim), device=device)
        teacher = (
            torch.as_tensor(buffer.teacher.reshape(-1, self.action_dim), device=device)
            if aux_pd_coef > 0
            else None
        )
        old_lp = torch.as_tensor(buffer.logprob.reshape(-1), device=device)
        adv = torch.as_tensor(buffer.advantage.reshape(-1), device=device)
        ret = torch.as_tensor(buffer.returns.reshape(-1), device=device)
        # GAE and stored values are always in reward units. Freeze target
        # statistics for every minibatch/epoch of this PPO update.
        self.update_value_statistics(ret)
        normalized_ret = self.value_norm.normalize(ret)
        old_values = self.value_norm.normalize(
            torch.as_tensor(buffer.values.reshape(-1), device=device)
        )
        if self.predictor is not None:
            o = torch.cat(
                [self.actor_features(part) for part in o.split(c["batch_size"])]
            )
            pr = torch.cat((o, pr[:, self.obs_dim :]), 1)
        logprob_error = 0.0
        if diagnose:
            with torch.no_grad():
                errors = []
                residuals = []
                for start in range(0, len(old_lp), c["batch_size"]):
                    end = start + c["batch_size"]
                    mean, std = self.distribution_parameters(
                        o[start:end], prepared=True
                    )
                    lp = self.logprob(mean, std, raw[start:end])
                    errors.append((lp - old_lp[start:end]).abs().max())
                    residuals.append(raw[start:end] - mean)
                logprob_error = float(torch.stack(errors).max().item())
                residual = torch.cat(residuals).reshape(buffer.raw.shape)
                if guard and not math.isfinite(logprob_error):
                    raise FloatingPointError(
                        "Nonfinite pre-update action log probabilities"
                    )
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        length = len(adv)
        stats = {
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_proxy": 0.0,
            "updates": 0,
            "aux_pd_loss": 0.0,
            "aux_pd_coef": float(aux_pd_coef),
            "actor_total_loss": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "actor_grad_norm_max": 0.0,
            "critic_grad_norm_max": 0.0,
            "critic_loss_max": 0.0,
        }
        if c.get("separate_critic_updates", False):
            stats["actor_minibatches"] = 0
            stats["critic_minibatches"] = 0
        self.last_update_diagnostics["partial_metrics"] = stats
        stop = False
        for epoch in range(c["epochs"]):
            order = torch.randperm(length, device=device)
            for ids in order.split(c["batch_size"]):
                self.last_update_diagnostics.update(
                    phase="minibatch",
                    epoch=epoch,
                    completed_minibatches=stats["updates"],
                )
                mean, std = self.distribution_parameters(o[ids], prepared=True)
                new_lp = self.logprob(mean, std, raw[ids])
                log_ratio = new_lp - old_lp[ids]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = (
                        ((ratio - 1).abs() > c["clip"]).float().mean().item()
                    )
                if guard and not math.isfinite(approx_kl):
                    raise FloatingPointError("Nonfinite PPO likelihood ratio / KL")
                if c["target_kl"] > 0 and approx_kl > 1.5 * c["target_kl"]:
                    stop = True
                    if not c.get("separate_critic_updates", False):
                        break
                pg_loss = -torch.minimum(
                    ratio * adv[ids],
                    torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * adv[ids],
                ).mean()
                # Gaussian raw entropy is a proxy; squashed entropy differs.
                ent = (0.5 + math.log(2 * math.pi) / 2 + self.log_std).sum()
                if self.rho:
                    ent = ent + self.action_dim * 0.5 * math.log(
                        1 - self.rho * self.rho
                    )
                self.actor_optimizer.zero_grad(set_to_none=True)
                # Deterministic teacher regression ONLY during policy update.
                # We never blend teacher actions with behavior actions: old_lp,
                # logprob ratios and on-policy PPO sampling remain valid.
                if teacher is not None:
                    aux_loss = F.smooth_l1_loss(torch.tanh(mean), teacher[ids])
                else:
                    aux_loss = pg_loss.new_zeros(())
                actor_loss = pg_loss - c["ent_coef"] * ent + aux_pd_coef * aux_loss
                if guard and not bool(torch.isfinite(actor_loss).item()):
                    raise FloatingPointError("Nonfinite actor loss")
                if not stop:
                    actor_loss.backward()
                    actor_grad = nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + [self.log_std],
                        c["max_grad_norm"],
                        error_if_nonfinite=guard,
                    )
                    self.actor_optimizer.step()
                    if c.get("separate_critic_updates", False):
                        stats["actor_minibatches"] += 1
                else:
                    actor_grad = actor_loss.new_zeros(())
                with torch.no_grad():
                    self.log_std.clamp_(
                        c.get("log_std_min", -4.0), c.get("log_std_max", 1.0)
                    )
                values = self.critic(pr[ids]).squeeze(-1)

                def value_error(prediction):
                    if c.get("critic_loss", "mse") == "huber":
                        return F.huber_loss(
                            prediction,
                            normalized_ret[ids],
                            delta=c.get("critic_huber_delta", 1.0),
                            reduction="none",
                        )
                    return 0.5 * (prediction - normalized_ret[ids]).square()

                losses = value_error(values)
                clip = c.get("value_clip", 0.0)
                if clip > 0:
                    clipped = old_values[ids] + (values - old_values[ids]).clamp(
                        -clip, clip
                    )
                    losses = torch.maximum(losses, value_error(clipped))
                vf_loss = losses.mean()
                if guard and not bool(torch.isfinite(vf_loss).item()):
                    raise FloatingPointError("Nonfinite critic loss")
                self.critic_optimizer.zero_grad(set_to_none=True)
                vf_loss.backward()
                critic_grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    c["max_grad_norm"],
                    error_if_nonfinite=guard,
                )
                self.critic_optimizer.step()
                if c.get("separate_critic_updates", False):
                    stats["critic_minibatches"] += 1
                stats["approx_kl"] += approx_kl
                stats["clip_fraction"] += clip_fraction
                stats["actor_loss"] += float(pg_loss.detach().item())
                stats["aux_pd_loss"] += float(aux_loss.detach().item())
                stats["critic_loss"] += float(vf_loss.detach().item())
                stats["entropy_proxy"] += float(ent.detach().item())
                stats["actor_total_loss"] += float(actor_loss.detach().item())
                ag, cg = float(actor_grad.item()), float(critic_grad.item())
                stats["actor_grad_norm"] += ag
                stats["critic_grad_norm"] += cg
                stats["actor_grad_norm_max"] = max(stats["actor_grad_norm_max"], ag)
                stats["critic_grad_norm_max"] = max(stats["critic_grad_norm_max"], cg)
                stats["critic_loss_max"] = max(
                    stats["critic_loss_max"], float(vf_loss.detach().item())
                )
                stats["updates"] += 1
            if stop and not c.get("separate_critic_updates", False):
                break
        for k in (
            "approx_kl",
            "clip_fraction",
            "actor_loss",
            "aux_pd_loss",
            "critic_loss",
            "entropy_proxy",
            "actor_total_loss",
            "actor_grad_norm",
            "critic_grad_norm",
        ):
            stats[k] /= max(stats["updates"], 1)
        stats["early_stop"] = bool(stop)
        with torch.no_grad():
            pred = self.value_norm.denormalize(self.critic(pr).squeeze(-1))
            target_var = ret.var(unbiased=False)
            explained = 1 - (ret - pred).var(unbiased=False) / target_var.clamp_min(
                1e-8
            )
        stats["explained_variance"] = float(explained.item())
        stats["action_std"] = float(self.log_std.detach().exp().mean().item())
        value_mean, value_std = self.value_norm.statistics()
        stats.update(
            value_target_mean=float(value_mean),
            value_target_std=float(value_std),
            critic_loss_units=(
                "normalized_returns" if self.value_norm.enabled else "reward"
            ),
        )
        if diagnose:
            diagnostics = {
                "schema": "ppo-numerics-v1",
                "rollout": rollout_diagnostics,
                "pre_update_logprob_error_max": logprob_error,
                "post_update_values": buffer.summarize(pred),
                "normalized_advantages": buffer.summarize(adv),
                "value_error_after": buffer.summarize(pred - ret),
                "gradient_norms_are_before_clipping": True,
                "raw_policy_residuals": buffer.summarize(residual),
                "raw_policy_residual_autocorrelation": buffer.temporal_summary(
                    residual
                ),
            }
            reasons = []
            for name, threshold in (
                ("values", "value_abs_threshold"),
                ("next_values", "value_abs_threshold"),
                ("returns", "target_abs_threshold"),
                ("advantage", "advantage_abs_threshold"),
            ):
                maximum = rollout_diagnostics[name]["max_abs"]
                if maximum is not None and maximum > options[threshold]:
                    reasons.append(name + "_threshold")
            if stats["critic_loss_max"] > options["critic_loss_threshold"]:
                reasons.append("critic_loss_threshold")
            if (
                max(stats["actor_grad_norm_max"], stats["critic_grad_norm_max"])
                > options["gradient_norm_threshold"]
            ):
                reasons.append("gradient_norm_threshold")
            if logprob_error > options["logprob_error_threshold"]:
                reasons.append("pre_update_logprob_mismatch")
            for name, summary in (
                ("post_update_values", diagnostics["post_update_values"]),
                *rollout_diagnostics.items(),
            ):
                if isinstance(summary, dict) and (
                    summary.get("nonfinite_count", 0)
                    or summary.get("nonfinite_statistics")
                ):
                    reasons.append("nonfinite_" + name)
            diagnostics["anomaly_reasons"] = list(dict.fromkeys(reasons))
            maximum = diagnostics["post_update_values"]["max_abs"]
            if maximum is not None and maximum > options["value_abs_threshold"]:
                diagnostics["anomaly_reasons"].append("post_update_values_threshold")
            stats["diagnostics"] = diagnostics
            self.last_update_diagnostics = diagnostics
            if guard and diagnostics["post_update_values"]["nonfinite_count"]:
                raise FloatingPointError(
                    "Nonfinite critic predictions after PPO update"
                )
        return stats
