"""On-policy storage, timeout-aware GAE, and read-only rollout diagnostics."""

import math
import numpy as np
import torch
from pursuit_rl.simulation.torch_device import resolve_torch_device


DIAGNOSTIC_DEFAULTS = {
    "enabled": True,
    "max_quantile_samples": 8192,
    "critic_loss_threshold": 100.0,
    "value_abs_threshold": 1000.0,
    "target_abs_threshold": 1000.0,
    "advantage_abs_threshold": 1000.0,
    "gradient_norm_threshold": 1000.0,
    "logprob_error_threshold": 0.0001,
    "save_first_anomaly": True,
    "fail_on_nonfinite": True,
    "calibration_every_updates": 0,
    "calibration_episodes": 32,
    "calibration_seed": 1900000,
}


def diagnostic_options(overrides=None):
    result = dict(DIAGNOSTIC_DEFAULTS)
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(result):
        raise ValueError("Unknown or malformed PPO diagnostics options")
    result.update(overrides)
    for key, default in DIAGNOSTIC_DEFAULTS.items():
        value = result[key]
        if isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = type(value) is int and value >= 0
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        if not valid:
            raise ValueError(f"Invalid diagnostics option: {key}")
    if result["max_quantile_samples"] < 2 or result["calibration_episodes"] < 1:
        raise ValueError(
            "Diagnostics require at least two quantile samples and one episode"
        )
    return result


def distribution_summary(values, max_samples=8192):
    """Exact finite moments/extrema; deterministic, bounded-sample quantiles.

    Never draw from a random generator or emit non-standard JSON NaN/Infinity.
    The sample count explicitly identifies approximate quantiles.
    """
    if isinstance(values, torch.Tensor):
        flat = values.detach().reshape(-1)
        finite = flat[torch.isfinite(flat)].to(torch.float64)
        count, finite_count = flat.numel(), finite.numel()
        if finite_count:
            if finite_count > max_samples:
                ids = torch.linspace(
                    0,
                    finite_count - 1,
                    max_samples,
                    device=finite.device,
                    dtype=torch.float64,
                ).long()
                sample = finite[ids]
            else:
                sample = finite
            qs = torch.quantile(sample, sample.new_tensor([0.01, 0.5, 0.99]))
            numbers = (
                torch.stack(
                    (
                        finite.mean(),
                        finite.std(unbiased=False),
                        finite.min(),
                        finite.max(),
                        finite.abs().max(),
                        *qs,
                    )
                )
                .cpu()
                .tolist()
            )
        else:
            sample = finite
            numbers = [None] * 8
    else:
        flat = np.asarray(values).reshape(-1)
        finite = flat[np.isfinite(flat)].astype(np.float64)
        count, finite_count = flat.size, finite.size
        sample = (
            finite[np.linspace(0, finite_count - 1, max_samples).astype(np.int64)]
            if finite_count > max_samples
            else finite
        )
        numbers = (
            [
                float(finite.mean()),
                float(finite.std()),
                float(finite.min()),
                float(finite.max()),
                float(np.abs(finite).max()),
                *np.quantile(sample, [0.01, 0.5, 0.99]).tolist(),
            ]
            if finite_count
            else [None] * 8
        )
    result = dict(
        zip(("mean", "std", "min", "max", "max_abs", "p01", "p50", "p99"), numbers)
    )
    result["nonfinite_statistics"] = [
        k for k, v in result.items() if v is not None and not math.isfinite(v)
    ]
    result = {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in result.items()
    }
    result.update(
        count=int(count),
        finite_count=int(finite_count),
        nonfinite_count=int(count - finite_count),
        quantile_sample_count=int(len(sample)),
    )
    return result


class _DiagnosticBuffer:
    def _init_diagnostics(self, options):
        self.diagnostic_options = diagnostic_options(options)
        self.reward_components = {}
        self.physics = None
        self.diagnostic_steps = 0
        self.gamma = None
        self.gae_lambda = None

    def summarize(self, values):
        return distribution_summary(
            values, self.diagnostic_options["max_quantile_samples"]
        )

    def temporal_summary(self, values):
        """Correlations within episodes, with lags expressed in physical time."""
        tensor = isinstance(values, torch.Tensor)
        dt = self.physics["dt"] if self.physics else None
        lags = {1}
        if dt:
            lags.update((max(1, round(0.05 / dt)), max(1, round(0.13 / dt))))
        if tensor:
            starts = torch.as_tensor(
                self.episode_start, device=values.device, dtype=torch.long
            )
        else:
            starts = (
                self.episode_start.detach().cpu().numpy()
                if isinstance(self.episode_start, torch.Tensor)
                else self.episode_start
            )
            starts = starts.astype(np.int64)
        ids = starts.cumsum(0)
        result = {}
        for lag in sorted(lags):
            if lag >= len(values):
                continue
            valid = ids[lag:] == ids[:-lag]
            a, b = values[:-lag][valid], values[lag:][valid]
            correlations = []
            for axis in range(values.shape[-1]):
                x, y = a[:, axis], b[:, axis]
                finite = (
                    (torch.isfinite(x) & torch.isfinite(y))
                    if tensor
                    else (np.isfinite(x) & np.isfinite(y))
                )
                x, y = x[finite], y[finite]
                if len(x) < 2:
                    correlations.append(None)
                    continue
                x, y = x - x.mean(), y - y.mean()
                denominator = (
                    (x.square().sum() * y.square().sum()).sqrt()
                    if tensor
                    else np.sqrt((x * x).sum() * (y * y).sum())
                )
                scale = float(denominator)
                correlations.append(
                    float((x * y).sum() / denominator)
                    if math.isfinite(scale) and scale > 1e-12
                    else None
                )
            result[str(lag)] = {
                "lag_steps": lag,
                "lag_seconds": lag * dt if dt else None,
                "correlation_by_axis": correlations,
            }
        return result

    def record_diagnostics(self, k, env, episode_start, info=None):
        if not self.diagnostic_options["enabled"]:
            return
        tensor = isinstance(self.reward, torch.Tensor)
        self.physics = dict(env.cfg["physics"])
        self.episode_start[k] = episode_start
        self.elapsed[k] = env.elapsed
        u = env.pursuer
        if tensor:
            self.actuator_state[k] = torch.cat((u.thrust[:, None], u.rate), dim=1)
            ev = env.last_event
            self.event_code[k] = torch.where(
                ev["hit"],
                1,
                torch.where(
                    ev["speed"] > self.physics["max_speed"],
                    2,
                    torch.where(
                        ev["outside"],
                        3,
                        torch.where(
                            ev["failed"], 4, torch.where(self.truncated[k], 5, 0)
                        ),
                    ),
                ),
            )
        else:
            self.actuator_state[k] = np.column_stack((u.thrust, u.rate))
            codes = {"success": 1, "speed": 2, "outside": 3, "tilt": 4, "timeout": 5}
            self.event_code[k] = [codes.get(rec.get("reason"), 0) for rec in info]
        for name, value in env.reward_components.items():
            if name not in self.reward_components:
                self.reward_components[name] = (
                    torch.zeros_like(self.reward)
                    if tensor
                    else np.zeros_like(self.reward)
                )
            self.reward_components[name][k] = value
        self.diagnostic_steps = k + 1

    def diagnostic_summary(self):
        fields = (
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
        result = {name: self.summarize(getattr(self, name)) for name in fields}
        result["reward_components"] = {
            k: self.summarize(v) for k, v in self.reward_components.items()
        }
        tensor = isinstance(self.reward, torch.Tensor)

        def count(mask):
            return int(mask.sum().item()) if tensor else int(mask.sum())

        done = self.terminated | self.truncated
        masks = {
            "terminated": self.terminated,
            "truncated": self.truncated & ~self.terminated,
            "ongoing": ~done,
            "episode_start": self.episode_start,
        }
        edge = torch.zeros_like(done) if tensor else np.zeros_like(done)
        edge[-1] = ~done[-1]
        masks["rollout_edge"] = edge
        result["events"] = {}
        for label, mask in masks.items():
            result["events"][label] = {
                "count": count(mask),
                **{
                    field: self.summarize(getattr(self, field)[mask])
                    for field in (
                        "values",
                        "next_values",
                        "bootstrap_values",
                        "delta",
                        "advantage",
                        "returns",
                    )
                },
            }
        if self.diagnostic_steps:
            for label, code in (
                ("success", 1),
                ("speed", 2),
                ("outside", 3),
                ("tilt", 4),
                ("timeout", 5),
            ):
                mask = self.event_code == code
                result["events"][label] = {
                    "count": count(mask),
                    "terminal_elapsed_seconds": self.summarize(self.elapsed[mask]),
                    "value_before_transition": self.summarize(self.values[mask]),
                    "reward": self.summarize(self.reward[mask]),
                }
            action = torch.tanh(self.raw) if tensor else np.tanh(self.raw)
            result["actions"] = {
                "normalized": [
                    self.summarize(action[..., i]) for i in range(action.shape[-1])
                ],
                "actual_thrust_and_rates": [
                    self.summarize(self.actuator_state[..., i]) for i in range(4)
                ],
            }
            result["actions"]["command_autocorrelation"] = self.temporal_summary(action)
            result["actions"]["actuator_autocorrelation"] = self.temporal_summary(
                self.actuator_state
            )
            if self.physics and getattr(self, "action_semantics", "ctbr") == "ctbr":
                p = self.physics
                unsaturated = p["hover_thrust"] + p["thrust_delta"] * action[..., 0]
                cmd = (
                    unsaturated.clamp(p["thrust_min"], p["thrust_max"])
                    if tensor
                    else np.clip(unsaturated, p["thrust_min"], p["thrust_max"])
                )
                result["actions"]["command_thrust"] = self.summarize(cmd)
                samples = self.reward.numel() if tensor else self.reward.size
                result["actions"]["thrust_lower_saturation_fraction"] = (
                    count(unsaturated <= p["thrust_min"]) / samples
                )
        result["gamma"] = self.gamma
        result["gae_lambda"] = self.gae_lambda
        return result

    def diagnostic_snapshot(self, include_teacher=False):
        names = [
            "obs",
            "priv",
            "raw",
            "logprob",
            "values",
            "reward",
            "next_values",
            "bootstrap_values",
            "delta",
            "terminated",
            "truncated",
            "advantage",
            "returns",
            "episode_start",
            "actuator_state",
            "event_code",
            "elapsed",
        ]
        if include_teacher:
            names.append("teacher")

        def host(value):
            return (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else torch.from_numpy(np.asarray(value).copy())
            )

        return {
            "tensors": {name: host(getattr(self, name)) for name in names},
            "reward_components": {
                name: host(value) for name, value in self.reward_components.items()
            },
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "physics": self.physics,
            "event_codes": {
                "ongoing": 0,
                "success": 1,
                "speed": 2,
                "outside": 3,
                "tilt": 4,
                "timeout": 5,
            },
        }


class RolloutBuffer(_DiagnosticBuffer):
    def __init__(self, steps, n, obs_dim, critic_dim, diagnostics=None, action_dim=4):
        shape = (steps, n)
        for name, size in {
            "obs": (obs_dim,),
            "priv": (critic_dim,),
            "raw": (action_dim,),
            "teacher": (action_dim,),
        }.items():
            setattr(self, name, np.zeros(shape + size, np.float32))
        for name in (
            "logprob",
            "values",
            "reward",
            "next_values",
            "bootstrap_values",
            "delta",
            "advantage",
            "returns",
            "elapsed",
        ):
            setattr(self, name, np.zeros(shape, np.float32))
        for name in ("terminated", "truncated", "episode_start"):
            setattr(self, name, np.zeros(shape, bool))
        self.actuator_state = np.zeros(shape + (4,), np.float32)
        self.event_code = np.zeros(shape, np.int8)
        self._init_diagnostics(diagnostics)

    def compute_gae(self, gamma, lam):
        """Bootstrap external truncations at final_obs; never recurse past reset."""
        self.gamma, self.gae_lambda = float(gamma), float(lam)
        gae = np.zeros(self.reward.shape[1], np.float64)
        for t in range(self.reward.shape[0] - 1, -1, -1):
            continuation = ~(self.terminated[t] | self.truncated[t])
            bootstrap = ~self.terminated[t]
            boot_value = np.where(bootstrap, self.next_values[t], 0.0).astype(
                np.float64
            )
            delta = self.reward[t] + gamma * boot_value - self.values[t]
            gae = delta + gamma * lam * np.where(continuation, gae, 0.0)
            self.bootstrap_values[t] = boot_value
            self.delta[t] = delta
            self.advantage[t] = gae.astype(np.float32)
        self.returns = self.advantage + self.values


class TensorRolloutBuffer(_DiagnosticBuffer):
    """Device-resident rollout storage; summaries synchronize once per rollout."""

    def __init__(
        self, steps, n, obs_dim, critic_dim, device, diagnostics=None, action_dim=4
    ):
        self.device = resolve_torch_device(device)
        shape = (steps, n)
        for name, size in {
            "obs": (obs_dim,),
            "priv": (critic_dim,),
            "raw": (action_dim,),
            "teacher": (action_dim,),
        }.items():
            setattr(self, name, torch.empty(shape + size, device=self.device))
        for name in (
            "logprob",
            "values",
            "reward",
            "next_values",
            "bootstrap_values",
            "delta",
            "advantage",
            "returns",
            "elapsed",
        ):
            setattr(self, name, torch.zeros(shape, device=self.device))
        for name in ("terminated", "truncated", "episode_start"):
            setattr(
                self, name, torch.zeros(shape, device=self.device, dtype=torch.bool)
            )
        self.actuator_state = torch.zeros(shape + (4,), device=self.device)
        self.event_code = torch.zeros(shape, device=self.device, dtype=torch.int8)
        self._init_diagnostics(diagnostics)

    @torch.no_grad()
    def compute_gae(self, gamma, lam):
        self.gamma, self.gae_lambda = float(gamma), float(lam)
        gae = torch.zeros(self.reward.shape[1], device=self.device)
        for k in range(self.reward.shape[0] - 1, -1, -1):
            cont = ~(self.terminated[k] | self.truncated[k])
            boot = ~self.terminated[k]
            boot_value = torch.where(boot, self.next_values[k], 0.0)
            delta = self.reward[k] + gamma * boot_value - self.values[k]
            gae = delta + gamma * lam * torch.where(cont, gae, 0.0)
            self.bootstrap_values[k] = boot_value
            self.delta[k] = delta
            self.advantage[k] = gae
        self.returns.copy_(self.advantage + self.values)
