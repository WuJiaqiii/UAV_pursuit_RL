"""Causal target forecasting, supervised only after a PPO rollout is consumed.

A history row contains observed world position, velocity, acceleration and a
validity mask. Positions are rebased to the latest target position before the
LSTM, so absolute world coordinates cannot become a shortcut. Future truth is
only a label; it never enters actor observations or the forecasting forward pass.
"""

import copy
import torch
from torch import nn
import torch.nn.functional as F


class TrajectoryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("horizons", torch.tensor(config["horizons_s"]))
        if config["mode"] == "lstm":
            self.lstm = nn.LSTM(10, config["hidden_size"], batch_first=True)
            self.head = nn.Linear(config["hidden_size"], 3 * len(self.horizons))
            # Begin with a causal constant-acceleration forecast.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, history):
        last = history[:, -1]
        h = self.horizons.to(history)[None, :, None]
        baseline = last[:, None, 3:6] * h
        if self.config["mode"] != "constant_velocity":
            baseline = baseline + 0.5 * last[:, None, 6:9] * h.square()
        scale = self.config["position_scale"]
        prediction = baseline / scale
        if self.config["mode"] == "lstm":
            centered = history.clone()
            centered[:, :, :3] = (history[:, :, :3] - last[:, None, :3]) / scale
            centered[:, :, 3:6] /= 15.0
            centered[:, :, 6:9] /= 7.0
            centered[:, :, :9] *= history[:, :, 9:10]
            output, _ = self.lstm(centered)
            prediction = prediction + self.head(output[:, -1]).reshape(
                -1, len(self.horizons), 3
            )
        return prediction


@torch.no_grad()
def supervised_windows(histories, next_positions, done, offsets, max_samples):
    """Return history/future-displacement pairs without crossing reset boundaries.

    histories[t] is observed BEFORE decision t; next_positions[t] is captured
    AFTER its physics substeps and BEFORE reset. Conservatively exclude even
    terminal endpoints, which may have shorter-than-nominal decision intervals.
    The final incomplete forecast window is discarded, never bootstrapped.
    """
    length, n = done.shape
    usable = length - max(offsets) + 1
    if usable <= 0:
        return None
    valid = histories[:usable, :, -1, 9] > 0
    labels = []
    origin = histories[:usable, :, -1, :3]
    for offset in offsets:
        labels.append(next_positions[offset - 1 : offset - 1 + usable] - origin)
        for tick in range(offset):
            valid &= ~done[tick : tick + usable]
    x = histories[:usable][valid]
    y = torch.stack(labels, 2)[valid]
    if not len(x):
        return None
    if len(x) > max_samples:
        ids = torch.randperm(len(x), device=x.device)[:max_samples]
        x, y = x[ids], y[ids]
    return x, y


def fit_predictor(model, data):
    """Separate supervised optimizer; caller must have finished the PPO update."""
    if data is None or model.predictor_optimizer is None:
        return dict(samples=0, updates=0)
    x, displacement = data
    config = model.prediction_config
    target = displacement / config["position_scale"]
    order = torch.randperm(len(x), device=x.device)
    guard_size = max(1, len(x) // 5)
    guard_ids = order[:guard_size]
    train_ids = order[guard_size:] if len(x) > 1 else order
    backup = copy.deepcopy(model.predictor.state_dict())
    optimizer_backup = copy.deepcopy(model.predictor_optimizer.state_dict())
    h = model.predictor.horizons.to(x)[None, :, None]
    cv = x[:, -1:, 3:6] * h
    ca = cv + 0.5 * x[:, -1:, 6:9] * h.square()
    with torch.no_grad():
        before = (
            float(F.mse_loss(model.predictor(x), target).sqrt())
            * config["position_scale"]
        )
        guard_before = float(
            F.mse_loss(model.predictor(x[guard_ids]), target[guard_ids])
        )
    updates = 0
    for _ in range(config["epochs"]):
        for ids in train_ids[torch.randperm(len(train_ids), device=x.device)].split(
            config["batch_size"]
        ):
            loss = F.mse_loss(model.predictor(x[ids]), target[ids])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite trajectory prediction loss")
            model.predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.predictor.parameters(), 1.0, error_if_nonfinite=True
            )
            model.predictor_optimizer.step()
            updates += 1
    with torch.no_grad():
        guard_after = float(
            F.mse_loss(model.predictor(x[guard_ids]), target[guard_ids])
        )
        accepted = guard_after <= guard_before * 1.001 + 1e-12
        if not accepted:
            model.predictor.load_state_dict(backup)
            model.predictor_optimizer.load_state_dict(optimizer_backup)
        predicted = model.predictor(x) * config["position_scale"]
        rmse = (predicted - displacement).square().mean((0, 2)).sqrt()
    return dict(
        samples=len(x),
        updates=updates if accepted else 0,
        attempted_updates=updates,
        accepted=accepted,
        guard_samples=len(guard_ids),
        guard_rmse_before_m=guard_before**0.5 * config["position_scale"],
        candidate_guard_rmse_after_m=guard_after**0.5 * config["position_scale"],
        fit_rmse_before_m=before,
        fit_rmse_after_m=float((predicted - displacement).square().mean().sqrt()),
        fit_rmse_by_horizon_m=rmse.cpu().tolist(),
        constant_velocity_rmse_m=float((cv - displacement).square().mean().sqrt()),
        constant_acceleration_rmse_m=float((ca - displacement).square().mean().sqrt()),
        horizon_seconds=config["horizons_s"],
        evaluation_population="rollout windows with a 20% fit guard; independent frozen evaluation is required for generalization",
    )


class PredictionMetrics:
    """Bounded-memory causal forecast audit on complete held-out trajectories."""

    def __init__(self, model, decision_dt):
        self.model = model
        self.offsets = [
            round(h / decision_dt) for h in model.prediction_config["horizons_s"]
        ]
        self.pending = []
        self.squared = torch.zeros(
            3, len(self.offsets), dtype=torch.float64, device=model.device
        )
        self.counts = torch.zeros(
            len(self.offsets), dtype=torch.float64, device=model.device
        )

    @torch.no_grad()
    def begin(self, history, active):
        last = history[:, -1]
        h = self.model.predictor.horizons.to(history)[None, :, None]
        cv = last[:, None, 3:6] * h
        ca = cv + 0.5 * last[:, None, 6:9] * h.square()
        pred = (
            self.model.predictor(history)
            * self.model.prediction_config["position_scale"]
        )
        self.pending.append(
            dict(
                age=0,
                origin=last[:, :3].clone(),
                forecasts=torch.stack((pred, cv, ca)),
                valid=active.clone(),
            )
        )

    @torch.no_grad()
    def advance(self, positions, done):
        kept = []
        for item in self.pending:
            item["age"] += 1
            item["valid"] &= ~done
            if item["age"] in self.offsets:
                k = self.offsets.index(item["age"])
                mask = item["valid"]
                truth = (positions - item["origin"]).float()
                errors = (item["forecasts"][:, :, k] - truth[None]).double().square()
                self.squared[:, k] += (errors * mask[None, :, None]).sum((1, 2))
                self.counts[k] += mask.sum() * 3
            if item["age"] < max(self.offsets):
                kept.append(item)
        self.pending = kept

    def report(self):
        counts = self.counts.cpu().tolist()
        rmse = (self.squared / self.counts.clamp_min(1)[None]).sqrt().cpu().tolist()
        return dict(
            horizons_s=self.model.prediction_config["horizons_s"],
            samples_by_horizon=[int(n / 3) for n in counts],
            rmse_by_horizon_m={
                name: [v if counts[k] else None for k, v in enumerate(row)]
                for name, row in zip(
                    ("learned", "constant_velocity", "constant_acceleration"), rmse
                )
            },
            terminal_crossing_windows_excluded=True,
        )
