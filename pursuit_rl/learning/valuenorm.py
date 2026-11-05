"""Debiased running normalization of return targets (never of environment rewards)."""

import torch
from torch import nn


class ValueNorm(nn.Module):
    def __init__(self, enabled=True, beta=0.995, min_variance=0.01):
        super().__init__()
        self.enabled = enabled
        self.beta = beta
        self.min_variance = min_variance
        for name in ("mean_ema", "square_ema", "weight"):
            self.register_buffer(name, torch.zeros((), dtype=torch.float64))

    def statistics(self):
        if not self.enabled or self.weight.item() == 0:
            return self.mean_ema.new_zeros(()), self.mean_ema.new_ones(())
        mean = self.mean_ema / self.weight
        variance = (self.square_ema / self.weight - mean.square()).clamp_min(
            self.min_variance
        )
        return mean, variance.sqrt()

    @torch.no_grad()
    def update(self, returns):
        if not self.enabled:
            return
        values = returns.detach().double().reshape(-1)
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("ValueNorm requires nonempty finite returns")
        self.mean_ema.mul_(self.beta).add_(values.mean(), alpha=1 - self.beta)
        self.square_ema.mul_(self.beta).add_(
            values.square().mean(), alpha=1 - self.beta
        )
        self.weight.mul_(self.beta).add_(1 - self.beta)

    def normalize(self, values):
        mean, std = self.statistics()
        return (values - mean.to(values)) / std.to(values)

    def denormalize(self, values):
        mean, std = self.statistics()
        return values * std.to(values) + mean.to(values)
