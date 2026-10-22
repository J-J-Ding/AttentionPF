"""Optional proper interval-score training objective."""

from __future__ import annotations

import torch
from torch import nn

from losses.quantile_loss import QuantileLoss


def interval_score_loss(
    lower: torch.Tensor,
    upper: torch.Tensor,
    target: torch.Tensor,
    coverage: float,
) -> torch.Tensor:
    if lower.shape != upper.shape or lower.shape != target.shape:
        raise ValueError("lower, upper, and target shapes must match exactly")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be inside (0, 1)")
    alpha = 1.0 - coverage
    width = upper - lower
    below = (2.0 / alpha) * (lower - target) * (target < lower)
    above = (2.0 / alpha) * (target - upper) * (target > upper)
    return (width + below + above).mean()


class QuantileIntervalLoss(nn.Module):
    """Quantile loss with an optional proper interval-score contribution."""

    def __init__(self, quantiles, coverage: float, interval_weight: float) -> None:
        super().__init__()
        if interval_weight < 0.0:
            raise ValueError("interval_weight must be non-negative")
        values = tuple(float(value) for value in quantiles)
        self.quantile = QuantileLoss(values)
        self.coverage = float(coverage)
        self.interval_weight = float(interval_weight)
        lower = (1.0 - self.coverage) / 2.0
        upper = (1.0 + self.coverage) / 2.0
        try:
            self.lower_index = next(
                index for index, value in enumerate(values) if abs(value - lower) <= 1.0e-9
            )
            self.upper_index = next(
                index for index, value in enumerate(values) if abs(value - upper) <= 1.0e-9
            )
        except StopIteration as error:
            raise ValueError(
                "coverage requires matching central lower and upper quantiles"
            ) from error

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.quantile(prediction, target)
        if self.interval_weight == 0.0:
            return loss
        interval = interval_score_loss(
            prediction[..., self.lower_index],
            prediction[..., self.upper_index],
            target,
            self.coverage,
        )
        return loss + self.interval_weight * interval

