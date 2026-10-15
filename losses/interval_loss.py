"""Optional proper interval-score training objective."""

from __future__ import annotations

import torch


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

