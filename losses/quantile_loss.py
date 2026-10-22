"""Pinball loss for an ordered set of quantile prediction slots."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        values = tuple(float(value) for value in quantiles)
        if not values:
            raise ValueError("at least one quantile is required")
        if not all(0.0 < value < 1.0 for value in values) or not all(
            left < right for left, right in zip(values, values[1:])
        ):
            raise ValueError("quantiles must be strictly increasing inside (0, 1)")
        self.register_buffer("quantiles", torch.tensor(values, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4 or prediction.shape[-1] != self.quantiles.numel():
            raise ValueError(
                "prediction shape must be [B,P,C,Q] with Q equal to configured quantiles"
            )
        if target.shape != prediction.shape[:-1]:
            raise ValueError(
                f"target shape {tuple(target.shape)} must equal prediction prefix "
                f"{tuple(prediction.shape[:-1])}"
            )
        error = target.unsqueeze(-1) - prediction
        quantiles = self.quantiles.to(dtype=prediction.dtype, device=prediction.device)
        return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()

