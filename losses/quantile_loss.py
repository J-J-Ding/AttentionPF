"""Pinball loss for fixed lower/median/upper prediction slots."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        if len(quantiles) != 3:
            raise ValueError("Exactly three quantiles are required")
        values = tuple(float(value) for value in quantiles)
        if not (0.0 < values[0] < values[1] < values[2] < 1.0):
            raise ValueError("quantiles must be strictly increasing inside (0, 1)")
        self.register_buffer("quantiles", torch.tensor(values, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4 or prediction.shape[-1] != 3:
            raise ValueError("prediction shape must be [B,P,C,3]")
        if target.shape != prediction.shape[:-1]:
            raise ValueError(
                f"target shape {tuple(target.shape)} must equal prediction prefix "
                f"{tuple(prediction.shape[:-1])}"
            )
        error = target.unsqueeze(-1) - prediction
        quantiles = self.quantiles.to(dtype=prediction.dtype, device=prediction.device)
        return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()

