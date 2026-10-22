"""Common interface for every forecasting model version."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseForecastModel(nn.Module, ABC):
    """Forecast models map history windows to ordered quantile forecasts."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions with shape ``[B, P, C_out, Q]``."""
        raise NotImplementedError

