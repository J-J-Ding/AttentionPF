"""Serializable per-channel standardization fitted on training rows only."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class StandardScaler:
    def __init__(self, epsilon: float = 1.0e-8) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.scale_ is not None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        values = np.asarray(data)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("fit data must have shape [time, channels] and be non-empty")
        if not np.isfinite(values).all():
            raise ValueError("fit data contains NaN or infinity")
        values64 = values.astype(np.float64, copy=False)
        self.mean_ = values64.mean(axis=0)
        std = values64.std(axis=0)
        self.scale_ = np.maximum(std, self.epsilon)
        return self

    def _statistics(
        self, data: np.ndarray, channel_indices: Sequence[int] | None
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if not self.is_fitted:
            raise RuntimeError("StandardScaler must be fitted before use")
        values = np.asarray(data)
        channel_axis = values.ndim - 2 if values.ndim == 4 else values.ndim - 1
        indices = np.arange(len(self.mean_)) if channel_indices is None else np.asarray(channel_indices)
        indices = np.where(indices < 0, indices + len(self.mean_), indices)
        if np.any(indices < 0) or np.any(indices >= len(self.mean_)):
            raise ValueError("channel_indices contains an out-of-range channel")
        if values.shape[channel_axis] != len(indices):
            raise ValueError(
                f"data has {values.shape[channel_axis]} channels but {len(indices)} statistics were selected"
            )
        shape = [1] * values.ndim
        shape[channel_axis] = len(indices)
        mean = self.mean_[indices].reshape(shape)
        scale = self.scale_[indices].reshape(shape)
        return mean, scale, channel_axis

    def transform(
        self, data: np.ndarray, channel_indices: Sequence[int] | None = None
    ) -> np.ndarray:
        values = np.asarray(data)
        mean, scale, _ = self._statistics(values, channel_indices)
        return ((values - mean) / scale).astype(values.dtype, copy=False)

    def inverse_transform(
        self, data: np.ndarray, channel_indices: Sequence[int] | None = None
    ) -> np.ndarray:
        values = np.asarray(data)
        mean, scale, _ = self._statistics(values, channel_indices)
        return (values * scale + mean).astype(values.dtype, copy=False)

    def state_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted StandardScaler")
        return {
            "epsilon": self.epsilon,
            "mean": self.mean_.copy(),
            "scale": self.scale_.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        required = {"epsilon", "mean", "scale"}
        if set(state) != required:
            raise ValueError(f"Scaler state must contain exactly {sorted(required)}")
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape or np.any(scale <= 0):
            raise ValueError("Invalid scaler state shapes or scale values")
        self.epsilon = float(state["epsilon"])
        self.mean_ = mean.copy()
        self.scale_ = scale.copy()

