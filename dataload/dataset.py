"""ETT-style CSV validation and leakage-safe sliding-window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LoadedTimeSeries:
    values: np.ndarray
    timestamps: pd.DatetimeIndex
    feature_cols: tuple[str, ...]


def load_time_series(
    path: str | Path, date_col: str, feature_cols: Sequence[str]
) -> LoadedTimeSeries:
    """Load, validate, and chronologically order an ETT-style CSV."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = [date_col, *feature_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing column(s): {', '.join(missing)}")

    timestamps = pd.to_datetime(frame[date_col], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Dataset contains an invalid timestamp in '{date_col}'")
    if timestamps.duplicated().any():
        raise ValueError("Dataset contains a duplicate timestamp")

    numeric = frame.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Dataset features contain NaN or infinity")

    order = np.argsort(timestamps.to_numpy(), kind="stable")
    sorted_timestamps = pd.DatetimeIndex(timestamps.iloc[order].to_numpy())
    sorted_values = values[order].astype(np.float32)
    return LoadedTimeSeries(
        values=sorted_values,
        timestamps=sorted_timestamps,
        feature_cols=tuple(feature_cols),
    )


class TimeSeriesWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windows a shared scaled timeline while constraining forecast targets."""

    def __init__(
        self,
        values: np.ndarray,
        target_start: int,
        target_end: int,
        seq_len: int,
        pred_len: int,
        stride: int,
        output_indices: Sequence[int],
        split_name: str,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("values must be a finite [time, channels] array")
        if seq_len <= 0 or pred_len <= 0 or stride <= 0:
            raise ValueError("seq_len, pred_len, and stride must be positive")
        if not 0 <= target_start < target_end <= len(values):
            raise ValueError(f"Invalid target bounds for {split_name} split")
        required = seq_len + pred_len
        if target_end - target_start < required:
            raise ValueError(
                f"{split_name} split needs at least {required} rows, "
                f"got {target_end - target_start}"
            )

        normalized_indices = []
        for index in output_indices:
            normalized = index + values.shape[1] if index < 0 else index
            if normalized < 0 or normalized >= values.shape[1]:
                raise ValueError(f"Output channel index {index} is out of range")
            normalized_indices.append(normalized)
        if not normalized_indices:
            raise ValueError("At least one output channel is required")

        first_forecast = max(target_start, seq_len)
        last_forecast = target_end - pred_len
        self.forecast_starts = np.arange(
            first_forecast, last_forecast + 1, stride, dtype=np.int64
        )
        if len(self.forecast_starts) == 0:
            raise ValueError(f"{split_name} split produces no complete windows")

        self.values = values
        self.target_start = int(target_start)
        self.target_end = int(target_end)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.output_indices = tuple(normalized_indices)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.forecast_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        forecast_start = int(self.forecast_starts[index])
        history = self.values[forecast_start - self.seq_len : forecast_start]
        future = self.values[
            forecast_start : forecast_start + self.pred_len,
            list(self.output_indices),
        ]
        return torch.from_numpy(history.copy()), torch.from_numpy(future.copy())

