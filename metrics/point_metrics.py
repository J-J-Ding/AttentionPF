"""Point-forecast metrics computed over every supplied element."""

from __future__ import annotations

import numpy as np


def point_metrics(point: np.ndarray, target: np.ndarray) -> dict[str, float]:
    point_values = np.asarray(point, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if point_values.shape != target_values.shape:
        raise ValueError("point and target shapes must match exactly")
    if point_values.size == 0 or not np.isfinite(point_values).all() or not np.isfinite(target_values).all():
        raise ValueError("point and target must be non-empty and finite")
    error = point_values - target_values
    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }

