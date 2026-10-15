"""Prediction-interval coverage, width, and proper scoring metrics."""

from __future__ import annotations

import numpy as np


def interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    coverage: float,
    target_range: float | None = None,
) -> dict[str, float | None]:
    lower_values = np.asarray(lower, dtype=np.float64)
    upper_values = np.asarray(upper, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if lower_values.shape != upper_values.shape or lower_values.shape != target_values.shape:
        raise ValueError("lower, upper, and target shapes must match exactly")
    if lower_values.size == 0:
        raise ValueError("interval arrays must be non-empty")
    if not all(np.isfinite(values).all() for values in (lower_values, upper_values, target_values)):
        raise ValueError("interval arrays must be finite")
    if np.any(lower_values > upper_values):
        raise ValueError("lower values must not exceed upper values")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be inside (0, 1)")

    width = upper_values - lower_values
    covered = (lower_values <= target_values) & (target_values <= upper_values)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(width))
    normalization = float(np.ptp(target_values)) if target_range is None else float(target_range)
    pinaw = None if normalization == 0.0 else float(mpiw / normalization)
    alpha = 1.0 - coverage
    score = width.copy()
    below = target_values < lower_values
    above = target_values > upper_values
    score[below] += (2.0 / alpha) * (lower_values[below] - target_values[below])
    score[above] += (2.0 / alpha) * (target_values[above] - upper_values[above])
    return {
        "picp": picp,
        "mpiw": mpiw,
        "pinaw": pinaw,
        "coverage_error": float(abs(picp - coverage)),
        "interval_score": float(np.mean(score)),
    }

