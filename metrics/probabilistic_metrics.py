"""Distributional scores reconstructed from a finite set of quantile forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_inputs(
    prediction: np.ndarray, target: np.ndarray, quantiles: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_values = np.asarray(prediction, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    quantile_values = np.asarray(tuple(quantiles), dtype=np.float64)
    if prediction_values.ndim < 1 or prediction_values.shape[-1] != quantile_values.size:
        raise ValueError("prediction's last dimension must match quantiles")
    if target_values.shape != prediction_values.shape[:-1]:
        raise ValueError("target shape must equal prediction shape without its quantile axis")
    if quantile_values.size < 2:
        raise ValueError("at least two quantiles are required")
    if not np.all((0.0 < quantile_values) & (quantile_values < 1.0)):
        raise ValueError("quantiles must be inside (0, 1)")
    if not np.all(np.diff(quantile_values) > 0.0):
        raise ValueError("quantiles must be strictly increasing")
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("prediction and target must be finite")
    if np.any(np.diff(prediction_values, axis=-1) < 0.0):
        raise ValueError("predicted quantiles must be nondecreasing")
    return prediction_values, target_values, quantile_values


def _complete_quantile_function(
    prediction: np.ndarray, quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly extrapolate the finite quantile function to probabilities 0 and 1."""
    left_slope = (prediction[..., 1] - prediction[..., 0]) / (
        quantiles[1] - quantiles[0]
    )
    right_slope = (prediction[..., -1] - prediction[..., -2]) / (
        quantiles[-1] - quantiles[-2]
    )
    left = prediction[..., 0] - quantiles[0] * left_slope
    right = prediction[..., -1] + (1.0 - quantiles[-1]) * right_slope
    levels = np.concatenate(([0.0], quantiles, [1.0]))
    values = np.concatenate(
        (left[..., None], prediction, right[..., None]), axis=-1
    )
    return levels, values


def _integrated_pinball(
    target: np.ndarray,
    level_low: float,
    level_high: float,
    quantile_low: np.ndarray,
    quantile_high: np.ndarray,
) -> np.ndarray:
    """Exactly integrate pinball loss over one linear quantile-function segment."""
    slope = (quantile_high - quantile_low) / (level_high - level_low)
    intercept = quantile_low - slope * level_low
    residual_intercept = target - intercept

    def positive_primitive(level: np.ndarray | float) -> np.ndarray:
        return (
            0.5 * residual_intercept * level**2
            - (slope / 3.0) * level**3
        )

    def negative_primitive(level: np.ndarray | float) -> np.ndarray:
        return (
            0.5 * (residual_intercept + slope) * level**2
            - residual_intercept * level
            - (slope / 3.0) * level**3
        )

    positive = positive_primitive(level_high) - positive_primitive(level_low)
    negative = negative_primitive(level_high) - negative_primitive(level_low)
    flat = slope == 0.0
    crossing = np.divide(
        target - intercept,
        slope,
        out=np.full_like(target, level_low),
        where=~flat,
    )
    crossing = np.clip(crossing, level_low, level_high)
    split = (
        positive_primitive(crossing)
        - positive_primitive(level_low)
        + negative_primitive(level_high)
        - negative_primitive(crossing)
    )
    return np.where(
        target >= quantile_high,
        positive,
        np.where(target <= quantile_low, negative, np.where(flat, positive, split)),
    )


def quantile_crps(
    prediction: np.ndarray, target: np.ndarray, quantiles: Sequence[float]
) -> float:
    """Approximate mean CRPS using a piecewise-linear reconstructed quantile function."""
    prediction_values, target_values, quantile_values = _validate_inputs(
        prediction, target, quantiles
    )
    levels, values = _complete_quantile_function(prediction_values, quantile_values)
    integrated = np.zeros_like(target_values)
    for index in range(levels.size - 1):
        integrated += _integrated_pinball(
            target_values,
            float(levels[index]),
            float(levels[index + 1]),
            values[..., index],
            values[..., index + 1],
        )
    return float(np.mean(2.0 * integrated))


def qice_percent(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: Sequence[float],
    bins: int = 10,
) -> float:
    """Return QICE in percentage points over equal-probability quantile intervals."""
    prediction_values, target_values, quantile_values = _validate_inputs(
        prediction, target, quantiles
    )
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer of at least 2")
    levels, values = _complete_quantile_function(prediction_values, quantile_values)
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    interpolated = np.empty((*target_values.shape, boundaries.size), dtype=np.float64)
    for index, boundary in enumerate(boundaries):
        segment = min(np.searchsorted(levels, boundary, side="right") - 1, levels.size - 2)
        weight = (boundary - levels[segment]) / (levels[segment + 1] - levels[segment])
        interpolated[..., index] = (
            values[..., segment]
            + weight * (values[..., segment + 1] - values[..., segment])
        )

    coverages = []
    for index in range(int(bins)):
        lower = interpolated[..., index]
        upper = interpolated[..., index + 1]
        if index == int(bins) - 1:
            inside = (target_values >= lower) & (target_values <= upper)
        else:
            inside = (target_values >= lower) & (target_values < upper)
        coverages.append(float(np.mean(inside)))
    expected = 1.0 / int(bins)
    return float(100.0 * np.mean(np.abs(np.asarray(coverages) - expected)))


def probabilistic_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: Sequence[float],
    qice_bins: int = 10,
) -> dict[str, float]:
    """Compute CRPS and percentage-point QICE from finite quantile forecasts."""
    return {
        "crps": quantile_crps(prediction, target, quantiles),
        "qice": qice_percent(prediction, target, quantiles, bins=qice_bins),
    }


__all__ = ["probabilistic_metrics", "qice_percent", "quantile_crps"]
