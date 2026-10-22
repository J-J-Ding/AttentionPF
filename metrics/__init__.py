"""Point and prediction-interval evaluation metrics."""

from __future__ import annotations

import numpy as np

from metrics.interval_metrics import interval_metrics
from metrics.point_metrics import point_metrics
from metrics.probabilistic_metrics import probabilistic_metrics
from utils.config import central_interval_indices


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    coverage: float | None = None,
    quantiles: tuple[float, ...] | None = None,
    coverage_levels: tuple[float, ...] | None = None,
    primary_coverage: float | None = None,
) -> dict[str, object]:
    """Return original-scale metrics overall and for every forecast horizon."""
    prediction_values = np.asarray(prediction)
    target_values = np.asarray(target)
    if prediction_values.ndim != 4:
        raise ValueError("prediction shape must be [N,P,C,Q]")
    if target_values.shape != prediction_values.shape[:-1]:
        raise ValueError(
            f"target shape {target_values.shape} must equal prediction prefix "
            f"{prediction_values.shape[:-1]}"
        )
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("prediction and target must be finite")
    if quantiles is None:
        if coverage is None:
            raise ValueError("coverage is required when quantiles are omitted")
        tail = (1.0 - float(coverage)) / 2.0
        quantiles = (tail, 0.5, 1.0 - tail)
    quantiles = tuple(float(value) for value in quantiles)
    if prediction_values.shape[-1] != len(quantiles):
        raise ValueError("prediction quantile axis must match configured quantiles")
    if np.any(np.diff(prediction_values, axis=-1) <= 0.0):
        raise ValueError("predicted quantiles must be strictly increasing")
    if coverage_levels is None:
        if coverage is None:
            raise ValueError("coverage_levels or coverage must be provided")
        coverage_levels = (float(coverage),)
        legacy_qice = True
    else:
        coverage_levels = tuple(float(value) for value in coverage_levels)
        legacy_qice = False
    if not coverage_levels:
        raise ValueError("coverage_levels must be non-empty")
    if primary_coverage is None:
        primary_coverage = float(coverage if coverage is not None else coverage_levels[-1])
    primary_coverage = float(primary_coverage)
    median_matches = [
        index for index, value in enumerate(quantiles) if abs(value - 0.5) <= 1.0e-9
    ]
    if len(median_matches) != 1:
        raise ValueError("configured quantiles must contain exactly one median at 0.5")
    median_index = median_matches[0]
    interval_indices = {
        level: central_interval_indices(quantiles, level) for level in coverage_levels
    }
    if not any(abs(primary_coverage - level) <= 1.0e-9 for level in coverage_levels):
        raise ValueError("primary_coverage must be included in coverage_levels")

    target_range = float(np.ptp(target_values))

    def summarize(pred: np.ndarray, truth: np.ndarray) -> dict[str, object]:
        result: dict[str, object] = point_metrics(pred[..., median_index], truth)
        records = []
        for level in coverage_levels:
            lower_index, upper_index = interval_indices[level]
            record: dict[str, object] = {
                "coverage": level,
                "lower_quantile": quantiles[lower_index],
                "upper_quantile": quantiles[upper_index],
            }
            record.update(
                interval_metrics(
                    pred[..., lower_index],
                    pred[..., upper_index],
                    truth,
                    coverage=level,
                    target_range=target_range,
                )
            )
            records.append(record)
        primary = next(
            record for record in records
            if abs(float(record["coverage"]) - primary_coverage) <= 1.0e-9
        )
        for key in ("picp", "mpiw", "pinaw", "coverage_error", "interval_score"):
            result[key] = primary[key]
        result["primary_coverage"] = primary_coverage
        result["coverage_intervals"] = records
        distributional = probabilistic_metrics(pred, truth, quantiles)
        result["crps"] = distributional["crps"]
        result["qice"] = (
            distributional["qice"]
            if legacy_qice
            else float(
                100.0
                * np.mean([float(record["coverage_error"]) for record in records])
            )
        )
        return result

    overall = summarize(prediction_values, target_values)
    by_horizon = []
    for horizon in range(prediction_values.shape[1]):
        item: dict[str, object] = {"horizon": horizon + 1}
        item.update(summarize(prediction_values[:, horizon], target_values[:, horizon]))
        by_horizon.append(item)
    return {"overall": overall, "by_horizon": by_horizon}


__all__ = [
    "compute_metrics",
    "interval_metrics",
    "point_metrics",
    "probabilistic_metrics",
]

