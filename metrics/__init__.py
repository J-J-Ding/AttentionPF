"""Point and prediction-interval evaluation metrics."""

from __future__ import annotations

import numpy as np

from metrics.interval_metrics import interval_metrics
from metrics.point_metrics import point_metrics
from metrics.probabilistic_metrics import probabilistic_metrics


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    coverage: float,
    quantiles: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Return original-scale metrics overall and for every forecast horizon."""
    prediction_values = np.asarray(prediction)
    target_values = np.asarray(target)
    if prediction_values.ndim != 4 or prediction_values.shape[-1] != 3:
        raise ValueError("prediction shape must be [N,P,C,3]")
    if target_values.shape != prediction_values.shape[:-1]:
        raise ValueError(
            f"target shape {target_values.shape} must equal prediction prefix "
            f"{prediction_values.shape[:-1]}"
        )
    if not np.isfinite(prediction_values).all() or not np.isfinite(target_values).all():
        raise ValueError("prediction and target must be finite")
    if quantiles is None:
        tail = (1.0 - float(coverage)) / 2.0
        quantiles = (tail, 0.5, 1.0 - tail)

    target_range = float(np.ptp(target_values))

    def summarize(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
        result: dict[str, float | None] = point_metrics(pred[..., 1], truth)
        result.update(
            interval_metrics(
                pred[..., 0],
                pred[..., 2],
                truth,
                coverage=coverage,
                target_range=target_range,
            )
        )
        result.update(probabilistic_metrics(pred, truth, quantiles))
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

