"""Headless, publication-ready figures for interval forecasting experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


_COLORS = {
    "target": "#222222",
    "point": "#1565C0",
    "interval": "#90CAF9",
    "train": "#00897B",
    "validation": "#E64A19",
    "reference": "#6A1B9A",
}

_RC_PARAMS = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}


def _validate_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    channel_index: int,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets)
    if prediction_array.ndim != 4 or prediction_array.shape[-1] != 3:
        raise ValueError("prediction shape must be [N, P, C_out, 3]")
    if target_array.shape != prediction_array.shape[:-1]:
        raise ValueError("target shape must equal prediction shape without its last axis")
    if prediction_array.shape[0] == 0:
        raise ValueError("predictions and targets must contain at least one sample")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("predictions and targets must contain only finite values")
    if channel_index >= prediction_array.shape[2]:
        raise ValueError(
            f"visualization.channel_index={channel_index} is out of range for "
            f"{prediction_array.shape[2]} output channels"
        )
    if sample_index >= prediction_array.shape[0]:
        raise ValueError(
            f"visualization.sample_index={sample_index} is out of range for "
            f"{prediction_array.shape[0]} samples"
        )
    return prediction_array, target_array


def _save_figure(
    figure: plt.Figure,
    figures_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    paths: dict[str, Path],
) -> None:
    try:
        for image_format in formats:
            path = figures_dir / f"{stem}.{image_format}"
            figure.savefig(path, format=image_format, dpi=dpi)
            paths[f"{stem}_{image_format}"] = path
    finally:
        plt.close(figure)


def _training_figure(history: list[dict[str, Any]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    if history:
        epochs = [int(record["epoch"]) for record in history]
        axis.plot(
            epochs,
            [float(record["train_loss"]) for record in history],
            color=_COLORS["train"],
            marker="o",
            linewidth=2,
            markersize=4,
            label="Train loss",
        )
        axis.plot(
            epochs,
            [float(record["val_loss"]) for record in history],
            color=_COLORS["validation"],
            marker="s",
            linewidth=2,
            markersize=4,
            label="Validation loss",
        )
        axis.set_xticks(epochs if len(epochs) <= 12 else epochs[:: max(1, len(epochs) // 10)])
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No new epochs in resumed run", ha="center", va="center")
    axis.set_title("Training convergence")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Objective loss")
    return figure


def _forecast_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    channel_index: int,
    sample_index: int,
    max_points: int,
) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), constrained_layout=True)
    count = min(max_points, predictions.shape[0])
    x_time = np.arange(1, count + 1)
    time_pred = predictions[:count, 0, channel_index]
    time_target = targets[:count, 0, channel_index]
    axes[0].fill_between(
        x_time,
        time_pred[:, 0],
        time_pred[:, 2],
        color=_COLORS["interval"],
        alpha=0.45,
        label="Prediction interval",
    )
    axes[0].plot(x_time, time_target, color=_COLORS["target"], linewidth=1.7, label="Observed")
    axes[0].plot(x_time, time_pred[:, 1], color=_COLORS["point"], linewidth=1.7, label="Point forecast")
    axes[0].set_title("Rolling one-step interval forecasts")
    axes[0].set_xlabel("Test sample")
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=False, ncol=3)

    horizons = np.arange(1, predictions.shape[1] + 1)
    sample_pred = predictions[sample_index, :, channel_index]
    sample_target = targets[sample_index, :, channel_index]
    axes[1].fill_between(
        horizons,
        sample_pred[:, 0],
        sample_pred[:, 2],
        color=_COLORS["interval"],
        alpha=0.45,
        label="Prediction interval",
    )
    axes[1].plot(horizons, sample_target, color=_COLORS["target"], marker="o", linewidth=1.7, label="Observed")
    axes[1].plot(horizons, sample_pred[:, 1], color=_COLORS["point"], marker="s", linewidth=1.7, label="Point forecast")
    axes[1].set_title(f"Forecast profile for test sample {sample_index}")
    axes[1].set_xlabel("Forecast horizon")
    axes[1].set_ylabel("Value")
    axes[1].legend(frameon=False, ncol=3)
    return figure


def _horizon_metrics_figure(
    metrics: dict[str, Any], nominal_coverage: float
) -> plt.Figure:
    records = metrics.get("by_horizon")
    if not isinstance(records, list) or not records:
        raise ValueError("metrics.by_horizon must be a non-empty list")
    horizons = [int(record["horizon"]) for record in records]
    figure, axes = plt.subplots(3, 2, figsize=(10.0, 9.5), constrained_layout=True)
    flat_axes = axes.ravel()
    point_axis = flat_axes[0]
    for key, label, color, marker in (
        ("mae", "MAE", _COLORS["point"], "o"),
        ("rmse", "RMSE", _COLORS["validation"], "s"),
    ):
        try:
            values = [float(record[key]) for record in records]
        except KeyError as error:
            raise ValueError(f"metrics.by_horizon is missing '{key}'") from error
        point_axis.plot(
            horizons,
            values,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.8,
            label=label,
        )
    point_axis.set_title("Point forecast error")
    point_axis.set_xlabel("Forecast horizon")
    point_axis.set_ylabel("Error")
    point_axis.legend(frameon=False)

    names = (
        ("picp", "PICP"),
        ("mpiw", "MPIW"),
        ("pinaw", "PINAW"),
        ("coverage_error", "Coverage error"),
        ("interval_score", "Interval score"),
    )
    for axis, (key, label) in zip(flat_axes[1:], names):
        try:
            values = [float(record[key]) for record in records]
        except KeyError as error:
            raise ValueError(f"metrics.by_horizon is missing '{key}'") from error
        axis.plot(
            horizons,
            values,
            color=_COLORS["point"],
            marker="o",
            markersize=3,
            linewidth=1.8,
        )
        if key == "picp":
            axis.axhline(
                nominal_coverage,
                color=_COLORS["reference"],
                linestyle="--",
                linewidth=1.5,
                label=f"Nominal ({nominal_coverage:.0%})",
            )
            axis.legend(frameon=False)
            axis.set_ylim(0.0, 1.05)
        axis.set_title(label)
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel(label)
    figure.suptitle("Forecast quality by horizon", fontsize=14)
    return figure


def generate_experiment_figures(
    result_dir: str | Path,
    history: list[dict[str, Any]],
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Path]:
    """Generate all configured figures inside one experiment result directory."""
    options = config["visualization"]
    if not options["enabled"]:
        return {}
    channel_index = int(options["channel_index"])
    sample_index = int(options["sample_index"])
    prediction_array, target_array = _validate_arrays(
        predictions, targets, channel_index, sample_index
    )
    figures_dir = Path(result_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    formats = list(options["formats"])
    dpi = int(options["dpi"])
    paths: dict[str, Path] = {}
    with plt.rc_context(_RC_PARAMS):
        _save_figure(
            _training_figure(history), figures_dir, "training_curves", formats, dpi, paths
        )
        _save_figure(
            _forecast_figure(
                prediction_array,
                target_array,
                channel_index,
                sample_index,
                int(options["max_points"]),
            ),
            figures_dir,
            "forecast_interval",
            formats,
            dpi,
            paths,
        )
        _save_figure(
            _horizon_metrics_figure(metrics, float(config["evaluation"]["coverage"])),
            figures_dir,
            "horizon_metrics",
            formats,
            dpi,
            paths,
        )
    return paths


def generate_pretrain_figure(
    result_dir: str | Path,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path | None:
    """Save the Stage-I convergence curve using the shared paper style."""
    options = config["visualization"]
    if not options["enabled"]:
        return None
    figures_dir = Path(result_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    with plt.rc_context(_RC_PARAMS):
        _save_figure(
            _training_figure(history),
            figures_dir,
            "training_curves",
            list(options["formats"]),
            int(options["dpi"]),
            paths,
        )
    return paths[f"training_curves_{options['formats'][0]}"]
