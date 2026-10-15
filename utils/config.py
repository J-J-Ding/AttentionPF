"""YAML loading and strict validation for experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_SCHEMA: dict[str, set[str]] = {
    "": {
        "experiment",
        "data",
        "model",
        "interval",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "evaluation",
        "logging",
    },
    "experiment": {"name", "seed", "device", "output_root", "deterministic"},
    "data": {
        "path",
        "date_col",
        "feature_cols",
        "target_cols",
        "features",
        "seq_len",
        "pred_len",
        "stride",
        "batch_size",
        "num_workers",
        "pin_memory",
        "split",
    },
    "data.split": {"type", "train", "val", "test"},
    "model": {
        "name",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "dropout",
        "activation",
        "positional_encoding",
    },
    "interval": {"lower_quantile", "median_quantile", "upper_quantile", "epsilon"},
    "loss": {"interval_weight"},
    "optimizer": {"name", "lr", "weight_decay"},
    "scheduler": {"name"},
    "training": {
        "epochs",
        "early_stopping_patience",
        "monitor",
        "monitor_mode",
        "grad_clip_norm",
        "resume_from",
    },
    "evaluation": {"coverage", "save_predictions"},
    "logging": {"level"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and validate its complete V0 contract."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    return config


def _check_unknown_keys(config: dict[str, Any]) -> None:
    for path, allowed in _SCHEMA.items():
        node: Any = config
        if path:
            for part in path.split("."):
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f"Missing configuration section: {path}")
                node = node[part]
        if not isinstance(node, dict):
            raise ValueError(f"Configuration section '{path or 'root'}' must be a mapping")
        unknown = set(node) - allowed
        if unknown:
            prefix = f"{path}." if path else ""
            names = ", ".join(f"{prefix}{key}" for key in sorted(unknown))
            raise ValueError(f"Unknown configuration key(s): {names}")
        missing = allowed - set(node)
        if missing:
            prefix = f"{path}." if path else ""
            names = ", ".join(f"{prefix}{key}" for key in sorted(missing))
            raise ValueError(f"Missing configuration key(s): {names}")


def validate_config(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a V0 configuration violates the spec."""
    _check_unknown_keys(config)
    data = config["data"]
    model = config["model"]
    interval = config["interval"]
    evaluation = config["evaluation"]

    for name in ("seq_len", "pred_len", "stride", "batch_size"):
        if not isinstance(data[name], int) or data[name] <= 0:
            raise ValueError(f"data.{name} must be a positive integer")
    if not isinstance(data["num_workers"], int) or data["num_workers"] < 0:
        raise ValueError("data.num_workers must be a non-negative integer")

    split = data["split"]
    if split["type"] != "ratio":
        raise ValueError("data.split.type must be 'ratio' in V0")
    ratios = [split[name] for name in ("train", "val", "test")]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in ratios):
        raise ValueError("split ratios must be positive numbers")
    if abs(sum(ratios) - 1.0) > 1.0e-9:
        raise ValueError("split ratios must sum to 1")

    features = data["feature_cols"]
    targets = data["target_cols"]
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("data.feature_cols must be a non-empty list of unique names")
    if not isinstance(targets, list) or not targets:
        raise ValueError("data.target_cols must be a non-empty list")
    mode = data["features"]
    if mode == "MS":
        if targets != [features[-1]]:
            raise ValueError("MS target_cols must contain exactly the last feature")
    elif mode == "M":
        if targets != features:
            raise ValueError("M target_cols must equal feature_cols in the same order")
    else:
        raise ValueError("data.features must be 'M' or 'MS'")

    for name in ("d_model", "n_heads", "n_layers", "d_ff"):
        if not isinstance(model[name], int) or model[name] <= 0:
            raise ValueError(f"model.{name} must be a positive integer")
    if model["d_model"] % model["n_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if model["activation"] not in {"relu", "gelu"}:
        raise ValueError("model.activation must be 'relu' or 'gelu'")
    if model["positional_encoding"] != "sinusoidal":
        raise ValueError("V0 positional_encoding must be 'sinusoidal'")

    lower = float(interval["lower_quantile"])
    median = float(interval["median_quantile"])
    upper = float(interval["upper_quantile"])
    if not (0.0 < lower < median == 0.5 < upper < 1.0):
        raise ValueError("quantiles must satisfy 0 < lower < median == 0.5 < upper < 1")
    if float(interval["epsilon"]) <= 0.0:
        raise ValueError("interval.epsilon must be positive")
    if abs(float(evaluation["coverage"]) - (upper - lower)) > 1.0e-9:
        raise ValueError("evaluation.coverage must equal upper_quantile - lower_quantile")

    training = config["training"]
    if not isinstance(training["epochs"], int) or training["epochs"] <= 0:
        raise ValueError("training.epochs must be a positive integer")
    if not isinstance(training["early_stopping_patience"], int) or training["early_stopping_patience"] <= 0:
        raise ValueError("training.early_stopping_patience must be a positive integer")
    if training["monitor_mode"] not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be 'min' or 'max'")

