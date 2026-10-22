"""YAML loading and strict validation for experiment configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_COMMON_SCHEMA: dict[str, set[str]] = {
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
        "visualization",
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
    "visualization": {
        "enabled",
        "formats",
        "dpi",
        "max_points",
        "sample_index",
        "channel_index",
    },
}

V1_QUANTILES = tuple(index / 100.0 for index in range(5, 100, 5))
V1_COVERAGE_LEVELS = tuple(index / 10.0 for index in range(1, 10))

_V0_SCHEMA = {path: set(keys) for path, keys in _COMMON_SCHEMA.items()}
_V1_SCHEMA: dict[str, set[str]] = {
    **{path: set(keys) for path, keys in _COMMON_SCHEMA.items()},
    "": _COMMON_SCHEMA[""] | {"patch", "mask", "pretrain", "adaptation"},
    "interval": {"quantiles", "epsilon"},
    "evaluation": {"coverage_levels", "primary_coverage", "save_predictions"},
    "patch": {"length", "stride"},
    "mask": {"random_ratio", "tail_ratio", "random_probability"},
    "pretrain": {
        "checkpoint",
        "epochs",
        "val_ratio",
        "early_stopping_patience",
        "learning_rate",
        "weight_decay",
    },
    "adaptation": {"mode", "rank", "alpha", "dropout"},
}


def quantile_levels(config: dict[str, Any]) -> tuple[float, ...]:
    """Return the ordered quantile levels for either configuration version."""
    interval = config["interval"]
    if "quantiles" in interval:
        return tuple(float(value) for value in interval["quantiles"])
    return (
        float(interval["lower_quantile"]),
        float(interval["median_quantile"]),
        float(interval["upper_quantile"]),
    )


def coverage_levels(config: dict[str, Any]) -> tuple[float, ...]:
    """Return every central interval coverage requested for evaluation."""
    evaluation = config["evaluation"]
    if "coverage_levels" in evaluation:
        return tuple(float(value) for value in evaluation["coverage_levels"])
    return (float(evaluation["coverage"]),)


def primary_coverage(config: dict[str, Any]) -> float:
    """Return the coverage represented by backward-compatible flat metrics."""
    evaluation = config["evaluation"]
    return float(evaluation.get("primary_coverage", evaluation.get("coverage")))


def central_interval_indices(
    quantiles: tuple[float, ...], coverage: float
) -> tuple[int, int]:
    """Resolve a central coverage to exact lower and upper quantile indices."""
    lower = (1.0 - float(coverage)) / 2.0
    upper = (1.0 + float(coverage)) / 2.0

    def locate(level: float) -> int:
        matches = [
            index for index, value in enumerate(quantiles)
            if abs(value - level) <= 1.0e-9
        ]
        if len(matches) != 1:
            raise ValueError(
                f"coverage {coverage} requires quantile {level}, which is not uniquely configured"
            )
        return matches[0]

    return locate(lower), locate(upper)


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


def _check_unknown_keys(config: dict[str, Any], schema: dict[str, set[str]]) -> None:
    for path, allowed in schema.items():
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
    """Raise ``ValueError`` when a versioned configuration violates its spec."""
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    model = config.get("model")
    name = model.get("name") if isinstance(model, dict) else None
    if name == "v0":
        validate_v0_config(config)
    elif name == "v1":
        validate_v1_config(config)
    else:
        raise ValueError("model.name must be 'v0' or 'v1'")


def _validate_shared_config(config: dict[str, Any]) -> None:
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
    if float(interval["epsilon"]) <= 0.0:
        raise ValueError("interval.epsilon must be positive")
    if float(config["loss"]["interval_weight"]) < 0.0:
        raise ValueError("loss.interval_weight must be non-negative")

    training = config["training"]
    if not isinstance(training["epochs"], int) or training["epochs"] <= 0:
        raise ValueError("training.epochs must be a positive integer")
    if not isinstance(training["early_stopping_patience"], int) or training["early_stopping_patience"] <= 0:
        raise ValueError("training.early_stopping_patience must be a positive integer")
    if training["monitor_mode"] not in {"min", "max"}:
        raise ValueError("training.monitor_mode must be 'min' or 'max'")
    if training["monitor"] != "val_loss":
        raise ValueError("V0 training.monitor must be 'val_loss'")
    if training["monitor_mode"] != "min":
        raise ValueError("V0 val_loss requires training.monitor_mode='min'")

    visualization = config["visualization"]
    if not isinstance(visualization["enabled"], bool):
        raise ValueError("visualization.enabled must be a boolean")
    formats = visualization["formats"]
    if (
        not isinstance(formats, list)
        or not formats
        or len(set(formats)) != len(formats)
        or any(item not in {"png", "pdf"} for item in formats)
    ):
        raise ValueError("visualization.formats must be a non-empty unique list of png/pdf")
    for name in ("dpi", "max_points"):
        if not isinstance(visualization[name], int) or visualization[name] <= 0:
            raise ValueError(f"visualization.{name} must be a positive integer")
    for name in ("sample_index", "channel_index"):
        if not isinstance(visualization[name], int) or visualization[name] < 0:
            raise ValueError(f"visualization.{name} must be a non-negative integer")


def validate_v0_config(config: dict[str, Any]) -> None:
    """Validate the unchanged V0 configuration contract."""
    _check_unknown_keys(config, _V0_SCHEMA)
    _validate_shared_config(config)
    interval = config["interval"]
    lower = float(interval["lower_quantile"])
    median = float(interval["median_quantile"])
    upper = float(interval["upper_quantile"])
    if not (0.0 < lower < median == 0.5 < upper < 1.0):
        raise ValueError("quantiles must satisfy 0 < lower < median == 0.5 < upper < 1")
    if abs(float(config["evaluation"]["coverage"]) - (upper - lower)) > 1.0e-9:
        raise ValueError("evaluation.coverage must equal upper_quantile - lower_quantile")
    if config["model"]["positional_encoding"] != "sinusoidal":
        raise ValueError("V0 positional_encoding must be 'sinusoidal'")


def validate_v1_config(config: dict[str, Any]) -> None:
    """Validate V1 patch pretraining and adaptation settings."""
    _check_unknown_keys(config, _V1_SCHEMA)
    _validate_shared_config(config)
    model = config["model"]
    if model["positional_encoding"] != "learned":
        raise ValueError("V1 positional_encoding must be 'learned'")

    quantiles = config["interval"]["quantiles"]
    if not isinstance(quantiles, list) or len(quantiles) != len(V1_QUANTILES):
        raise ValueError("V1 interval.quantiles must contain the 19 levels from 0.05 to 0.95")
    try:
        quantile_values = tuple(float(value) for value in quantiles)
    except (TypeError, ValueError) as error:
        raise ValueError("V1 interval.quantiles must contain finite numeric values") from error
    if any(not value == value or abs(value) == float("inf") for value in quantile_values):
        raise ValueError("V1 interval.quantiles must contain finite numeric values")
    if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(quantile_values, V1_QUANTILES)):
        raise ValueError("V1 interval.quantiles must equal 0.05 through 0.95 in steps of 0.05")

    levels = config["evaluation"]["coverage_levels"]
    if not isinstance(levels, list) or len(levels) != len(V1_COVERAGE_LEVELS):
        raise ValueError("V1 evaluation.coverage_levels must contain 0.1 through 0.9")
    try:
        coverage_values = tuple(float(value) for value in levels)
    except (TypeError, ValueError) as error:
        raise ValueError("V1 evaluation.coverage_levels must be numeric") from error
    if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(coverage_values, V1_COVERAGE_LEVELS)):
        raise ValueError("V1 evaluation.coverage_levels must equal 0.1 through 0.9")
    primary = float(config["evaluation"]["primary_coverage"])
    if abs(primary - 0.9) > 1.0e-9 or not any(
        abs(primary - value) <= 1.0e-9 for value in coverage_values
    ):
        raise ValueError("V1 evaluation.primary_coverage must be 0.9")
    for coverage in coverage_values:
        central_interval_indices(quantile_values, coverage)

    patch = config["patch"]
    for name in ("length", "stride"):
        if not isinstance(patch[name], int) or patch[name] <= 0:
            raise ValueError(f"patch.{name} must be a positive integer")
    seq_len = config["data"]["seq_len"]
    if patch["length"] > seq_len or (seq_len - patch["length"]) % patch["stride"] != 0:
        raise ValueError("patch length/stride must exactly cover data.seq_len without padding")

    mask = config["mask"]
    for name in ("random_ratio", "tail_ratio"):
        value = mask[name]
        if not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
            raise ValueError(f"mask.{name} must be in (0, 1)")
    probability = mask["random_probability"]
    if not isinstance(probability, (int, float)) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("mask.random_probability must be in [0, 1]")

    pretrain = config["pretrain"]
    checkpoint = pretrain["checkpoint"]
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("pretrain.checkpoint must be null or a path string")
    for name in ("epochs", "early_stopping_patience"):
        if not isinstance(pretrain[name], int) or pretrain[name] <= 0:
            raise ValueError(f"pretrain.{name} must be a positive integer")
    if not isinstance(pretrain["val_ratio"], (int, float)) or not 0.0 < float(pretrain["val_ratio"]) < 1.0:
        raise ValueError("pretrain.val_ratio must be in (0, 1)")
    if float(pretrain["learning_rate"]) <= 0.0:
        raise ValueError("pretrain.learning_rate must be positive")
    if float(pretrain["weight_decay"]) < 0.0:
        raise ValueError("pretrain.weight_decay must be non-negative")

    adaptation = config["adaptation"]
    if adaptation["mode"] not in {"full", "qk", "qkv"}:
        raise ValueError("adaptation.mode must be 'full', 'qk', or 'qkv'")
    rank = adaptation["rank"]
    if not isinstance(rank, int) or not 0 < rank <= model["d_model"]:
        raise ValueError("adaptation.rank must be in [1, model.d_model]")
    if float(adaptation["alpha"]) <= 0.0:
        raise ValueError("adaptation.alpha must be positive")
    if not 0.0 <= float(adaptation["dropout"]) < 1.0:
        raise ValueError("adaptation.dropout must be in [0, 1)")

