from copy import deepcopy

import pytest

from utils.config import load_config, validate_config


def valid_config():
    return {
        "experiment": {
            "name": "v0_etth1_ms",
            "seed": 2026,
            "device": "cpu",
            "output_root": ".",
            "deterministic": True,
        },
        "data": {
            "path": "datasets/raw/ETTh1.csv",
            "date_col": "date",
            "feature_cols": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "target_cols": ["OT"],
            "features": "MS",
            "seq_len": 96,
            "pred_len": 24,
            "stride": 1,
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "split": {"type": "ratio", "train": 0.7, "val": 0.1, "test": 0.2},
        },
        "model": {
            "name": "v0",
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 64,
            "dropout": 0.1,
            "activation": "gelu",
            "positional_encoding": "sinusoidal",
        },
        "interval": {
            "lower_quantile": 0.05,
            "median_quantile": 0.5,
            "upper_quantile": 0.95,
            "epsilon": 1.0e-6,
        },
        "loss": {"interval_weight": 0.0},
        "optimizer": {"name": "adamw", "lr": 1.0e-3, "weight_decay": 1.0e-4},
        "scheduler": {"name": "none"},
        "training": {
            "epochs": 2,
            "early_stopping_patience": 2,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "grad_clip_norm": 1.0,
            "resume_from": None,
        },
        "evaluation": {"coverage": 0.9, "save_predictions": True},
        "logging": {"level": "INFO"},
        "visualization": {
            "enabled": True,
            "formats": ["png", "pdf"],
            "dpi": 300,
            "max_points": 300,
            "sample_index": 0,
            "channel_index": 0,
        },
    }


def test_v0_yaml_loads_and_validates(project_root):
    config = load_config(project_root / "configs" / "v0.yaml")
    assert config["data"]["features"] == "MS"
    assert config["data"]["feature_cols"][-1] == "OT"
    assert config["interval"]["lower_quantile"] == 0.05


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update({"surprise": 1}), "surprise"),
        (lambda c: c["data"].update({"seq_len": 0}), "seq_len"),
        (lambda c: c["data"]["split"].update({"train": 0.8}), "sum to 1"),
        (lambda c: c["model"].update({"d_model": 30}), "n_heads"),
        (lambda c: c["interval"].update({"lower_quantile": 0.6}), "quantile"),
        (lambda c: c["evaluation"].update({"coverage": 0.8}), "coverage"),
        (lambda c: c["interval"].update({"epsilon": 0.0}), "epsilon"),
        (lambda c: c["loss"].update({"interval_weight": -0.1}), "interval_weight"),
        (lambda c: c["training"].update({"monitor": "nonsense"}), "monitor"),
        (lambda c: c["training"].update({"monitor_mode": "max"}), "monitor_mode"),
    ],
)
def test_invalid_config_is_rejected(mutate, message):
    config = valid_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_ms_target_must_be_the_last_input_feature():
    config = valid_config()
    config["data"]["target_cols"] = ["HUFL"]
    with pytest.raises(ValueError, match="last feature"):
        validate_config(config)


def test_m_targets_must_equal_features_in_order():
    config = valid_config()
    config["data"]["features"] = "M"
    config["data"]["target_cols"] = deepcopy(config["data"]["feature_cols"][:-1])
    with pytest.raises(ValueError, match="target_cols.*feature_cols"):
        validate_config(config)


def test_visualization_config_accepts_paper_formats():
    config = valid_config()
    config["visualization"] = {
        "enabled": True,
        "formats": ["png", "pdf"],
        "dpi": 300,
        "max_points": 300,
        "sample_index": 0,
        "channel_index": 0,
    }

    validate_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("formats", ["svg"], "formats"),
        ("formats", [], "formats"),
        ("dpi", 0, "dpi"),
        ("max_points", 0, "max_points"),
        ("sample_index", -1, "sample_index"),
        ("channel_index", -1, "channel_index"),
    ],
)
def test_visualization_config_rejects_invalid_values(field, value, message):
    config = valid_config()
    config["visualization"] = {
        "enabled": True,
        "formats": ["png", "pdf"],
        "dpi": 300,
        "max_points": 300,
        "sample_index": 0,
        "channel_index": 0,
    }
    config["visualization"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)

