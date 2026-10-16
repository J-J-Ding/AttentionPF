from copy import deepcopy

import pytest

from test_config import valid_config
from utils.config import load_config, validate_config


def valid_v1_config():
    config = deepcopy(valid_config())
    config["experiment"]["name"] = "v1_etth1_ms"
    config["model"].update({"name": "v1", "positional_encoding": "learned"})
    config.update(
        {
            "patch": {"length": 16, "stride": 8},
            "mask": {
                "random_ratio": 0.4,
                "tail_ratio": 0.25,
                "random_probability": 0.5,
            },
            "pretrain": {
                "checkpoint": None,
                "epochs": 50,
                "val_ratio": 0.1,
                "early_stopping_patience": 10,
                "learning_rate": 1.0e-3,
                "weight_decay": 1.0e-4,
            },
            "adaptation": {
                "mode": "qk",
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
            },
        }
    )
    return config


def test_v1_yaml_loads_with_strict_sections(project_root):
    config = load_config(project_root / "configs" / "v1.yaml")
    assert config["model"]["name"] == "v1"
    assert config["adaptation"]["mode"] == "qk"
    assert config["pretrain"]["checkpoint"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["patch"].update({"surprise": 1}), "surprise"),
        (lambda c: c["patch"].update({"stride": 7}), "cover"),
        (lambda c: c["mask"].update({"random_ratio": 0.0}), "random_ratio"),
        (lambda c: c["mask"].update({"tail_ratio": 1.0}), "tail_ratio"),
        (lambda c: c["mask"].update({"random_probability": 1.1}), "random_probability"),
        (lambda c: c["adaptation"].update({"rank": 64}), "rank"),
        (lambda c: c["adaptation"].update({"alpha": 0.0}), "alpha"),
        (lambda c: c["adaptation"].update({"mode": "bad"}), "mode"),
        (lambda c: c["adaptation"].update({"dropout": 1.0}), "dropout"),
        (lambda c: c["pretrain"].update({"checkpoint": 7}), "checkpoint"),
        (lambda c: c["pretrain"].update({"val_ratio": 1.0}), "val_ratio"),
    ],
)
def test_invalid_v1_contract_is_rejected(mutate, message):
    config = valid_v1_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_v1_requires_all_version_specific_sections():
    config = valid_v1_config()
    del config["mask"]
    with pytest.raises(ValueError, match="mask"):
        validate_config(config)
