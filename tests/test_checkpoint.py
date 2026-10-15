from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dataload.scaler import StandardScaler
from utils.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from utils.seed import set_seed


def _checkpoint_path(project_root: Path, name: str) -> Path:
    return project_root / "checkpoints" / name


def _state(model, optimizer, scaler, model_name="v0"):
    return {
        "format_version": 1,
        "model_name": model_name,
        "epoch": 3,
        "global_step": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": scaler.state_dict(),
        "config": {"model": {"name": model_name}},
        "best_metric": 0.25,
        "rng_state": capture_rng_state(),
    }


def test_checkpoint_round_trips_model_scaler_config_and_rng(project_root):
    set_seed(19, deterministic=True)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 6.0]]))
    sample = torch.randn(4, 3)
    expected_output = model(sample).detach().clone()
    path = _checkpoint_path(project_root, "roundtrip.pt")
    state = _state(model, optimizer, scaler)
    save_checkpoint(path, state)
    expected_random = torch.rand(3)

    with torch.no_grad():
        model.weight.add_(100.0)
    set_seed(999, deterministic=True)
    loaded = load_checkpoint(path, model, optimizer=optimizer, map_location="cpu")

    torch.testing.assert_close(model(sample), expected_output)
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    np.testing.assert_allclose(loaded["scaler_state_dict"]["mean"], scaler.mean_)
    assert loaded["config"] == {"model": {"name": "v0"}}
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_incompatible_format_version_is_rejected(project_root):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))
    state = _state(model, optimizer, scaler)
    state["format_version"] = 99
    path = _checkpoint_path(project_root, "bad_version.pt")
    save_checkpoint(path, state)

    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(path, model, map_location="cpu")


def test_incompatible_model_shape_fails_clearly(project_root):
    source = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters())
    scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))
    path = _checkpoint_path(project_root, "bad_shape.pt")
    save_checkpoint(path, _state(source, optimizer, scaler))

    incompatible = nn.Linear(3, 4)
    with pytest.raises(ValueError, match="incompatible model state"):
        load_checkpoint(path, incompatible, map_location="cpu")


def test_checkpoint_model_name_can_be_validated(project_root):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))
    path = _checkpoint_path(project_root, "bad_name.pt")
    save_checkpoint(path, _state(model, optimizer, scaler, model_name="v0"))

    with pytest.raises(ValueError, match="model name"):
        load_checkpoint(path, model, map_location="cpu", expected_model_name="v1")


def test_checkpoint_rejects_same_shape_but_incompatible_contract(project_root):
    model = nn.Linear(3, 6)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))
    state = _state(model, optimizer, scaler)
    state["config"] = {
        "model": {"name": "v0", "d_model": 8},
        "data": {
            "path": "datasets/raw/ETTh1.csv",
            "feature_cols": ["A", "OT"],
            "target_cols": ["OT"],
            "features": "MS",
            "seq_len": 24,
            "pred_len": 24,
            "split": {"type": "ratio", "train": 0.7, "val": 0.1, "test": 0.2},
        },
        "interval": {"lower_quantile": 0.05, "median_quantile": 0.5, "upper_quantile": 0.95, "epsilon": 1e-6},
    }
    path = _checkpoint_path(project_root, "bad_contract.pt")
    save_checkpoint(path, state)
    expected = deepcopy(state["config"])
    expected["data"]["pred_len"] = 12
    expected["data"]["features"] = "M"

    with pytest.raises(ValueError, match="checkpoint contract"):
        load_checkpoint(path, model, map_location="cpu", expected_config=expected)
