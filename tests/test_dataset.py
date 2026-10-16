from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import SequentialSampler

from dataload.dataloader import create_dataloaders
from dataload.dataset import TimeSeriesWindowDataset, load_time_series
from utils.config import load_config


FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _config(project_root, mode="MS"):
    config = load_config(project_root / "configs" / "v0.yaml")
    config = deepcopy(config)
    config["experiment"]["device"] = "cpu"
    config["data"]["batch_size"] = 8
    config["data"]["num_workers"] = 0
    config["data"]["pred_len"] = 24
    config["data"]["features"] = mode
    config["data"]["target_cols"] = FEATURES if mode == "M" else ["OT"]
    return config


def test_etth1_schema_is_loaded_in_time_order_without_date_feature(etth1_path):
    loaded = load_time_series(etth1_path, "date", FEATURES)

    assert loaded.values.shape == (17420, 7)
    assert np.isfinite(loaded.values).all()
    assert loaded.timestamps.is_monotonic_increasing
    assert loaded.timestamps.is_unique
    assert loaded.feature_cols == tuple(FEATURES)


@pytest.mark.parametrize(
    ("mode", "expected_target_channels"), [("M", 7), ("MS", 1)]
)
def test_real_etth1_batch_shapes(project_root, mode, expected_target_channels):
    bundle = create_dataloaders(_config(project_root, mode))
    x, y = next(iter(bundle.train_loader))

    assert x.shape == (8, 96, 7)
    assert y.shape == (8, 24, expected_target_channels)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert bundle.input_dim == 7
    assert bundle.output_dim == expected_target_channels


def test_ms_target_is_the_scaled_future_last_column(project_root):
    bundle = create_dataloaders(_config(project_root, "MS"))
    dataset = bundle.train_loader.dataset
    forecast_start = dataset.forecast_starts[0]
    _, y = dataset[0]

    expected = dataset.values[
        forecast_start : forecast_start + dataset.pred_len, -1:
    ]
    np.testing.assert_allclose(y.numpy(), expected)
    assert y.shape == (24, 1)


def test_split_targets_do_not_leak_and_window_counts_are_exact(project_root):
    config = _config(project_root, "MS")
    bundle = create_dataloaders(config)
    n_rows = 17420
    train_end = int(n_rows * 0.7)
    val_end = train_end + int(n_rows * 0.1)
    seq_len = 96
    pred_len = 24

    train_ds = bundle.train_loader.dataset
    val_ds = bundle.val_loader.dataset
    test_ds = bundle.test_loader.dataset

    assert len(train_ds) == train_end - seq_len - pred_len + 1
    assert len(val_ds) == val_end - train_end - pred_len + 1
    assert len(test_ds) == n_rows - val_end - pred_len + 1
    assert train_ds.forecast_starts[0] == seq_len
    assert train_ds.forecast_starts[-1] + pred_len == train_end
    assert val_ds.forecast_starts[0] == train_end
    assert val_ds.forecast_starts[-1] + pred_len == val_end
    assert test_ds.forecast_starts[0] == val_end
    assert test_ds.forecast_starts[-1] + pred_len == n_rows
    assert val_ds.forecast_starts[0] - seq_len < train_end
    assert isinstance(bundle.val_loader.sampler, SequentialSampler)
    assert isinstance(bundle.test_loader.sampler, SequentialSampler)


def test_scaler_statistics_use_only_train_rows(project_root, etth1_path):
    config = _config(project_root, "MS")
    bundle = create_dataloaders(config)
    raw = pd.read_csv(etth1_path, usecols=FEATURES).to_numpy(dtype=np.float64)
    train_end = int(len(raw) * 0.7)

    np.testing.assert_allclose(bundle.scaler.mean_, raw[:train_end].mean(axis=0))
    assert not np.allclose(bundle.scaler.mean_, raw.mean(axis=0), atol=1e-3)


def test_too_short_split_fails_with_split_name(etth1_path):
    loaded = load_time_series(etth1_path, "date", FEATURES)
    with pytest.raises(ValueError, match="validation.*120"):
        TimeSeriesWindowDataset(
            values=loaded.values[:200],
            target_start=120,
            target_end=200,
            seq_len=96,
            pred_len=24,
            stride=1,
            output_indices=[-1],
            split_name="validation",
        )


def test_duplicate_timestamps_are_rejected(project_root):
    invalid_path = project_root / "tests/fixtures/etth1_duplicate_timestamp.csv"
    with pytest.raises(ValueError, match="duplicate timestamp"):
        load_time_series(invalid_path, "date", FEATURES)


def test_missing_and_nonfinite_features_are_rejected(project_root):
    missing_path = project_root / "tests/fixtures/etth1_missing_ot.csv"
    with pytest.raises(ValueError, match="missing column.*OT"):
        load_time_series(missing_path, "date", FEATURES)

    nonfinite_path = project_root / "tests/fixtures/etth1_nonfinite.csv"
    with pytest.raises(ValueError, match="NaN or infinity"):
        load_time_series(nonfinite_path, "date", FEATURES)
