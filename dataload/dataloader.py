"""Construction of reproducible ETTh1 DataLoaders."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataload.dataset import TimeSeriesWindowDataset, load_time_series
from dataload.scaler import StandardScaler
from utils.config import validate_config


@dataclass(frozen=True)
class SplitMetadata:
    n_rows: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    input_dim: int
    output_dim: int
    split_metadata: SplitMetadata


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(config: dict[str, Any]) -> DataBundle:
    """Create chronological train/validation/test loaders from validated config."""
    validate_config(config)
    data = config["data"]
    loaded = load_time_series(
        Path(data["path"]), data["date_col"], data["feature_cols"]
    )
    n_rows = len(loaded.values)
    train_end = int(n_rows * float(data["split"]["train"]))
    val_end = train_end + int(n_rows * float(data["split"]["val"]))
    metadata = SplitMetadata(
        n_rows=n_rows,
        train_start=0,
        train_end=train_end,
        val_start=train_end,
        val_end=val_end,
        test_start=val_end,
        test_end=n_rows,
    )

    scaler = StandardScaler().fit(loaded.values[:train_end])
    scaled_values = scaler.transform(loaded.values)
    output_indices = (
        list(range(len(data["feature_cols"])))
        if data["features"] == "M"
        else [-1]
    )
    common = {
        "values": scaled_values,
        "seq_len": data["seq_len"],
        "pred_len": data["pred_len"],
        "stride": data["stride"],
        "output_indices": output_indices,
    }
    train_dataset = TimeSeriesWindowDataset(
        target_start=0,
        target_end=train_end,
        split_name="train",
        **common,
    )
    val_dataset = TimeSeriesWindowDataset(
        target_start=train_end,
        target_end=val_end,
        split_name="validation",
        **common,
    )
    test_dataset = TimeSeriesWindowDataset(
        target_start=val_end,
        target_end=n_rows,
        split_name="test",
        **common,
    )

    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    loader_kwargs = {
        "batch_size": data["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": data["pin_memory"],
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        input_dim=len(data["feature_cols"]),
        output_dim=len(output_indices),
        split_metadata=metadata,
    )

