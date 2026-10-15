"""Atomic, versioned, and device-portable experiment checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    expected_model_name: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {state.get('format_version')}"
        )
    if expected_model_name is not None and state.get("model_name") != expected_model_name:
        raise ValueError(
            f"Checkpoint model name '{state.get('model_name')}' does not match "
            f"expected model name '{expected_model_name}'"
        )
    try:
        model.load_state_dict(state["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"Checkpoint has incompatible model state: {error}") from error
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if restore_rng and state.get("rng_state") is not None:
        restore_rng_state(state["rng_state"])
    return state

