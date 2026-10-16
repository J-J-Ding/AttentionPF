"""Forecasting model definitions and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from models.base_model import BaseForecastModel
from models.v0_transformer import V0Transformer
from models.v1_pretrain_adapter import V1PretrainAdapter


ModelBuilder = Callable[[dict[str, Any], int, int], BaseForecastModel]


def _build_v0(config: dict[str, Any], input_dim: int, output_dim: int) -> BaseForecastModel:
    model = config["model"]
    data = config["data"]
    return V0Transformer(
        input_dim=input_dim, output_dim=output_dim, seq_len=data["seq_len"],
        pred_len=data["pred_len"], d_model=model["d_model"], n_heads=model["n_heads"],
        n_layers=model["n_layers"], d_ff=model["d_ff"], dropout=model["dropout"],
        activation=model["activation"], epsilon=config["interval"]["epsilon"],
    )


def _build_v1(config: dict[str, Any], input_dim: int, output_dim: int) -> BaseForecastModel:
    checkpoint = config["pretrain"]["checkpoint"]
    if not checkpoint:
        raise ValueError("pretrain.checkpoint must explicitly name a Stage-I best checkpoint")
    from pretrain import load_pretrain_checkpoint

    state = load_pretrain_checkpoint(checkpoint, expected_config=config)
    model_config, data, patch, adaptation = (
        config["model"], config["data"], config["patch"], config["adaptation"]
    )
    model = V1PretrainAdapter(
        input_dim=input_dim, output_dim=output_dim, seq_len=data["seq_len"],
        pred_len=data["pred_len"], patch_length=patch["length"],
        patch_stride=patch["stride"], d_model=model_config["d_model"],
        n_heads=model_config["n_heads"], n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"], dropout=model_config["dropout"],
        activation=model_config["activation"], epsilon=config["interval"]["epsilon"],
        adaptation_mode=adaptation["mode"], adapter_rank=adaptation["rank"],
        adapter_alpha=adaptation["alpha"], adapter_dropout=adaptation["dropout"],
    )
    model.load_pretrained_backbone(state["backbone_state_dict"])
    checkpoint_path = Path(checkpoint)
    model.pretrain_provenance = {
        "pretrain_checkpoint": str(checkpoint_path),
        "pretrain_checkpoint_id": checkpoint_path.parent.name,
        "adaptation_mode": adaptation["mode"],
        "adapter_rank": adaptation["rank"],
        "adapter_alpha": float(adaptation["alpha"]),
    }
    model.pretrain_scaler_state = state["scaler_state_dict"]
    return model


MODEL_REGISTRY: dict[str, ModelBuilder] = {"v0": _build_v0, "v1": _build_v1}


def build_model(
    name: str, config: dict[str, Any], input_dim: int, output_dim: int
) -> BaseForecastModel:
    """Construct a registered model from explicit data dimensions and config."""
    normalized = name.lower()
    if normalized not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model '{name}'; available models: {available}")
    return MODEL_REGISTRY[normalized](config, input_dim, output_dim)


__all__ = ["BaseForecastModel", "MODEL_REGISTRY", "V0Transformer", "V1PretrainAdapter", "build_model"]

