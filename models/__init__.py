"""Forecasting model definitions and registry."""

from __future__ import annotations

from typing import Any

from models.base_model import BaseForecastModel
from models.v0_transformer import V0Transformer


MODEL_REGISTRY: dict[str, type[BaseForecastModel]] = {
    "v0": V0Transformer,
}


def build_model(
    name: str, config: dict[str, Any], input_dim: int, output_dim: int
) -> BaseForecastModel:
    """Construct a registered model from explicit data dimensions and config."""
    normalized = name.lower()
    if normalized not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model '{name}'; available models: {available}")
    model_config = config["model"]
    data_config = config["data"]
    return MODEL_REGISTRY[normalized](
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=data_config["seq_len"],
        pred_len=data_config["pred_len"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"],
        dropout=model_config["dropout"],
        activation=model_config["activation"],
        epsilon=config["interval"]["epsilon"],
    )


__all__ = ["BaseForecastModel", "MODEL_REGISTRY", "V0Transformer", "build_model"]

