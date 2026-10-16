"""V1 Stage-I masked temporal pretraining command and reusable engine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from dataload import create_dataloaders
from dataload.pretrain import choose_mask, create_pretrain_loaders
from losses.reconstruction_loss import masked_reconstruction_loss
from models.v1_pretrain_adapter import V1MaskedPretrainer, V1PatchBackbone
from utils.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from utils.config import load_config, validate_config
from utils.logger import append_jsonl, create_experiment_paths, setup_logger
from utils.seed import set_seed
from utils.visualization import generate_pretrain_figure


PRETRAIN_CONTRACT_PATHS = (
    "data.feature_cols",
    "data.seq_len",
    "model.d_model",
    "model.n_heads",
    "model.n_layers",
    "model.d_ff",
    "model.activation",
    "patch.length",
    "patch.stride",
)


def _nested(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_pretrain_checkpoint(
    path: str | Path,
    *,
    expected_config: dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if state.get("format_version") != 1 or state.get("stage") != "pretrain":
        raise ValueError("Unsupported or non-pretrain checkpoint")
    saved_config = state.get("config", {})
    mismatches = [
        field
        for field in PRETRAIN_CONTRACT_PATHS
        if _nested(saved_config, field) != _nested(expected_config, field)
    ]
    if mismatches:
        raise ValueError("Incompatible pretrain checkpoint contract for: " + ", ".join(mismatches))
    required = {
        "model_state_dict",
        "backbone_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "mask_generator_state",
        "train_loader_generator_state",
    }
    missing = required - set(state)
    if missing:
        raise ValueError("Pretrain checkpoint missing fields: " + ", ".join(sorted(missing)))
    return state


def _build_model(config: dict[str, Any], input_dim: int) -> V1MaskedPretrainer:
    model = config["model"]
    patch = config["patch"]
    return V1MaskedPretrainer(
        V1PatchBackbone(
            input_dim=input_dim,
            seq_len=config["data"]["seq_len"],
            patch_length=patch["length"],
            patch_stride=patch["stride"],
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            n_layers=model["n_layers"],
            d_ff=model["d_ff"],
            dropout=model["dropout"],
            activation=model["activation"],
        )
    )


def _batch_mask(
    model: V1MaskedPretrainer,
    batch_size: int,
    config: dict[str, Any],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    options = config["mask"]
    return choose_mask(
        batch_size=batch_size,
        n_patches=model.backbone.n_patches,
        random_ratio=options["random_ratio"],
        tail_ratio=options["tail_ratio"],
        random_probability=options["random_probability"],
        generator=generator,
        device=device,
    )[0]


def train_pretrain_epoch(
    model: V1MaskedPretrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    mask_generator: torch.Generator,
) -> dict[str, float | int]:
    model.train()
    weighted_loss = 0.0
    masked_elements = 0
    for x, _ in loader:
        x = x.to(device)
        mask = _batch_mask(model, x.shape[0], config, mask_generator, device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, target = model(x, mask)
        loss = masked_reconstruction_loss(reconstructed, target, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("pretraining produced a non-finite loss")
        loss.backward()
        optimizer.step()
        elements = int(mask.sum()) * target.shape[-1]
        weighted_loss += float(loss.detach()) * elements
        masked_elements += elements
    if masked_elements == 0:
        raise ValueError("pretrain loader produced no masked elements")
    return {"loss": weighted_loss / masked_elements, "masked_elements": masked_elements}


def validate_pretrain(
    model: V1MaskedPretrainer,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float | int]:
    model.eval()
    weighted_loss = 0.0
    masked_elements = 0
    seed = int(config["experiment"]["seed"]) + 100_000
    with torch.inference_mode():
        for batch_index, (x, _) in enumerate(loader):
            x = x.to(device)
            generator = torch.Generator().manual_seed(seed + batch_index)
            mask = _batch_mask(model, x.shape[0], config, generator, device)
            reconstructed, target = model(x, mask)
            loss = masked_reconstruction_loss(reconstructed, target, mask)
            elements = int(mask.sum()) * target.shape[-1]
            weighted_loss += float(loss) * elements
            masked_elements += elements
    if masked_elements == 0:
        raise ValueError("validation loader produced no masked elements")
    return {"loss": weighted_loss / masked_elements, "masked_elements": masked_elements}


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state(
    model: V1MaskedPretrainer,
    optimizer: torch.optim.Optimizer,
    bundle: Any,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    patience: int,
    best_model_state_dict: dict[str, torch.Tensor],
    train_loader: DataLoader,
    mask_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "stage": "pretrain",
        "model_name": "v1",
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "backbone_state_dict": model.backbone_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": bundle.scaler.state_dict(),
        "config": config,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "patience": patience,
        "best_model_state_dict": best_model_state_dict,
        "rng_state": capture_rng_state(),
        "mask_generator_state": mask_generator.get_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
    }


def run_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    """Run leakage-safe Stage-I pretraining and save a reproducible checkpoint."""
    validate_config(config)
    if config["model"]["name"] != "v1":
        raise ValueError("Stage I requires model.name='v1'")
    seed = int(config["experiment"]["seed"])
    set_seed(seed, bool(config["experiment"]["deterministic"]))
    requested = str(config["experiment"]["device"]).lower()
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    bundle = create_dataloaders(config)
    train_loader, val_loader = create_pretrain_loaders(bundle, config)
    model = _build_model(config, bundle.input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["pretrain"]["learning_rate"]),
        weight_decay=float(config["pretrain"]["weight_decay"]),
    )
    mask_generator = torch.Generator().manual_seed(seed + 1)
    paths = create_experiment_paths(config)
    logger = setup_logger(paths, config["logging"]["level"])
    with (paths.result_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    best_epoch = 0
    patience = 0
    best_model_state = _clone_state(model)
    resume_from = config["training"]["resume_from"]
    if resume_from:
        resumed = load_pretrain_checkpoint(resume_from, expected_config=config, map_location=device)
        try:
            model.load_state_dict(resumed["model_state_dict"], strict=True)
        except RuntimeError as error:
            raise ValueError(f"Incompatible pretrain model state: {error}") from error
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        restore_rng_state(resumed["rng_state"])
        mask_generator.set_state(resumed["mask_generator_state"].cpu())
        train_loader.generator.set_state(resumed["train_loader_generator_state"].cpu())
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["global_step"])
        best_metric = float(resumed["best_metric"])
        best_epoch = int(resumed["best_epoch"])
        patience = int(resumed["patience"])
        best_model_state = resumed["best_model_state_dict"]
        resumed_best = dict(resumed)
        resumed_best["model_state_dict"] = best_model_state
        save_checkpoint(paths.checkpoint_dir / "best.pt", resumed_best)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["pretrain"]["epochs"]) + 1):
        train_result = train_pretrain_epoch(model, train_loader, optimizer, device, config, mask_generator)
        val_result = validate_pretrain(model, val_loader, device, config)
        global_step += len(train_loader)
        val_loss = float(val_result["loss"])
        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            patience = 0
            best_model_state = _clone_state(model)
            save_checkpoint(
                paths.checkpoint_dir / "best.pt",
                _state(model, optimizer, bundle, config, epoch, global_step, best_metric, best_epoch, patience, best_model_state, train_loader, mask_generator),
            )
        else:
            patience += 1
        save_checkpoint(
            paths.checkpoint_dir / "last.pt",
            _state(model, optimizer, bundle, config, epoch, global_step, best_metric, best_epoch, patience, best_model_state, train_loader, mask_generator),
        )
        record = {"epoch": epoch, "train_loss": float(train_result["loss"]), "val_loss": val_loss}
        history.append(record)
        append_jsonl(paths.metrics_path, record)
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, record["train_loss"], val_loss)
        if patience >= int(config["pretrain"]["early_stopping_patience"]):
            break
    if best_epoch == 0:
        raise RuntimeError("Pretraining completed without a best checkpoint")
    serializable = {
        "experiment_id": paths.experiment_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_metric,
        "train_seconds": time.perf_counter() - started,
        "history": history,
    }
    metrics_path = paths.result_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, allow_nan=False)
    curve = generate_pretrain_figure(paths.result_dir, history, config)
    return {
        **serializable,
        "paths": {
            "best_checkpoint": paths.checkpoint_dir / "best.pt",
            "last_checkpoint": paths.checkpoint_dir / "last.pt",
            "resolved_config": paths.result_dir / "config.resolved.yaml",
            "metrics_json": metrics_path,
            "metrics_jsonl": paths.metrics_path,
            "log": paths.log_path,
            "training_curve": curve,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 masked temporal pretraining")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_pretraining(load_config(args.config))
    print(result["paths"]["best_checkpoint"])


if __name__ == "__main__":
    main()
