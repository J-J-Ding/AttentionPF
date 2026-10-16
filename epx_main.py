"""Version-independent training, evaluation, prediction, and experiment lifecycle."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dataload.scaler import StandardScaler
from losses.interval_loss import QuantileIntervalLoss
from metrics import compute_metrics
from utils.checkpoint import capture_rng_state, load_checkpoint, save_checkpoint
from utils.logger import ExperimentPaths, append_jsonl, create_experiment_paths, setup_logger
from utils.seed import set_seed
from utils.visualization import generate_experiment_figures


def _validate_prediction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    expected = (*target.shape, 3)
    if tuple(prediction.shape) != expected:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match expected {expected}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("model produced a non-finite prediction")
    if not torch.all(prediction[..., 0] < prediction[..., 1]):
        raise ValueError("model violated Lower < Point")
    if not torch.all(prediction[..., 1] < prediction[..., 2]):
        raise ValueError("model violated Point < Upper")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_elements = 0
    for x, target in loader:
        x = x.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        _validate_prediction(prediction, target)
        loss = loss_fn(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("training produced a non-finite loss")
        loss.backward()
        grad_clip = config["training"]["grad_clip_norm"]
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        elements = target.numel()
        total_loss += float(loss.detach()) * elements
        total_elements += elements
    if total_elements == 0:
        raise ValueError("training loader produced no target elements")
    return {"loss": total_loss / total_elements, "elements": total_elements}


def _target_channel_indices(config: dict[str, Any]) -> list[int]:
    if config["data"]["features"] == "MS":
        return [-1]
    return list(range(len(config["data"]["feature_cols"])))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    config: dict[str, Any],
    *, collect_attention: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_elements = 0
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    attention_sum = None
    attention_samples = 0
    attention_fn = getattr(model, 'forward_with_attention', None) if collect_attention else None
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(device)
            target = target.to(device)
            if callable(attention_fn):
                prediction, weights = attention_fn(x)
                if (weights.ndim != 3 or weights.shape[:2] != target.shape[:2]
                        or weights.shape[2] == 0):
                    raise ValueError('attention must have shape [B,P,N]')
                if not torch.isfinite(weights).all():
                    raise FloatingPointError('non-finite attention')
                if (weights < 0).any() or not torch.allclose(
                    weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-5):
                    raise ValueError('attention must be nonnegative and normalized along N')
                batch_sum = weights.double().sum(0).cpu().numpy()
                attention_sum = batch_sum if attention_sum is None else attention_sum + batch_sum
                attention_samples += x.shape[0]
            else:
                prediction = model(x)
            _validate_prediction(prediction, target)
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("evaluation produced a non-finite loss")
            elements = target.numel()
            total_loss += float(loss) * elements
            total_elements += elements
            prediction_batches.append(prediction.cpu().numpy())
            target_batches.append(target.cpu().numpy())
    if total_elements == 0:
        raise ValueError("evaluation loader produced no target elements")
    scaled_predictions = np.concatenate(prediction_batches, axis=0)
    scaled_targets = np.concatenate(target_batches, axis=0)
    channels = _target_channel_indices(config)
    predictions = scaler.inverse_transform(scaled_predictions, channels)
    targets = scaler.inverse_transform(scaled_targets, channels)
    metrics = compute_metrics(predictions, targets, config["evaluation"]["coverage"])
    result = {
        "loss": total_loss / total_elements,
        "elements": total_elements,
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
    }
    if attention_sum is not None:
        result['attention_map'] = attention_sum / attention_samples
    return result


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    return _evaluate(model, loader, loss_fn, scaler, device, config)


def test(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    return _evaluate(model, loader, loss_fn, scaler, device, config, collect_attention=True)


def predict(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    config: dict[str, Any],
) -> np.ndarray:
    result = _evaluate(model, loader, build_loss(config), scaler, device, config)
    return result["predictions"]


def build_loss(config: dict[str, Any]) -> nn.Module:
    quantiles = (
        config["interval"]["lower_quantile"],
        config["interval"]["median_quantile"],
        config["interval"]["upper_quantile"],
    )
    return QuantileIntervalLoss(
        quantiles,
        coverage=float(config["evaluation"]["coverage"]),
        interval_weight=float(config["loss"]["interval_weight"]),
    )


def _resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["experiment"]["device"]).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    scaler: StandardScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    patience: int,
    best_model_state_dict: dict[str, torch.Tensor],
    train_loader: DataLoader,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": config["model"]["name"],
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "patience": patience,
        "best_model_state_dict": best_model_state_dict,
        "train_loader_generator_state": (
            train_loader.generator.get_state()
            if train_loader.generator is not None
            else None
        ),
        "rng_state": capture_rng_state(),
    }


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _scaler_states_match(current: dict[str, Any], saved: dict[str, Any]) -> bool:
    return (
        np.allclose(current["mean"], saved["mean"], rtol=0.0, atol=0.0)
        and np.allclose(current["scale"], saved["scale"], rtol=0.0, atol=0.0)
    )


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def run_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict[str, Any],
    scaler: StandardScaler,
) -> dict[str, Any]:
    """Train, select the best validation checkpoint, test once, and save artifacts."""
    seed = int(config["experiment"]["seed"])
    set_seed(seed, bool(config["experiment"]["deterministic"]))
    device = _resolve_device(config)
    model.to(device)
    paths: ExperimentPaths = create_experiment_paths(config)
    logger = setup_logger(paths, config["logging"]["level"])
    resolved_config = paths.result_dir / "config.resolved.yaml"
    with resolved_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    optimizer_name = config["optimizer"]["name"].lower()
    if optimizer_name != "adamw":
        raise ValueError("V0 currently supports optimizer.name='adamw' only")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    if config["scheduler"]["name"].lower() != "none":
        raise ValueError("V0 currently supports scheduler.name='none' only")
    loss_fn = build_loss(config).to(device)
    best_path = paths.checkpoint_dir / "best.pt"
    last_path = paths.checkpoint_dir / "last.pt"
    best_metric = float("inf") if config["training"]["monitor_mode"] == "min" else -float("inf")
    best_epoch = 0
    patience = 0
    global_step = 0
    start_epoch = 1
    best_model_state_dict = _clone_state_dict(model)
    resume_from = config["training"]["resume_from"]
    if resume_from:
        resumed = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            map_location=device,
            expected_model_name=config["model"]["name"],
            expected_config=config,
        )
        if not _scaler_states_match(scaler.state_dict(), resumed["scaler_state_dict"]):
            raise ValueError("Checkpoint scaler does not match data fitted from current config")
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed["global_step"])
        best_metric = float(resumed["best_metric"])
        best_epoch = int(resumed.get("best_epoch", resumed["epoch"]))
        patience = int(resumed.get("patience", 0))
        best_model_state_dict = resumed.get(
            "best_model_state_dict", _clone_state_dict(model)
        )
        generator_state = resumed.get("train_loader_generator_state")
        if generator_state is not None:
            if train_loader.generator is None:
                raise ValueError("Checkpoint has sampler RNG state but train loader has no generator")
            train_loader.generator.set_state(generator_state.cpu())
        resumed_best = dict(resumed)
        resumed_best["model_state_dict"] = best_model_state_dict
        resumed_best["epoch"] = best_epoch
        resumed_best["config"] = config
        save_checkpoint(best_path, resumed_best)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    summary_fn = getattr(model, "parameter_summary", None)
    parameter_summary = (
        summary_fn()
        if callable(summary_fn)
        else {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params else 0.0,
        }
    )
    provenance = dict(getattr(model, "pretrain_provenance", {}))
    logger.info(
        "experiment=%s device=%s seed=%d total_params=%d trainable_params=%d",
        paths.experiment_id,
        device,
        seed,
        total_params,
        trainable_params,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()
    history = []
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_result = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, config
        )
        global_step += len(train_loader)
        val_result = validate(model, val_loader, loss_fn, scaler, device, config)
        val_loss = float(val_result["loss"])
        improved = val_loss < best_metric if config["training"]["monitor_mode"] == "min" else val_loss > best_metric
        if improved:
            best_metric = val_loss
            best_epoch = epoch
            patience = 0
            best_model_state_dict = _clone_state_dict(model)
            save_checkpoint(
                best_path,
                _checkpoint_state(
                    model, optimizer, config, scaler, epoch, global_step, best_metric,
                    best_epoch, patience, best_model_state_dict, train_loader
                ),
            )
        else:
            patience += 1
        save_checkpoint(
            last_path,
            _checkpoint_state(
                model, optimizer, config, scaler, epoch, global_step, best_metric,
                best_epoch, patience, best_model_state_dict, train_loader
            ),
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_result["loss"]),
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "val_metrics": val_result["metrics"]["overall"],
        }
        history.append(epoch_record)
        append_jsonl(paths.metrics_path, epoch_record)
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f",
            epoch,
            train_result["loss"],
            val_loss,
        )
        if patience >= int(config["training"]["early_stopping_patience"]):
            logger.info("early_stopping epoch=%d", epoch)
            break
    train_seconds = time.perf_counter() - train_started
    if best_epoch == 0:
        raise RuntimeError("Training completed without producing a best checkpoint")

    load_checkpoint(
        best_path,
        model,
        optimizer=None,
        map_location=device,
        expected_model_name=config["model"]["name"],
        expected_config=config,
        restore_rng=False,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    test_result = test(model, test_loader, loss_fn, scaler, device, config)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    n_predictions = len(test_result["predictions"])
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    predictions_path = paths.result_dir / "predictions.npz"
    if config["evaluation"]["save_predictions"]:
        np.savez_compressed(
            predictions_path,
            predictions=test_result["predictions"],
            targets=test_result["targets"],
        )
    metrics_path = paths.result_dir / "metrics.json"
    diagnostic_paths = {}
    if 'attention_map' in test_result:
        attention_path = paths.result_dir / 'attention_map.npy'
        np.save(attention_path, test_result['attention_map'])
        diagnostic_paths['attention_map'] = attention_path
    cache_benchmark = None
    benchmark_fn = getattr(model, 'benchmark_kv_cache', None)
    if callable(benchmark_fn) and getattr(model, 'cache_kv', False):
        x, _ = next(iter(test_loader))
        chunks = list(torch.arange(config['data']['pred_len'], device=device).split(24))
        cache_benchmark = benchmark_fn(x.to(device), chunks, repeats=5)
    serializable = {
        "experiment_id": paths.experiment_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_metric,
        "test_loss": float(test_result["loss"]),
        "test_metrics": test_result["metrics"],
        "model_name": config["model"]["name"],
        **parameter_summary,
        **provenance,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_latency_seconds": inference_seconds / max(n_predictions, 1),
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "history": history,
    }
    if callable(benchmark_fn):
        serializable['kv_cache_benchmark'] = cache_benchmark
    _json_dump(metrics_path, serializable)
    figure_paths = generate_experiment_figures(
        paths.result_dir,
        history,
        test_result["predictions"],
        test_result["targets"],
        test_result["metrics"],
        config,
    )
    logger.info("test_loss=%.6f results=%s", test_result["loss"], paths.result_dir)
    return {
        **serializable,
        "test": test_result,
        "paths": {
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "metrics_json": metrics_path,
            "predictions_npz": predictions_path,
            "resolved_config": resolved_config,
            "log": paths.log_path,
            "metrics_jsonl": paths.metrics_path,
            "report": paths.report_path,
            "figures_dir": paths.result_dir / "figures",
            "figures": figure_paths,
            **diagnostic_paths,
        },
    }

