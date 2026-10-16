"""Primary command-line entry point for every model version."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from dataload import create_dataloaders
from epx_main import run_experiment
from models import MODEL_REGISTRY, build_model
from utils.config import load_config, validate_config
from utils.checkpoint import validate_scaler_state
from utils.seed import set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AttentionPF experiment")
    parser.add_argument("--model", required=True, help="Registered model name, e.g. v0")
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration path")
    parser.add_argument("--device", default=None, help="Override config device")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume checkpoint")
    return parser.parse_args(argv)


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_report(result: dict[str, Any], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    overall = result["test_metrics"]["overall"]
    peak = result["peak_gpu_memory_bytes"]
    peak_text = "N/A (CPU)" if peak is None else f"{peak / (1024 ** 2):.2f} MiB"
    model_name = str(result.get("model_name", "v0")).upper()
    ratio = result.get(
        "trainable_ratio", result["trainable_params"] / result["total_params"]
    )
    lines = [
        f"# {model_name} ETTh1 Interval Forecast Result",
        "",
        f"- Experiment ID: `{result['experiment_id']}`",
        f"- Best epoch: {result['best_epoch']}",
        f"- Best validation loss: {_format_metric(result['best_val_loss'])}",
        f"- Total parameters: {result['total_params']}",
        f"- Trainable parameters: {result['trainable_params']}",
        f"- Trainable ratio: {ratio:.6f}",
        f"- Peak GPU memory: {peak_text}",
        f"- Train time: {result['train_seconds']:.3f} s",
        f"- Inference time: {result['inference_seconds']:.3f} s",
        f"- Inference latency: {result['inference_latency_seconds']:.6f} s/sample",
        f"- Checkpoint: `{result['paths']['best_checkpoint']}`",
        f"- Resolved config: `{result['paths']['resolved_config']}`",
    ]
    if model_name == "V1":
        lines.extend(
            [
                f"- Pretrain checkpoint ID: `{result['pretrain_checkpoint_id']}`",
                f"- Pretrain checkpoint: `{result['pretrain_checkpoint']}`",
                f"- Adaptation mode: `{result['adaptation_mode']}`",
                f"- Adapter rank: {result['adapter_rank']}",
                f"- Adapter alpha: {result['adapter_alpha']}",
            ]
        )
    lines.extend([
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| MAE | {_format_metric(overall.get('mae'))} |",
        f"| RMSE | {_format_metric(overall.get('rmse'))} |",
        f"| PICP | {_format_metric(overall.get('picp'))} |",
        f"| MPIW | {_format_metric(overall.get('mpiw'))} |",
        f"| PINAW | {_format_metric(overall.get('pinaw'))} |",
        f"| Coverage Error | {_format_metric(overall.get('coverage_error'))} |",
        f"| Interval Score | {_format_metric(overall.get('interval_score'))} |",
        "",
        "## By Horizon",
        "",
        "| Horizon | MAE | RMSE | PICP | MPIW | PINAW | Coverage Error | Interval Score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in result["test_metrics"]["by_horizon"]:
        lines.append(
            "| {horizon} | {mae} | {rmse} | {picp} | {mpiw} | {pinaw} | {coverage_error} | {interval_score} |".format(
                horizon=item["horizon"],
                mae=_format_metric(item.get("mae")),
                rmse=_format_metric(item.get("rmse")),
                picp=_format_metric(item.get("picp")),
                mpiw=_format_metric(item.get("mpiw")),
                pinaw=_format_metric(item.get("pinaw")),
                coverage_error=_format_metric(item.get("coverage_error")),
                interval_score=_format_metric(item.get("interval_score")),
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_v0_report(result: dict[str, Any], path: str | Path) -> None:
    """Backward-compatible alias for the version-aware report renderer."""
    render_report(result, path)


def execute(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    normalized = model_name.lower()
    if normalized not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model '{model_name}'; available models: {available}")
    if normalized != str(config["model"]["name"]).lower():
        raise ValueError(
            f"CLI model '{model_name}' does not match config model "
            f"'{config['model']['name']}'"
        )
    validate_config(config)
    set_seed(
        int(config["experiment"]["seed"]),
        bool(config["experiment"]["deterministic"]),
    )
    bundle = create_dataloaders(config)
    model = build_model(normalized, config, bundle.input_dim, bundle.output_dim)
    pretrained_scaler = getattr(model, "pretrain_scaler_state", None)
    if pretrained_scaler is not None:
        validate_scaler_state(pretrained_scaler, bundle.scaler.state_dict())
    result = run_experiment(
        model,
        bundle.train_loader,
        bundle.val_loader,
        bundle.test_loader,
        config,
        bundle.scaler,
    )
    render_v0_report(result, result["paths"]["report"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config["experiment"]["device"] = args.device
    if args.resume_from is not None:
        config["training"]["resume_from"] = str(args.resume_from)
    execute(args.model, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

