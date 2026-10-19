from copy import deepcopy
from pathlib import Path

import pytest

from run import execute, parse_args, render_report, render_v0_report
from utils.config import load_config


def test_parse_args_accepts_primary_entry_point():
    args = parse_args(["--model", "v0", "--config", "configs/v0.yaml"])
    assert args.model == "v0"
    assert args.config == Path("configs/v0.yaml")
    assert args.device is None
    assert args.resume_from is None


def test_execute_rejects_model_config_mismatch(project_root):
    config = deepcopy(load_config(project_root / "configs/v0.yaml"))
    config["model"]["name"] = "v1"
    with pytest.raises(ValueError, match="does not match"):
        execute("v0", config)


def test_execute_rejects_unknown_model(project_root):
    config = load_config(project_root / "configs/v0.yaml")
    with pytest.raises(ValueError, match="unknown.*available.*v0"):
        execute("missing", config)


def test_v1_is_registered_and_requires_an_explicit_pretrain_checkpoint(project_root):
    from models import MODEL_REGISTRY, build_model

    config = load_config(project_root / "configs" / "v1.yaml")
    assert set(MODEL_REGISTRY) >= {"v0", "v1"}
    with pytest.raises(ValueError, match="pretrain.checkpoint"):
        build_model("v1", config, input_dim=7, output_dim=1)


def test_report_renderer_contains_required_fields(project_root):
    path = project_root / "results" / "test_report.md"
    result = {
        "experiment_id": "demo",
        "best_epoch": 2,
        "best_val_loss": 0.1,
        "total_params": 100,
        "trainable_params": 80,
        "peak_gpu_memory_bytes": None,
        "train_seconds": 1.5,
        "inference_seconds": 0.2,
        "inference_latency_seconds": 0.01,
        "test_loss": 0.2,
        "test_metrics": {
            "overall": {
                "mae": 1.0,
                "rmse": 2.0,
                "picp": 0.9,
                "mpiw": 3.0,
                "pinaw": 0.4,
                "coverage_error": 0.0,
                "interval_score": 4.0,
                "crps": 0.75,
                "qice": 1.25,
            },
            "by_horizon": [{"horizon": 1, "mae": 1.0, "picp": 0.9, "mpiw": 3.0}],
        },
        "paths": {"best_checkpoint": Path("checkpoints/demo/best.pt"), "resolved_config": Path("results/demo/config.resolved.yaml")},
    }

    render_v0_report(result, path)
    content = path.read_text(encoding="utf-8")

    for label in (
        "Total parameters",
        "Trainable parameters",
        "Peak GPU memory",
        "Train time",
        "Inference latency",
        "MAE",
        "RMSE",
        "PICP",
        "MPIW",
        "PINAW",
        "Coverage Error",
        "Interval Score",
        "CRPS",
        "QICE (%)",
        "By Horizon",
    ):
        assert label in content


def test_v1_report_contains_provenance_adaptation_ratio_and_metrics(tmp_path):
    result = {
        "model_name": "v1",
        "experiment_id": "v1-demo",
        "best_epoch": 1,
        "best_val_loss": 0.2,
        "total_params": 200,
        "trainable_params": 20,
        "trainable_ratio": 0.1,
        "peak_gpu_memory_bytes": None,
        "train_seconds": 1.0,
        "inference_seconds": 0.1,
        "inference_latency_seconds": 0.01,
        "pretrain_checkpoint": "checkpoints/pretrain-id/best.pt",
        "pretrain_checkpoint_id": "pretrain-id",
        "adaptation_mode": "qk",
        "adapter_rank": 8,
        "adapter_alpha": 16.0,
        "test_metrics": {
            "overall": {"mae": 1.0, "rmse": 2.0, "picp": 0.9, "mpiw": 3.0, "pinaw": 0.4, "coverage_error": 0.0, "interval_score": 4.0},
            "by_horizon": [{"horizon": 1, "mae": 1.0, "rmse": 2.0, "picp": 0.9, "mpiw": 3.0, "pinaw": 0.4, "coverage_error": 0.0, "interval_score": 4.0}],
        },
        "paths": {"best_checkpoint": "best.pt", "resolved_config": "config.yaml"},
    }
    path = tmp_path / "V1_result.md"
    render_report(result, path)
    content = path.read_text(encoding="utf-8")
    for label in ("V1", "pretrain-id", "qk", "Adapter rank", "Adapter alpha", "Trainable ratio", "GPU memory", "MAE", "RMSE", "PICP", "PINAW", "Interval Score", "By Horizon"):
        assert label in content

