from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from utils.visualization import generate_experiment_figures


def _inputs(project_root: Path):
    n_samples, pred_len, channels = 12, 4, 2
    targets = np.arange(n_samples * pred_len * channels, dtype=np.float32).reshape(
        n_samples, pred_len, channels
    )
    point = targets + 0.25
    predictions = np.stack((point - 1.0, point, point + 1.5), axis=-1)
    history = [
        {"epoch": 1, "train_loss": 0.8, "val_loss": 0.9},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
    ]
    by_horizon = []
    for horizon in range(1, pred_len + 1):
        by_horizon.append(
            {
                "horizon": horizon,
                "mae": 0.1 * horizon,
                "rmse": 0.2 * horizon,
                "picp": 0.85 + 0.01 * horizon,
                "mpiw": 2.0 + horizon,
                "pinaw": 0.1 + 0.01 * horizon,
                "coverage_error": abs(0.9 - (0.85 + 0.01 * horizon)),
                "interval_score": 3.0 + horizon,
                "crps": 0.3 + 0.1 * horizon,
                "qice": 1.0 + 0.2 * horizon,
            }
        )
    metrics = {"overall": {}, "by_horizon": by_horizon}
    config = {
        "interval": {
            "lower_quantile": 0.05,
            "median_quantile": 0.5,
            "upper_quantile": 0.95,
            "epsilon": 1.0e-6,
        },
        "evaluation": {"coverage": 0.9},
        "visualization": {
            "enabled": True,
            "formats": ["png", "pdf"],
            "dpi": 120,
            "max_points": 8,
            "sample_index": 1,
            "channel_index": 1,
        },
    }
    result_dir = project_root / "results" / f"visualization_test_{uuid4().hex}"
    return result_dir, history, predictions, targets, metrics, config


def test_generate_experiment_figures_writes_three_publication_figures(project_root):
    args = _inputs(project_root)

    paths = generate_experiment_figures(*args)

    assert set(paths) == {
        "training_curves_png",
        "training_curves_pdf",
        "forecast_interval_png",
        "forecast_interval_pdf",
        "horizon_metrics_png",
        "horizon_metrics_pdf",
    }
    for path in paths.values():
        assert path.parent == args[0] / "figures"
        assert path.is_file()
        assert path.stat().st_size > 500
    assert paths["training_curves_png"].read_bytes().startswith(b"\x89PNG")
    assert paths["training_curves_pdf"].read_bytes().startswith(b"%PDF")


def test_generate_experiment_figures_rejects_bad_shapes_and_channel(project_root):
    result_dir, history, predictions, targets, metrics, config = _inputs(project_root)
    with pytest.raises(ValueError, match="prediction.*shape"):
        generate_experiment_figures(
            result_dir, history, predictions[..., 1], targets, metrics, config
        )

    config["visualization"]["channel_index"] = 2
    with pytest.raises(ValueError, match="channel_index"):
        generate_experiment_figures(
            result_dir, history, predictions, targets, metrics, config
        )


def test_disabled_visualization_creates_no_directory(project_root):
    args = list(_inputs(project_root))
    args[-1]["visualization"]["enabled"] = False

    paths = generate_experiment_figures(*args)

    assert paths == {}
    assert not (args[0] / "figures").exists()
