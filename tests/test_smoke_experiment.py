from copy import deepcopy

import numpy as np
import pytest

from run import execute
from utils.config import load_config


pytestmark = pytest.mark.slow


def test_real_etth1_end_to_end_smoke(project_root):
    config = deepcopy(load_config(project_root / "configs/v0.yaml"))
    config["experiment"].update(
        {
            "name": "v0_smoke",
            "device": "cpu",
            "output_root": str(project_root / "results" / "test-runs"),
        }
    )
    config["data"].update(
        {"seq_len": 16, "pred_len": 2, "stride": 500, "batch_size": 4}
    )
    config["model"].update(
        {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0}
    )
    config["training"].update({"epochs": 1, "early_stopping_patience": 1})

    result = execute("v0", config)

    paths = result["paths"]
    for key in (
        "best_checkpoint",
        "last_checkpoint",
        "metrics_json",
        "predictions_npz",
        "resolved_config",
        "log",
        "metrics_jsonl",
        "report",
    ):
        assert paths[key].is_file(), key
    saved = np.load(paths["predictions_npz"])
    prediction = saved["predictions"]
    assert prediction.ndim == 4
    assert prediction.shape[1:] == (2, 1, 3)
    assert np.all(prediction[..., 0] < prediction[..., 1])
    assert np.all(prediction[..., 1] < prediction[..., 2])
    report = paths["report"].read_text(encoding="utf-8")
    assert "V0 ETTh1 Interval Forecast Result" in report
    assert "By Horizon" in report
