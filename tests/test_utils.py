import json
import os

import numpy as np
import torch

from utils.logger import append_jsonl, create_experiment_paths, setup_logger
from utils.seed import set_seed


def _path_config(tmp_path):
    return {
        "experiment": {
            "name": "utils_test",
            "seed": 7,
            "device": "cpu",
            "output_root": str(tmp_path),
            "deterministic": True,
        },
        "logging": {"level": "INFO"},
    }


def test_set_seed_repeats_numpy_and_torch_sequences():
    set_seed(123, deterministic=True)
    first_numpy = np.random.rand(4)
    first_torch = torch.rand(4)
    set_seed(123, deterministic=True)

    np.testing.assert_array_equal(np.random.rand(4), first_numpy)
    torch.testing.assert_close(torch.rand(4), first_torch, rtol=0, atol=0)


def test_deterministic_seed_configures_cublas_before_cuda(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(123, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_experiment_paths_are_unique_and_scoped(tmp_path):
    first = create_experiment_paths(_path_config(tmp_path))
    second = create_experiment_paths(_path_config(tmp_path))

    assert first.experiment_id != second.experiment_id
    assert first.log_dir.parent == tmp_path / "logs"
    assert first.checkpoint_dir.parent == tmp_path / "checkpoints"
    assert first.result_dir.parent == tmp_path / "results"
    assert first.log_dir.is_dir()
    assert first.checkpoint_dir.is_dir()
    assert first.result_dir.is_dir()


def test_logger_setup_does_not_duplicate_handlers_and_jsonl_is_valid(tmp_path):
    paths = create_experiment_paths(_path_config(tmp_path))
    first = setup_logger(paths, level="INFO")
    second = setup_logger(paths, level="INFO")

    assert first is second
    assert len(second.handlers) == 2
    second.info("hello")
    append_jsonl(paths.metrics_path, {"epoch": 1, "loss": 0.25})

    assert "hello" in paths.log_path.read_text(encoding="utf-8")
    assert json.loads(paths.metrics_path.read_text(encoding="utf-8")) == {
        "epoch": 1,
        "loss": 0.25,
    }

