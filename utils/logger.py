"""Experiment paths plus human-readable and structured logging."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentPaths:
    experiment_id: str
    log_dir: Path
    checkpoint_dir: Path
    result_dir: Path
    report_path: Path

    @property
    def log_path(self) -> Path:
        return self.log_dir / "train.log"

    @property
    def metrics_path(self) -> Path:
        return self.log_dir / "metrics.jsonl"


def create_experiment_paths(config: dict[str, Any]) -> ExperimentPaths:
    root = Path(config["experiment"]["output_root"]).resolve()
    name = str(config["experiment"]["name"])
    seed = int(config["experiment"]["seed"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    experiment_id = f"{name}_seed{seed}_{stamp}_{uuid.uuid4().hex[:6]}"
    log_dir = root / "logs" / experiment_id
    checkpoint_dir = root / "checkpoints" / experiment_id
    result_dir = root / "results" / experiment_id
    for path in (log_dir, checkpoint_dir, result_dir, root / "reports"):
        path.mkdir(parents=True, exist_ok=False if path in (log_dir, checkpoint_dir, result_dir) else True)
    return ExperimentPaths(
        experiment_id=experiment_id,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        report_path=root
        / "reports"
        / f"{str(config.get('model', {}).get('name', 'v0')).upper()}_result.md",
    )


def setup_logger(paths: ExperimentPaths, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(f"attentionpf.{paths.experiment_id}")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(paths.log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

