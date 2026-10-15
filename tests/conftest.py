from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def etth1_path(project_root: Path) -> Path:
    return project_root / "datasets" / "raw" / "ETTh1.csv"
