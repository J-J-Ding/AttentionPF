import importlib


EXPECTED_HEADER = "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"
PACKAGES = ("dataload", "models", "losses", "metrics", "utils")
RUNTIME_DIRS = ("datasets/processed", "reports", "checkpoints", "logs", "results")


def test_etth1_is_stored_at_the_canonical_path(project_root, etth1_path):
    assert etth1_path.is_file()
    assert not (project_root / "ETTh1.csv").exists()
    assert etth1_path.open(encoding="utf-8").readline().strip() == EXPECTED_HEADER


def test_required_packages_are_importable():
    for package in PACKAGES:
        assert importlib.import_module(package).__name__ == package


def test_runtime_directories_exist(project_root):
    for relative_path in RUNTIME_DIRS:
        assert (project_root / relative_path).is_dir(), relative_path


def test_v1_two_stage_files_and_spec_exist(project_root):
    for relative_path in (
        "specs/V1_pretrain_adapter_spec.md",
        "models/v1_pretrain_adapter.py",
        "configs/v1.yaml",
        "pretrain.py",
        "reports/V1_result.md",
    ):
        assert (project_root / relative_path).is_file(), relative_path
