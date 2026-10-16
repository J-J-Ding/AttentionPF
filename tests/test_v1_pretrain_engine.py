from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, Subset

from dataload import create_dataloaders
from dataload.pretrain import create_pretrain_loaders
from pretrain import load_pretrain_checkpoint, run_pretraining
from test_v1_config import valid_v1_config


def tiny_pretrain_config(etth1_path: Path, output_root: Path):
    config = deepcopy(valid_v1_config())
    config["experiment"].update({"name": "v1_pretrain_test", "output_root": str(output_root)})
    config["data"].update({"path": str(etth1_path), "stride": 500, "batch_size": 8})
    config["model"].update({"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0})
    config["adaptation"].update({"rank": 2, "alpha": 4})
    config["pretrain"].update({"epochs": 1, "early_stopping_patience": 1})
    config["visualization"].update({"formats": ["png"], "dpi": 50})
    return config


def test_pretrain_internal_split_is_ordered_disjoint_and_stays_inside_official_train(etth1_path, tmp_path):
    config = tiny_pretrain_config(etth1_path, tmp_path)
    bundle = create_dataloaders(config)
    train_loader, val_loader = create_pretrain_loaders(bundle, config)
    assert isinstance(train_loader.dataset, Subset)
    assert isinstance(val_loader.dataset, Subset)
    train_indices = list(train_loader.dataset.indices)
    val_indices = list(val_loader.dataset.indices)
    assert max(train_indices) < min(val_indices)
    assert set(train_indices).isdisjoint(val_indices)
    assert sorted(train_indices + val_indices) == list(range(len(bundle.train_loader.dataset)))
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    base = bundle.train_loader.dataset
    assert int(base.forecast_starts[max(val_indices)]) + base.pred_len <= bundle.split_metadata.train_end


def test_stage_i_run_writes_complete_artifacts_and_strict_checkpoint(etth1_path, tmp_path):
    config = tiny_pretrain_config(etth1_path, tmp_path)
    result = run_pretraining(config)
    paths = result["paths"]
    for key in ("best_checkpoint", "last_checkpoint", "resolved_config", "metrics_json", "metrics_jsonl", "log", "training_curve"):
        assert Path(paths[key]).is_file(), key
    state = load_pretrain_checkpoint(paths["best_checkpoint"], expected_config=config)
    required = {
        "backbone_state_dict", "model_state_dict", "optimizer_state_dict", "scaler_state_dict",
        "config", "rng_state", "mask_generator_state", "train_loader_generator_state",
        "best_metric", "best_epoch", "patience", "best_checkpoint_state",
    }
    assert required <= set(state)
    assert state["stage"] == "pretrain"
    assert result["history"][0]["train_loss"] > 0
    assert result["history"][0]["val_loss"] > 0
    last = torch.load(paths["last_checkpoint"], map_location="cpu", weights_only=False)
    best = last["best_checkpoint_state"]
    exported = {
        name.removeprefix("backbone."): value
        for name, value in best["model_state_dict"].items()
        if name.startswith("backbone.") and name != "backbone.mask_token"
    }
    assert exported.keys() == best["backbone_state_dict"].keys()
    assert all(torch.equal(exported[name], best["backbone_state_dict"][name]) for name in exported)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda c: c["data"].update({"feature_cols": list(reversed(c["data"]["feature_cols"]))}), "feature_cols"),
        (lambda c: c["data"].update({"path": "datasets/raw/different.csv"}), "data.path"),
        (lambda c: c["data"]["split"].update({"train": 0.6, "test": 0.3}), "data.split"),
        (lambda c: c["patch"].update({"stride": 4}), "patch.stride"),
        (lambda c: c["model"].update({"n_layers": 2}), "model.n_layers"),
        (lambda c: c["model"].update({"dropout": 0.2}), "model.dropout"),
        (lambda c: c["mask"].update({"random_ratio": 0.3}), "mask.random_ratio"),
    ],
)
def test_pretrain_checkpoint_rejects_semantic_contract_mismatch(etth1_path, tmp_path, mutation, message):
    config = tiny_pretrain_config(etth1_path, tmp_path)
    result = run_pretraining(config)
    incompatible = deepcopy(config)
    mutation(incompatible)
    with pytest.raises(ValueError, match=message):
        load_pretrain_checkpoint(result["paths"]["best_checkpoint"], expected_config=incompatible)


def test_resume_restores_loader_and_mask_rng_for_next_epoch(etth1_path, tmp_path):
    first = tiny_pretrain_config(etth1_path, tmp_path / "first")
    initial = run_pretraining(first)
    resumed = deepcopy(first)
    resumed["experiment"]["output_root"] = str(tmp_path / "resumed")
    resumed["pretrain"]["epochs"] = 2
    resumed["training"]["resume_from"] = str(initial["paths"]["last_checkpoint"])
    result = run_pretraining(resumed)
    state = load_pretrain_checkpoint(result["paths"]["last_checkpoint"], expected_config=resumed)
    assert state["epoch"] == 2
    assert len(result["history"]) == 1
    assert torch.is_tensor(state["mask_generator_state"])
