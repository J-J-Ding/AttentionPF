from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from dataload import create_dataloaders
from epx_main import run_experiment
from models import build_model
from models.v1_pretrain_adapter import V1MaskedPretrainer, V1PatchBackbone
from test_v1_config import valid_v1_config
from utils.checkpoint import capture_rng_state, save_checkpoint


def _config(etth1_path, output_root, mode):
    config = deepcopy(valid_v1_config())
    config["experiment"].update({"name": f"v1_{mode}_integration", "device": "cpu", "output_root": str(output_root)})
    config["data"].update({"path": str(etth1_path), "seq_len": 16, "pred_len": 2, "stride": 500, "batch_size": 4})
    config["patch"].update({"length": 8, "stride": 8})
    config["model"].update({"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0})
    config["adaptation"].update({"mode": mode, "rank": 2, "alpha": 4.0})
    config["training"].update({"epochs": 1, "early_stopping_patience": 1})
    config["visualization"].update({"formats": ["png"], "dpi": 50})
    return config


def _write_stage_i_checkpoint(path, config):
    backbone = V1PatchBackbone(
        input_dim=7, seq_len=16, patch_length=8, patch_stride=8,
        d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, activation="gelu",
    )
    model = V1MaskedPretrainer(backbone)
    optimizer = torch.optim.AdamW(model.parameters())
    generator = torch.Generator().manual_seed(1)
    save_checkpoint(path, {
        "format_version": 1, "stage": "pretrain", "model_name": "v1", "epoch": 1,
        "global_step": 1, "model_state_dict": model.state_dict(),
        "backbone_state_dict": model.backbone_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": {"mean": np.zeros(7), "scale": np.ones(7)},
        "config": config, "best_metric": 1.0, "best_epoch": 1, "patience": 0,
        "best_model_state_dict": model.state_dict(), "rng_state": capture_rng_state(),
        "mask_generator_state": generator.get_state(),
        "train_loader_generator_state": generator.get_state(),
    })


@pytest.mark.parametrize("mode", ["full", "qk", "qkv"])
def test_v1_modes_run_through_shared_supervised_engine(etth1_path, tmp_path, mode):
    config = _config(etth1_path, tmp_path / mode, mode)
    checkpoint = tmp_path / "shared-pretrain" / "best.pt"
    if not checkpoint.exists():
        _write_stage_i_checkpoint(checkpoint, config)
    config["pretrain"]["checkpoint"] = str(checkpoint)
    bundle = create_dataloaders(config)

    def small(loader, training=False):
        return DataLoader(
            Subset(loader.dataset, range(min(8, len(loader.dataset)))), batch_size=4,
            shuffle=training,
            generator=torch.Generator().manual_seed(2026) if training else None,
        )

    model = build_model("v1", config, bundle.input_dim, bundle.output_dim)
    result = run_experiment(model, small(bundle.train_loader, True), small(bundle.val_loader), small(bundle.test_loader), config, bundle.scaler)
    prediction = result["test"]["predictions"]
    assert prediction.shape[1:] == (2, 1, 19)
    assert np.all(np.diff(prediction, axis=-1) > 0)
    for key in ("best_checkpoint", "last_checkpoint", "metrics_json", "predictions_npz", "resolved_config", "figures_dir"):
        assert result["paths"][key].exists()
    assert result["adaptation_mode"] == mode
    assert result["pretrain_checkpoint_id"] == "shared-pretrain"
    assert result["trainable_ratio"] == pytest.approx(result["trainable_params"] / result["total_params"])
    saved = np.load(result["paths"]["predictions_npz"])
    assert saved["quantile_levels"].shape == (19,)
    assert saved["coverage_levels"].shape == (9,)
    assert "coverage_calibration_png" in result["paths"]["figures"]
