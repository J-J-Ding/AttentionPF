from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataload.dataloader import create_dataloaders
from epx_main import predict, run_experiment, train_one_epoch, validate
from losses.quantile_loss import QuantileLoss
from models import build_model
from models.base_model import BaseForecastModel
from utils.config import load_config
from utils.seed import set_seed


def _small_setup(project_root):
    config = deepcopy(load_config(project_root / "configs/v0.yaml"))
    config["experiment"].update(
        {"name": "engine_test", "device": "cpu", "output_root": str(project_root)}
    )
    config["data"].update({"seq_len": 24, "pred_len": 4, "batch_size": 8})
    config["model"].update(
        {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0}
    )
    config["training"].update({"epochs": 2, "early_stopping_patience": 2})
    bundle = create_dataloaders(config)

    def subset_loader(loader, count=16):
        return DataLoader(Subset(loader.dataset, range(count)), batch_size=8, shuffle=False)

    model = build_model("v0", config, bundle.input_dim, bundle.output_dim)
    return config, bundle, model, subset_loader(bundle.train_loader), subset_loader(bundle.val_loader), subset_loader(bundle.test_loader)


def test_train_updates_parameters_and_returns_element_weighted_loss(project_root):
    config, _, model, train_loader, _, _ = _small_setup(project_root)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = QuantileLoss((0.05, 0.5, 0.95))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = train_one_epoch(model, train_loader, optimizer, loss_fn, torch.device("cpu"), config)

    assert np.isfinite(result["loss"])
    assert result["elements"] == 16 * 4 * 1
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_validate_and_predict_do_not_update_model_and_use_original_scale(project_root):
    config, bundle, model, _, val_loader, _ = _small_setup(project_root)
    loss_fn = QuantileLoss((0.05, 0.5, 0.95))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = validate(model, val_loader, loss_fn, bundle.scaler, torch.device("cpu"), config)
    predictions = predict(model, val_loader, bundle.scaler, torch.device("cpu"), config)

    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert result["predictions"].shape == (16, 4, 1, 3)
    assert result["targets"].shape == (16, 4, 1)
    assert predictions.shape == (16, 4, 1, 3)
    assert len(result["metrics"]["by_horizon"]) == 4
    assert abs(result["targets"].mean()) > 1.0  # OT is inverse-scaled, not near standardized zero.


class _NonFiniteModel(BaseForecastModel):
    def forward(self, x):
        return torch.full((x.shape[0], 4, 1, 3), float("nan"), device=x.device, requires_grad=True)


def test_training_fails_fast_on_nonfinite_predictions(project_root):
    config, _, _, train_loader, _, _ = _small_setup(project_root)
    model = _NonFiniteModel()
    dummy_parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([dummy_parameter])
    with pytest.raises(FloatingPointError, match="non-finite prediction"):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            QuantileLoss((0.05, 0.5, 0.95)),
            torch.device("cpu"),
            config,
        )


def test_run_experiment_saves_and_reloads_best_checkpoint(project_root):
    set_seed(2026, deterministic=True)
    config, bundle, model, train_loader, val_loader, test_loader = _small_setup(project_root)

    result = run_experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        bundle.scaler,
    )

    assert result["best_epoch"] in {1, 2}
    assert result["test"]["predictions"].shape == (16, 4, 1, 3)
    assert result["paths"]["best_checkpoint"].is_file()
    assert result["paths"]["last_checkpoint"].is_file()
    assert result["paths"]["metrics_json"].is_file()
    assert result["paths"]["predictions_npz"].is_file()
    assert result["paths"]["resolved_config"].is_file()
    saved = np.load(result["paths"]["predictions_npz"])
    assert saved["predictions"].shape == (16, 4, 1, 3)
    assert np.all(saved["predictions"][..., 0] < saved["predictions"][..., 1])
    assert np.all(saved["predictions"][..., 1] < saved["predictions"][..., 2])

