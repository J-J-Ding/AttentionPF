from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataload.dataloader import create_dataloaders
from epx_main import build_loss, predict, run_experiment, train_one_epoch, validate
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
    config["data"].update(
        {
            "seq_len": 24,
            "pred_len": 4,
            "batch_size": 8,
            "features": "MS",
            "target_cols": ["OT"],
        }
    )
    config["model"].update(
        {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0}
    )
    config["training"].update({"epochs": 2, "early_stopping_patience": 2})
    bundle = create_dataloaders(config)

    def subset_loader(loader, count=16, *, training=False):
        generator = torch.Generator().manual_seed(config["experiment"]["seed"]) if training else None
        return DataLoader(
            Subset(loader.dataset, range(count)),
            batch_size=8,
            shuffle=training,
            generator=generator,
        )

    model = build_model("v0", config, bundle.input_dim, bundle.output_dim)
    return config, bundle, model, subset_loader(bundle.train_loader, training=True), subset_loader(bundle.val_loader), subset_loader(bundle.test_loader)


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
    assert result["paths"]["figures_dir"].is_dir()
    assert set(result["paths"]["figures"]) == {
        "training_curves_png",
        "training_curves_pdf",
        "forecast_interval_png",
        "forecast_interval_pdf",
        "horizon_metrics_png",
        "horizon_metrics_pdf",
    }
    assert all(path.is_file() for path in result["paths"]["figures"].values())
    saved = np.load(result["paths"]["predictions_npz"])
    assert saved["predictions"].shape == (16, 4, 1, 3)
    assert np.all(saved["predictions"][..., 0] < saved["predictions"][..., 1])
    assert np.all(saved["predictions"][..., 1] < saved["predictions"][..., 2])


def test_nonzero_interval_weight_contributes_to_training_objective():
    config = deepcopy(load_config("configs/v0.yaml"))
    config["loss"]["interval_weight"] = 0.5
    prediction = torch.tensor([[[[0.0, 1.0, 2.0]]]])
    target = torch.tensor([[[1.0]]])
    quantile_only = QuantileLoss((0.05, 0.5, 0.95))(prediction, target)

    combined = build_loss(config)(prediction, target)

    assert combined > quantile_only
    torch.testing.assert_close(combined, quantile_only + torch.tensor(1.0))


def test_resume_at_completed_budget_preserves_best_and_loader_rng(project_root):
    set_seed(2026, deterministic=True)
    config, bundle, model, train_loader, val_loader, test_loader = _small_setup(project_root)
    config["training"]["epochs"] = 1
    first = run_experiment(model, train_loader, val_loader, test_loader, config, bundle.scaler)
    saved = torch.load(first["paths"]["last_checkpoint"], map_location="cpu", weights_only=False)
    expected_generator_state = saved["train_loader_generator_state"]

    resumed_config, resumed_bundle, resumed_model, resumed_train, resumed_val, resumed_test = _small_setup(project_root)
    resumed_config["training"].update({"epochs": 1, "resume_from": str(first["paths"]["last_checkpoint"])})
    resumed = run_experiment(
        resumed_model, resumed_train, resumed_val, resumed_test, resumed_config, resumed_bundle.scaler
    )

    assert resumed["best_epoch"] == first["best_epoch"]
    assert resumed["paths"]["best_checkpoint"].is_file()
    assert torch.equal(resumed_train.generator.get_state(), expected_generator_state)


def test_checkpoint_records_patience_best_weights_and_sampler_state(project_root):
    set_seed(2026, deterministic=True)
    config, bundle, model, train_loader, val_loader, test_loader = _small_setup(project_root)
    config["training"]["epochs"] = 1
    result = run_experiment(model, train_loader, val_loader, test_loader, config, bundle.scaler)
    state = torch.load(result["paths"]["last_checkpoint"], map_location="cpu", weights_only=False)

    assert state["best_epoch"] == 1
    assert state["patience"] == 0
    assert state["best_model_state_dict"]
    assert torch.equal(state["train_loader_generator_state"], train_loader.generator.get_state())

