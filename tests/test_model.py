from copy import deepcopy

import pytest
import torch

from models import MODEL_REGISTRY, build_model
from models.base_model import BaseForecastModel
from models.v0_transformer import V0Transformer
from utils.config import load_config


def _model(output_dim=1, dropout=0.0):
    return V0Transformer(
        input_dim=7,
        output_dim=output_dim,
        seq_len=96,
        pred_len=24,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=dropout,
        activation="gelu",
        epsilon=1.0e-6,
    )


@pytest.mark.parametrize(("output_dim", "shape"), [(7, (2, 24, 7, 3)), (1, (2, 24, 1, 3))])
def test_v0_forward_has_unified_shape(output_dim, shape):
    model = _model(output_dim=output_dim)
    prediction = model(torch.randn(2, 96, 7))

    assert prediction.shape == shape
    assert torch.isfinite(prediction).all()


def test_v0_is_a_base_forecast_model_and_registry_builds_it(project_root):
    config = deepcopy(load_config(project_root / "configs/v0.yaml"))
    config["model"].update(
        {"d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "dropout": 0.0}
    )
    model = build_model("v0", config, input_dim=7, output_dim=1)

    assert isinstance(model, BaseForecastModel)
    assert MODEL_REGISTRY["v0"] is V0Transformer
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_registry_rejects_unknown_model(project_root):
    config = load_config(project_root / "configs/v0.yaml")
    with pytest.raises(ValueError, match="unknown.*available.*v0"):
        build_model("missing", config, input_dim=7, output_dim=1)


@pytest.mark.parametrize(
    ("bad_input", "message"),
    [
        (torch.randn(96, 7), "rank 3"),
        (torch.randn(2, 95, 7), "sequence length"),
        (torch.randn(2, 96, 6), "input channels"),
    ],
)
def test_v0_rejects_invalid_input_shape(bad_input, message):
    with pytest.raises(ValueError, match=message):
        _model()(bad_input)


def test_extreme_negative_gap_logits_still_produce_strict_intervals():
    model = _model()
    with torch.no_grad():
        model.output_projection.weight.zero_()
        raw_bias = model.output_projection.bias.view(24, 1, 3)
        raw_bias[..., 0].zero_()
        raw_bias[..., 1:].fill_(-100.0)

    prediction = model(torch.zeros(2, 96, 7))

    assert torch.all(prediction[..., 0] < prediction[..., 1])
    assert torch.all(prediction[..., 1] < prediction[..., 2])


def test_forward_does_not_depend_on_sort(monkeypatch):
    def forbidden_sort(*args, **kwargs):
        raise AssertionError("sorting must not be used to construct intervals")

    monkeypatch.setattr(torch, "sort", forbidden_sort)
    monkeypatch.setattr(torch.Tensor, "sort", forbidden_sort)
    prediction = _model()(torch.randn(1, 96, 7))

    assert prediction.shape == (1, 24, 1, 3)


def test_v0_backward_produces_finite_gradients():
    model = _model()
    prediction = model(torch.randn(2, 96, 7))
    prediction.square().mean().backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
