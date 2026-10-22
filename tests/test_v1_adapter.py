import pytest
import torch

from losses.interval_loss import QuantileIntervalLoss
from models.v1_pretrain_adapter import (
    LowRankProjectionAdapter,
    V1MaskedPretrainer,
    V1PatchBackbone,
    V1PretrainAdapter,
)
from utils.config import V1_QUANTILES


def backbone_kwargs():
    return dict(
        input_dim=3,
        seq_len=8,
        patch_length=4,
        patch_stride=2,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        dropout=0.0,
        activation="gelu",
    )


def pretrained_state():
    return V1MaskedPretrainer(V1PatchBackbone(**backbone_kwargs())).backbone_state_dict()


def make_model(mode):
    model = V1PretrainAdapter(
        **backbone_kwargs(), pred_len=3, output_dim=1, epsilon=1e-6,
        quantiles=V1_QUANTILES,
        adaptation_mode=mode, adapter_rank=2, adapter_alpha=4.0, adapter_dropout=0.0,
    )
    model.load_pretrained_backbone(pretrained_state())
    return model


def test_low_rank_adapter_is_zero_at_initialization_then_changes_projection():
    adapter = LowRankProjectionAdapter(8, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(2, 3, 8)
    assert torch.count_nonzero(adapter.B) == 0
    assert torch.count_nonzero(adapter.A) > 0
    torch.testing.assert_close(adapter(x), torch.zeros_like(x))
    with torch.no_grad():
        adapter.B.fill_(0.25)
    assert not torch.equal(adapter(x), torch.zeros_like(x))


@pytest.mark.parametrize("mode,expected", [("full", set()), ("qk", {"q", "k"}), ("qkv", {"q", "k", "v"})])
def test_adaptation_mode_installs_only_requested_projection_adapters(mode, expected):
    model = make_model(mode)
    for layer in model.backbone.layers:
        assert set(layer.self_attn.adapters) == expected


def test_projection_adapter_changes_only_its_named_projection():
    model = make_model("qk")
    attention = model.backbone.layers[0].self_attn
    x = torch.randn(2, 3, 8)
    before = {name: attention.project(name, x).detach().clone() for name in ("q", "k", "v")}
    with torch.no_grad():
        attention.adapters["q"].B.fill_(0.5)
    assert not torch.equal(attention.project("q", x), before["q"])
    torch.testing.assert_close(attention.project("k", x), before["k"])
    torch.testing.assert_close(attention.project("v", x), before["v"])


@pytest.mark.parametrize("breakage", ["missing", "extra", "shape"])
def test_pretrained_backbone_loading_is_strict_by_name_and_shape(breakage):
    model = make_model("qk")
    state = pretrained_state()
    if breakage == "missing":
        state.pop(next(iter(state)))
    elif breakage == "extra":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        key = next(name for name in state if name.endswith("q_proj.weight"))
        state[key] = torch.zeros(1)
    with pytest.raises(ValueError, match="backbone"):
        model.load_pretrained_backbone(state)


@pytest.mark.parametrize("mode", ["full", "qk", "qkv"])
def test_forecast_shape_non_crossing_and_backward_freezing(mode):
    model = make_model(mode)
    prediction = model(torch.randn(2, 8, 3))
    assert prediction.shape == (2, 3, 1, 19)
    assert torch.all(torch.diff(prediction, dim=-1) > 0)
    target = torch.randn(2, 3, 1)
    QuantileIntervalLoss(
        V1_QUANTILES, coverage=0.9, interval_weight=0.0
    )(prediction, target).backward()

    adapter_parameters = [(name, p) for name, p in model.named_parameters() if ".adapters." in name]
    frozen_parameters = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    head_parameters = [(name, p) for name, p in model.named_parameters() if name.startswith("output_projection")]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in head_parameters)
    if mode == "full":
        assert not adapter_parameters
        assert all(p.requires_grad for name, p in model.backbone.named_parameters() if name != "mask_token")
    else:
        assert adapter_parameters
        assert all(p.grad is None for _, p in frozen_parameters)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in adapter_parameters)


def test_qk_and_qkv_train_fewer_parameters_than_full():
    summaries = {mode: make_model(mode).parameter_summary() for mode in ("full", "qk", "qkv")}
    assert summaries["qk"]["trainable_params"] < summaries["full"]["trainable_params"]
    assert summaries["qkv"]["trainable_params"] < summaries["full"]["trainable_params"]
    assert summaries["qk"]["trainable_params"] < summaries["qkv"]["trainable_params"]


def test_extreme_gap_logits_keep_all_nineteen_quantiles_strictly_ordered(monkeypatch):
    model = make_model("qk")
    with torch.no_grad():
        model.output_projection.weight.zero_()
        bias = model.output_projection.bias.view(3, 1, 19)
        bias.fill_(-1000.0)
        bias[..., 9] = 0.0

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("sorting must not be used to construct quantiles")

    monkeypatch.setattr(torch, "sort", forbidden_sort)
    prediction = model(torch.zeros(1, 8, 3))

    assert prediction.shape == (1, 3, 1, 19)
    assert torch.all(torch.diff(prediction, dim=-1) > 0)
