import pytest
import torch
from torch import nn

from dataload.pretrain import patchify
from losses.reconstruction_loss import masked_reconstruction_loss
from models.v1_pretrain_adapter import V1MaskedPretrainer, V1PatchBackbone


def make_backbone():
    return V1PatchBackbone(
        input_dim=2,
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


def test_backbone_shapes_and_masked_input_values_are_hidden_before_encoder():
    backbone = make_backbone()
    patches = patchify(torch.arange(32.0).reshape(2, 8, 2), 4, 2)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    changed = patches + 10000.0
    original_embeddings = backbone.embed_patches(patches, mask)
    changed_embeddings = backbone.embed_patches(changed, mask)
    assert original_embeddings.shape == (2, 3, 8)
    assert torch.equal(original_embeddings[mask], changed_embeddings[mask])
    assert not torch.equal(original_embeddings[~mask], changed_embeddings[~mask])
    assert backbone(patches, mask).shape == (2, 3, 8)
    assert backbone.position_embedding.shape == (1, 3, 8)


def test_pretrainer_reconstructs_flat_patches_with_single_linear_head():
    model = V1MaskedPretrainer(make_backbone())
    x = torch.randn(2, 8, 2)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    reconstructed, target = model(x, mask)
    assert reconstructed.shape == target.shape == (2, 3, 8)
    assert isinstance(model.reconstruction_head, nn.Linear)
    assert "reconstruction_head.weight" not in model.backbone_state_dict()
    assert "mask_token" not in model.backbone_state_dict()


def test_masked_only_loss_ignores_visible_predictions_and_matches_hand_mse():
    target = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    prediction = torch.tensor([[[100.0, -100.0], [1.0, 9.0]]])
    mask = torch.tensor([[False, True]])
    loss = masked_reconstruction_loss(prediction, target, mask)
    assert loss.item() == pytest.approx(((1.0 - 3.0) ** 2 + (9.0 - 5.0) ** 2) / 2)
    prediction[:, 0] = 9999.0
    assert masked_reconstruction_loss(prediction, target, mask).item() == pytest.approx(loss.item())


@pytest.mark.parametrize(
    "prediction,target,mask,message",
    [
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool), "masked"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 3, 3), torch.ones(1, 2, dtype=torch.bool), "shape"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3, dtype=torch.bool), "mask"),
        (torch.full((1, 2, 3), float("nan")), torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), "finite"),
    ],
)
def test_masked_loss_rejects_invalid_inputs(prediction, target, mask, message):
    with pytest.raises(ValueError, match=message):
        masked_reconstruction_loss(prediction, target, mask)


def test_reconstruction_backward_reaches_all_pretrained_components():
    model = V1MaskedPretrainer(make_backbone())
    x = torch.randn(2, 8, 2)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    reconstructed, target = model(x, mask)
    masked_reconstruction_loss(reconstructed, target, mask).backward()
    required_fragments = (
        "patch_embedding", "position_embedding", "q_proj", "k_proj", "v_proj",
        "out_proj", "linear1", "linear2", "norm1", "norm2", "reconstruction_head",
    )
    names = {name for name, parameter in model.named_parameters() if parameter.grad is not None and torch.isfinite(parameter.grad).all()}
    for fragment in required_fragments:
        assert any(fragment in name for name in names), fragment
