import pytest
import torch

from dataload.pretrain import (
    choose_mask,
    patchify,
    sample_random_mask,
    sample_tail_mask,
)


def test_patchify_preserves_temporal_and_channel_order_without_padding():
    x = torch.tensor(
        [[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0], [5.0, 15.0]]]
    )
    patches = patchify(x, patch_length=4, patch_stride=2)
    assert patches.shape == (1, 2, 8)
    assert torch.equal(patches[0, 0], torch.tensor([0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]))
    assert torch.equal(patches[0, 1], torch.tensor([2.0, 12.0, 3.0, 13.0, 4.0, 14.0, 5.0, 15.0]))


@pytest.mark.parametrize(
    "shape,length,stride",
    [((2, 6), 4, 2), ((2, 6, 1), 0, 2), ((2, 6, 1), 7, 2), ((2, 6, 1), 3, 2)],
)
def test_patchify_rejects_invalid_shape_or_non_exact_coverage(shape, length, stride):
    with pytest.raises(ValueError):
        patchify(torch.zeros(shape), length, stride)


def test_random_mask_is_seeded_unique_and_keeps_visible_tokens():
    first = sample_random_mask(3, 10, 0.4, generator=torch.Generator().manual_seed(7), device=torch.device("cpu"))
    second = sample_random_mask(3, 10, 0.4, generator=torch.Generator().manual_seed(7), device=torch.device("cpu"))
    assert first.dtype == torch.bool
    assert torch.equal(first, second)
    assert torch.equal(first.sum(dim=1), torch.tensor([4, 4, 4]))
    assert torch.all(first.sum(dim=1) < 10)


def test_tail_mask_covers_only_the_contiguous_end():
    mask = sample_tail_mask(2, 8, 0.25, device=torch.device("cpu"))
    assert torch.equal(mask, torch.tensor([[False] * 6 + [True] * 2] * 2))


@pytest.mark.parametrize("probability,expected_kind", [(0.0, "tail"), (1.0, "random")])
def test_choose_mask_probability_endpoints_select_the_named_branch(probability, expected_kind):
    mask, kind = choose_mask(
        batch_size=2,
        n_patches=8,
        random_ratio=0.25,
        tail_ratio=0.25,
        random_probability=probability,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    assert kind == expected_kind
    assert mask.shape == (2, 8)
    assert torch.all(mask.any(dim=1))
    assert torch.all((~mask).any(dim=1))


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_mask_samplers_reject_ratios_that_cannot_keep_masked_and_visible_tokens(ratio):
    with pytest.raises(ValueError, match="ratio"):
        sample_tail_mask(1, 5, ratio, device=torch.device("cpu"))
