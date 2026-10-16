"""Masked reconstruction objective used only during V1 Stage I."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_reconstruction_loss(
    reconstructed: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    """Mean squared error over masked patches and all patch elements."""
    if reconstructed.shape != target.shape or reconstructed.ndim != 3:
        raise ValueError(
            "reconstructed and target shape must match as [B,N,patch_dim]"
        )
    if mask.shape != reconstructed.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B,N]")
    if not bool(mask.any()):
        raise ValueError("at least one masked patch is required")
    if not torch.isfinite(reconstructed).all() or not torch.isfinite(target).all():
        raise ValueError("reconstructed and target tensors must be finite")
    error = (reconstructed - target).square()
    return error[mask].mean()
