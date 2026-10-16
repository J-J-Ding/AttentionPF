"""V1 patch Transformer backbone and masked pretraining model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dataload.pretrain import patchify


class SeparatedSelfAttention(nn.Module):
    """Multi-head self-attention with explicit Q/K/V/O projections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        attended = self.attention_dropout(weights) @ v
        attended = attended.transpose(1, 2).contiguous().reshape(x.shape)
        return self.out_proj(attended)


class V1EncoderLayer(nn.Module):
    """Standard post-norm Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.self_attn = SeparatedSelfAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x)))
        feed_forward = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(feed_forward))


class V1PatchBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        seq_len: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_layers <= 0:
            raise ValueError("input_dim and n_layers must be positive")
        if patch_length > seq_len or (seq_len - patch_length) % patch_stride:
            raise ValueError("patch geometry must exactly cover seq_len")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.patch_dim = patch_length * input_dim
        self.n_patches = 1 + (seq_len - patch_length) // patch_stride
        self.d_model = d_model
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.layers = nn.ModuleList(
            [
                V1EncoderLayer(d_model, n_heads, d_ff, dropout, activation)
                for _ in range(n_layers)
            ]
        )

    def _validate(self, patches: Tensor, mask: Tensor | None) -> None:
        expected = (self.n_patches, self.patch_dim)
        if patches.ndim != 3 or tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"patches must have shape [B,{expected[0]},{expected[1]}], got {tuple(patches.shape)}"
            )
        if mask is not None and (mask.dtype != torch.bool or mask.shape != patches.shape[:2]):
            raise ValueError("mask must be boolean with shape [B,N]")

    def embed_patches(self, patches: Tensor, mask: Tensor | None = None) -> Tensor:
        self._validate(patches, mask)
        embedded = self.patch_embedding(patches)
        if mask is not None:
            token = self.mask_token.to(dtype=embedded.dtype).expand_as(embedded)
            embedded = torch.where(mask.unsqueeze(-1), token, embedded)
        return embedded + self.position_embedding.to(dtype=embedded.dtype)

    def forward(self, patches: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden = self.embed_patches(patches, mask)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class V1MaskedPretrainer(nn.Module):
    def __init__(self, backbone: V1PatchBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.reconstruction_head = nn.Linear(backbone.d_model, backbone.patch_dim)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.backbone.seq_len, self.backbone.input_dim):
            raise ValueError("x does not match the backbone sequence/channel contract")
        target = patchify(x, self.backbone.patch_length, self.backbone.patch_stride)
        hidden = self.backbone(target, mask)
        return self.reconstruction_head(hidden), target

    def backbone_state_dict(self) -> dict[str, Tensor]:
        """Return only Stage-II transferable parameters, excluding mask token."""
        return {
            name: value.detach().clone()
            for name, value in self.backbone.state_dict().items()
            if name != "mask_token"
        }
