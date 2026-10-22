"""V1 patch Transformer backbone and masked pretraining model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataload.pretrain import patchify
from models.base_model import BaseForecastModel


class LowRankProjectionAdapter(nn.Module):
    """LoRA-style additive projection delta with a zero-output initialization."""

    def __init__(self, d_model: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not 0 < rank <= d_model:
            raise ValueError("rank must be in [1, d_model]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.A = nn.Parameter(torch.empty(rank, d_model))
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank

    def forward(self, x: Tensor) -> Tensor:
        return ((self.dropout(x) @ self.A.t()) @ self.B.t()) * self.scale


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
        self.adapters = nn.ModuleDict()

    def configure_adapters(
        self, names: set[str], rank: int, alpha: float, dropout: float
    ) -> None:
        self.adapters = nn.ModuleDict(
            {
                name: LowRankProjectionAdapter(
                    self.d_model, rank=rank, alpha=alpha, dropout=dropout
                )
                for name in sorted(names)
            }
        )

    def project(self, name: str, x: Tensor) -> Tensor:
        projections = {"q": self.q_proj, "k": self.k_proj, "v": self.v_proj}
        if name not in projections:
            raise ValueError(f"unknown projection '{name}'")
        projected = projections[name](x)
        if name in self.adapters:
            projected = projected + self.adapters[name](x)
        return projected

    def _heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        q = self._heads(self.project("q", x))
        k = self._heads(self.project("k", x))
        v = self._heads(self.project("v", x))
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


class V1PretrainAdapter(BaseForecastModel):
    """Stage-II patch forecaster supporting full, Q/K, and Q/K/V adaptation."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
        epsilon: float,
        quantiles: tuple[float, ...],
        adaptation_mode: str,
        adapter_rank: int,
        adapter_alpha: float,
        adapter_dropout: float,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or pred_len <= 0 or epsilon <= 0:
            raise ValueError("output_dim, pred_len, and epsilon must be positive")
        values = tuple(float(value) for value in quantiles)
        if len(values) < 3 or len(values) % 2 == 0:
            raise ValueError("quantiles must have an odd length of at least 3")
        if not all(0.0 < value < 1.0 for value in values) or not all(
            left < right for left, right in zip(values, values[1:])
        ):
            raise ValueError("quantiles must be strictly increasing inside (0, 1)")
        median_index = len(values) // 2
        if abs(values[median_index] - 0.5) > 1.0e-9:
            raise ValueError("the center quantile must be 0.5")
        self.backbone = V1PatchBackbone(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.epsilon = float(epsilon)
        self.quantiles = values
        self.median_index = median_index
        self.output_projection = nn.Linear(
            d_model, pred_len * output_dim * len(values)
        )
        self.adaptation_mode = ""
        self._adapter_settings = (adapter_rank, adapter_alpha, adapter_dropout)
        self.configure_adaptation(adaptation_mode)

    def configure_adaptation(self, mode: str) -> None:
        if mode not in {"full", "qk", "qkv"}:
            raise ValueError("adaptation mode must be 'full', 'qk', or 'qkv'")
        names = set() if mode == "full" else ({"q", "k"} if mode == "qk" else {"q", "k", "v"})
        rank, alpha, dropout = self._adapter_settings
        for layer in self.backbone.layers:
            layer.self_attn.configure_adapters(names, rank, alpha, dropout)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(mode == "full")
        self.backbone.mask_token.requires_grad_(False)
        if mode != "full":
            for layer in self.backbone.layers:
                for parameter in layer.self_attn.adapters.parameters():
                    parameter.requires_grad_(True)
        for parameter in self.output_projection.parameters():
            parameter.requires_grad_(True)
        self.adaptation_mode = mode

    def load_pretrained_backbone(self, state: dict[str, Tensor]) -> None:
        current = self.backbone.state_dict()
        expected = {
            name
            for name in current
            if name != "mask_token" and ".adapters." not in name
        }
        received = set(state)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(
                f"Incompatible pretrained backbone keys; missing={missing}, extra={extra}"
            )
        for name in sorted(expected):
            if current[name].shape != state[name].shape:
                raise ValueError(
                    f"Incompatible pretrained backbone shape for {name}: "
                    f"expected {tuple(current[name].shape)}, got {tuple(state[name].shape)}"
                )
        merged = dict(current)
        merged.update(state)
        self.backbone.load_state_dict(merged, strict=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(
                f"x must have shape [B,{self.seq_len},{self.input_dim}], got {tuple(x.shape)}"
            )
        patches = patchify(x, self.backbone.patch_length, self.backbone.patch_stride)
        summary = self.backbone(patches)[:, -1]
        raw = self.output_projection(summary).reshape(
            x.shape[0], self.pred_len, self.output_dim, len(self.quantiles)
        )
        point = raw[..., self.median_index]
        left: list[Tensor] = []
        current = point
        for index in range(self.median_index - 1, -1, -1):
            current = current - F.softplus(raw[..., index]) - self.epsilon
            left.append(current)
        left.reverse()
        right: list[Tensor] = []
        current = point
        for index in range(self.median_index + 1, len(self.quantiles)):
            current = current + F.softplus(raw[..., index]) + self.epsilon
            right.append(current)
        return torch.stack((*left, point, *right), dim=-1)

    def parameter_summary(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": trainable / total if total else 0.0,
        }
