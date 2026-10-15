"""V0 encoder-only Transformer with a structural non-crossing head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.base_model import BaseForecastModel


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds positional capacity "
                f"{self.encoding.shape[1]}"
            )
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class V0Transformer(BaseForecastModel):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        activation: str,
        epsilon: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or seq_len <= 0 or pred_len <= 0:
            raise ValueError("input_dim, output_dim, seq_len, and pred_len must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.epsilon = float(epsilon)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = SinusoidalPositionEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, pred_len * output_dim * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have rank 3 [B,L,C], got shape {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"x sequence length must be {self.seq_len}, got {x.shape[1]}"
            )
        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"x input channels must be {self.input_dim}, got {x.shape[2]}"
            )
        hidden = self.input_projection(x)
        hidden = self.position_encoding(hidden)
        hidden = self.encoder(hidden)
        summary = hidden[:, -1, :]
        raw = self.output_projection(summary).reshape(
            x.shape[0], self.pred_len, self.output_dim, 3
        )
        point = raw[..., 0]
        lower = point - F.softplus(raw[..., 1]) - self.epsilon
        upper = point + F.softplus(raw[..., 2]) + self.epsilon
        return torch.stack((lower, point, upper), dim=-1)

