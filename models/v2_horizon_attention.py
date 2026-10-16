"""Horizon-specific temporal readout over the pretrained V1 patch backbone."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataload.pretrain import patchify
from models.v1_pretrain_adapter import V1PretrainAdapter


def _finite(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'{name} contains non-finite values')


class HorizonTemporalAttention(nn.Module):
    """Single vectorized temporal attention in a Q/K relation space."""

    def __init__(self, *, d_model: int, pred_len: int, relation_rank: int,
                 use_horizon_embedding: bool) -> None:
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (d_model, pred_len, relation_rank)):
            raise ValueError('d_model, pred_len and relation_rank must be positive integers')
        if relation_rank > d_model:
            raise ValueError('relation_rank must not exceed d_model')
        if type(use_horizon_embedding) is not bool:
            raise ValueError('use_horizon_embedding must be boolean')
        self.d_model, self.pred_len, self.relation_rank = d_model, pred_len, relation_rank
        self.query_projection = nn.Linear(d_model, relation_rank, bias=False)
        self.key_projection = nn.Linear(d_model, relation_rank, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.register_parameter('horizon_embedding', None)
        if use_horizon_embedding:
            self.horizon_embedding = nn.Parameter(torch.empty(pred_len, d_model))
            nn.init.trunc_normal_(self.horizon_embedding, std=0.02)

    def _validate_history(self, history: Tensor) -> None:
        if history.ndim != 3 or history.shape[-1] != self.d_model or min(history.shape) <= 0:
            raise ValueError(f'history must be nonempty [B,N,{self.d_model}]')
        if not history.is_floating_point():
            raise ValueError('history must be floating point')
        _finite(history, 'history')

    def _normalize_horizon_indices(self, indices: Tensor | None, device: torch.device) -> Tensor:
        if indices is None:
            return torch.arange(self.pred_len, device=device)
        if (not isinstance(indices, Tensor) or indices.ndim != 1 or indices.numel() == 0
                or indices.dtype != torch.long or indices.device != device):
            raise ValueError('horizon indices must be a nonempty 1D LongTensor on the query device')
        if (indices < 0).any() or (indices >= self.pred_len).any() or indices.unique().numel() != indices.numel():
            raise ValueError('horizon indices must be unique and in [0,pred_len)')
        return indices

    def pool_history(self, history: Tensor) -> Tensor:
        self._validate_history(history)
        context = history.mean(dim=1)
        _finite(context, 'global context')
        return context

    def project_history(self, history: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_history(history)
        keys, values = self.key_projection(history), self.value_projection(history)
        _finite(keys, 'historical keys')
        _finite(values, 'historical values')
        return keys, values

    def build_queries(self, global_context: Tensor,
                      horizon_indices: Tensor | None = None) -> Tensor:
        if global_context.ndim != 2 or global_context.shape[-1] != self.d_model or global_context.shape[0] == 0:
            raise ValueError(f'global context must be [B,{self.d_model}]')
        _finite(global_context, 'global context')
        indices = self._normalize_horizon_indices(horizon_indices, global_context.device)
        queries = global_context[:, None, :].expand(-1, indices.numel(), -1)
        if self.horizon_embedding is not None:
            queries = queries + self.horizon_embedding[indices].to(dtype=global_context.dtype)
        queries = self.query_projection(queries)
        _finite(queries, 'horizon queries')
        return queries

    def attend(self, queries: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        if (queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3
                or queries.shape[-1] != self.relation_rank or keys.shape[-1] != self.relation_rank
                or values.shape[-1] != self.d_model or keys.shape[:2] != values.shape[:2]
                or queries.shape[0] != keys.shape[0]
                or any(min(t.shape) <= 0 for t in (queries, keys, values))):
            raise ValueError('query/key/value shapes must be [B,P,R], [B,N,R], [B,N,D]')
        if any(t.device != queries.device or t.dtype != queries.dtype for t in (keys, values)):
            raise ValueError('query/key/value device and dtype must match')
        for name, tensor in (('queries', queries), ('keys', keys), ('values', values)):
            _finite(tensor, name)
        logits = queries @ keys.transpose(-2, -1) / math.sqrt(self.relation_rank)
        _finite(logits, 'attention logits')
        weights = torch.softmax(logits, dim=-1)
        context = weights @ values
        _finite(weights, 'attention weights')
        _finite(context, 'horizon context')
        return context, weights

    def forward(self, history: Tensor,
                horizon_indices: Tensor | None = None) -> tuple[Tensor, Tensor]:
        keys, values = self.project_history(history)
        queries = self.build_queries(self.pool_history(history), horizon_indices)
        return self.attend(queries, keys, values)


class V2HorizonAttentionForecaster(V1PretrainAdapter):
    """V1 adaptation backbone with horizon-specific temporal readout."""

    def __init__(self, *, relation_rank: int, use_horizon_embedding: bool,
                 cache_kv: bool, **backbone_options) -> None:
        output_dim = backbone_options['output_dim']
        super().__init__(**backbone_options,
                         output_projection_factory=lambda dim, _: nn.Linear(dim, output_dim * 3))
        self.horizon_attention = HorizonTemporalAttention(
            d_model=self.backbone.d_model, pred_len=self.pred_len,
            relation_rank=relation_rank, use_horizon_embedding=use_horizon_embedding)
        self.cache_kv = cache_kv

    def encode_history(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_dim) or x.shape[0] == 0:
            raise ValueError(f'x must be nonempty [B,{self.seq_len},{self.input_dim}]')
        _finite(x, 'input')
        history = self.backbone(patchify(x, self.backbone.patch_length, self.backbone.patch_stride))
        _finite(history, 'backbone history')
        return history

    def decode_context(self, context: Tensor) -> Tensor:
        if context.ndim != 3 or context.shape[-1] != self.backbone.d_model or min(context.shape) <= 0:
            raise ValueError('context must be nonempty [B,P,D]')
        _finite(context, 'horizon context')
        raw = self.output_projection(context).reshape(*context.shape[:2], self.output_dim, 3)
        _finite(raw, 'forecast logits')
        point = raw[..., 0]
        lower = point - F.softplus(raw[..., 1]) - self.epsilon
        upper = point + F.softplus(raw[..., 2]) + self.epsilon
        prediction = torch.stack((lower, point, upper), dim=-1)
        _finite(prediction, 'prediction')
        if not ((lower < point).all() and (point < upper).all()):
            raise FloatingPointError('forecast interval is not strictly ordered at current precision')
        return prediction

    def forward_with_attention(self, x: Tensor,
                               horizon_indices: Tensor | None = None) -> tuple[Tensor, Tensor]:
        context, weights = self.horizon_attention(self.encode_history(x), horizon_indices)
        return self.decode_context(context), weights

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_attention(x)[0]
