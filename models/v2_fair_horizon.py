"""Isolated HA ablation retaining every original V1 forecast-head parameter."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.v1_pretrain_adapter import V1PretrainAdapter
from models.v2_horizon_attention import HorizonTemporalAttention, _finite


class V1HorizonAttention(V1PretrainAdapter):
    """Original V1 backbone/adapters/head, with only temporal readout replaced."""

    def __init__(self, *, relation_rank: int, use_horizon_embedding: bool,
                 cache_kv: bool = False, **options) -> None:
        # Initialize all V1 parameters before HA consumes any random numbers.
        super().__init__(**options)
        self.horizon_attention=HorizonTemporalAttention(d_model=self.backbone.d_model,
            pred_len=self.pred_len,relation_rank=relation_rank,
            use_horizon_embedding=use_horizon_embedding)
        if cache_kv:
            raise ValueError('fair ablation uses vectorized readout with cache_kv=false')
        self.cache_kv=False

    def encode_history(self, x: Tensor) -> Tensor:
        from dataload.pretrain import patchify
        if x.ndim!=3 or x.shape[1:]!=(self.seq_len,self.input_dim) or x.shape[0]==0:
            raise ValueError('input must match V1 history contract')
        _finite(x,'input')
        return self.backbone(patchify(x,self.backbone.patch_length,self.backbone.patch_stride))

    def decode_context(self, context: Tensor) -> Tensor:
        if context.ndim!=3 or context.shape[1:]!=(self.pred_len,self.backbone.d_model):
            raise ValueError('fair context must have shape [B,P,D]')
        _finite(context,'horizon context')
        # These are views of the original Linear(D,P*C*3) parameters, not new heads.
        weight=self.output_projection.weight.view(self.pred_len,self.output_dim*3,-1)
        bias=self.output_projection.bias.view(self.pred_len,self.output_dim*3)
        raw=(torch.einsum('bpd,pcd->bpc',context,weight)+bias).reshape(
            context.shape[0],self.pred_len,self.output_dim,3)
        point=raw[...,0]
        lower=point-F.softplus(raw[...,1])-self.epsilon
        upper=point+F.softplus(raw[...,2])+self.epsilon
        prediction=torch.stack((lower,point,upper),-1)
        _finite(prediction,'prediction')
        return prediction

    def forward_with_attention(self, x: Tensor) -> tuple[Tensor,Tensor]:
        context,attention=self.horizon_attention(self.encode_history(x))
        return self.decode_context(context),attention

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_attention(x)[0]
