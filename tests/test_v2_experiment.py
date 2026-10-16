from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataload.scaler import StandardScaler
from epx_main import _evaluate, build_loss
from run import render_report
from test_v2_config import valid_v2_config
from test_v2_model import tiny_model


def test_attention_aggregation_weights_unequal_batches():
    model=tiny_model().eval()
    x=torch.randn(5,16,7)
    target=torch.zeros(5,4,1)
    loader=DataLoader(TensorDataset(x,target),batch_size=2)
    config=valid_v2_config()
    scaler=StandardScaler().fit(np.stack([np.zeros(7),np.ones(7)]))
    with torch.inference_mode():
        _,weights=model.forward_with_attention(x)
    result=_evaluate(model,loader,build_loss(config),scaler,torch.device('cpu'),config,
                     collect_attention=True)
    np.testing.assert_allclose(result['attention_map'],weights.mean(0).numpy(),atol=1e-7)
    plain=_evaluate(model,loader,build_loss(config),scaler,torch.device('cpu'),config)
    assert 'attention_map' not in plain


def test_cache_benchmark_is_separate_from_backbone_and_accuracy():
    model=tiny_model().eval()
    result=model.benchmark_kv_cache(torch.randn(2,16,7),
            [torch.tensor([0,1]),torch.tensor([2,3])],repeats=2)
    assert result['chunks']==2
    for key in ('cache_build_seconds','uncached_chunk_seconds','cached_chunk_seconds'):
        assert np.isfinite(result[key]) and result[key]>=0


def test_v2_report_has_relation_cache_and_provenance(tmp_path):
    metrics=dict(mae=1.,rmse=2.,picp=.9,mpiw=3.,pinaw=.1,coverage_error=0.,interval_score=4.)
    result=dict(model_name='v2',experiment_id='v2-run',best_epoch=1,best_val_loss=.1,
        total_params=100,trainable_params=20,trainable_ratio=.2,peak_gpu_memory_bytes=None,
        train_seconds=1.,inference_seconds=.1,inference_latency_seconds=.01,
        pretrain_checkpoint='pretrain/best.pt',pretrain_checkpoint_id='pretrain',
        adaptation_mode='qkv',adapter_rank=8,adapter_alpha=16.,
        horizon_attention_mode='low_rank',horizon_attention_rank=32,
        horizon_attention_use_horizon_embedding=True,horizon_attention_cache_kv=True,
        kv_cache_benchmark=dict(cache_build_seconds=.1,cached_chunk_seconds=.1,uncached_chunk_seconds=.2,chunks=2),
        test_metrics=dict(overall=metrics,by_horizon=[dict(horizon=1,**metrics)]),
        paths=dict(best_checkpoint='best.pt',resolved_config='config.yaml'))
    render_report(result,tmp_path/'V2_result.md')
    report=(tmp_path/'V2_result.md').read_text(encoding='utf-8')
    for word in ('V2','Horizon attention','low_rank','Horizon embedding','KV cache',
                 'Adapter rank','By Horizon','Pretrain checkpoint'):
        assert word in report
