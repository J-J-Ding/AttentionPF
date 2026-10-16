from copy import deepcopy

import pytest
import torch

from models import build_model
from test_v2_integration import config_and_checkpoint
from utils.seed import set_seed


def build_pair(config,rank):
    base_config=deepcopy(config)
    base_config['model']['name']='v1'
    base_config.pop('horizon_attention',None)
    set_seed(2026)
    base=build_model('v1',base_config,7,1)
    fair_config=deepcopy(config)
    fair_config['model']['name']='v2_fair'
    fair_config['horizon_attention'].update(mode='full_rank' if rank==8 else 'low_rank',rank=rank,cache_kv=False)
    set_seed(2026)
    fair=build_model('v2_fair',fair_config,7,1)
    return base,fair


@pytest.mark.parametrize('rank', [2,4,8])
def test_fair_non_attention_initial_parameters_identical(etth1_path,tmp_path,rank):
    from v2_fair_ablation import non_attention_manifest, assert_fair_initialization
    config=config_and_checkpoint(etth1_path,tmp_path)
    base,fair=build_pair(config,rank)
    assert non_attention_manifest(base)==non_attention_manifest(fair)
    assert_fair_initialization(base,fair)
    assert base.output_projection.weight.shape==fair.output_projection.weight.shape==(6,8)
    delta=fair.parameter_summary()['trainable_params']-base.parameter_summary()['trainable_params']
    assert delta==2*8*rank+8*8+2*8


def test_fair_decoder_equals_original_v1_when_context_repeated(etth1_path,tmp_path):
    config=config_and_checkpoint(etth1_path,tmp_path)
    base,fair=build_pair(config,4)
    base.eval(); fair.eval()
    from dataload.pretrain import patchify
    x=torch.randn(3,16,7)
    with torch.no_grad():
        history=base.backbone(patchify(x,8,8))
        repeated=history[:,-1:, :].expand(-1,2,-1)
        torch.testing.assert_close(fair.decode_context(repeated),base(x),rtol=1e-5,atol=1e-6)


def test_fair_attention_backward_and_non_crossing(etth1_path,tmp_path):
    config=config_and_checkpoint(etth1_path,tmp_path)
    _,fair=build_pair(config,4)
    prediction=fair(torch.randn(3,16,7))
    assert prediction.shape==(3,2,1,3)
    assert (prediction[...,0]<prediction[...,1]).all()
    assert (prediction[...,1]<prediction[...,2]).all()
    prediction.square().mean().backward()
    for name,p in fair.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
        else:
            assert p.grad is None,name


def test_fair_gate_rejects_changed_head(etth1_path,tmp_path):
    from v2_fair_ablation import assert_fair_initialization
    config=config_and_checkpoint(etth1_path,tmp_path)
    base,fair=build_pair(config,4)
    with torch.no_grad():
        fair.output_projection.bias.add_(1)
    with pytest.raises(ValueError,match='initialization'):
        assert_fair_initialization(base,fair)


def test_fair_configs_only_differ_in_attention(project_root):
    from v2_fair_ablation import load_fair_configs, assert_fair_configs
    configs=load_fair_configs(project_root/'configs/diagnosis')
    assert [c['model']['name'] for c in configs]==['v1','v2_fair','v2_fair','v2_fair']
    assert [c.get('horizon_attention',{}).get('rank') for c in configs]==[None,128,64,32]
    assert_fair_configs(configs)
    configs[2]['optimizer']['lr']=.1
    with pytest.raises(ValueError,match='optimizer'):
        assert_fair_configs(configs)
