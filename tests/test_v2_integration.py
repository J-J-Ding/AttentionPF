from copy import deepcopy

import pytest
import torch

from models import build_model
from test_v1_integration import _config, _write_stage_i_checkpoint
from utils.checkpoint import validate_checkpoint_contract, save_checkpoint, load_checkpoint


def config_and_checkpoint(etth1_path, tmp_path):
    config = _config(etth1_path, tmp_path, 'qkv')
    checkpoint = tmp_path / 'pretrain' / 'best.pt'
    _write_stage_i_checkpoint(checkpoint, config)
    config['pretrain']['checkpoint'] = str(checkpoint)
    config['model']['name'] = 'v2'
    config['horizon_attention'] = dict(mode='low_rank',rank=3,use_horizon_embedding=True,cache_kv=True)
    return config


def test_v2_registry_loads_v1_pretrain_and_roundtrips(etth1_path,tmp_path):
    config=config_and_checkpoint(etth1_path,tmp_path)
    model=build_model('v2',config,7,1).eval()
    saved=torch.load(config['pretrain']['checkpoint'],weights_only=False)
    for key,value in saved['backbone_state_dict'].items():
        torch.testing.assert_close(model.backbone.state_dict()[key],value)
    assert model.pretrain_provenance['horizon_attention_rank']==3
    assert model.pretrain_scaler_state is not None
    x=torch.randn(2,16,7)
    path=tmp_path/'supervised.pt'
    save_checkpoint(path,dict(format_version=1,model_name='v2',config=config,model_state_dict=model.state_dict()))
    other=build_model('v2',config,7,1).eval()
    load_checkpoint(path,other,expected_config=config,expected_model_name='v2',restore_rng=False)
    torch.testing.assert_close(model(x),other(x))


@pytest.mark.parametrize('section,key,value', [
    ('data','pred_len',3),('model','d_model',16),('horizon_attention','rank',4),
    ('horizon_attention','mode','full_rank'),('horizon_attention','use_horizon_embedding',False),
    ('patch','stride',4),('adaptation','alpha',8),('pretrain','checkpoint','other.pt'),
    ('data','features','M')])
def test_v2_checkpoint_rejects_semantic_changes(etth1_path,tmp_path,section,key,value):
    saved=config_and_checkpoint(etth1_path,tmp_path)
    changed=deepcopy(saved)
    changed[section][key]=value
    with pytest.raises(ValueError,match=section):
        validate_checkpoint_contract(saved,changed)


def test_v2_requires_explicit_compatible_pretrain(etth1_path,tmp_path):
    config=config_and_checkpoint(etth1_path,tmp_path)
    config['pretrain']['checkpoint']=None
    with pytest.raises(ValueError,match='pretrain.checkpoint'):
        build_model('v2',config,7,1)
    config['pretrain']['checkpoint']=str(tmp_path/'missing.pt')
    with pytest.raises(FileNotFoundError):
        build_model('v2',config,7,1)
