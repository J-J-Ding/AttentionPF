import math
from pathlib import Path
import hashlib

import numpy as np
import pytest


def test_inventory_explains_v1_v2_parameter_gap():
    from diagnose_v2 import parameter_inventory
    from models.v1_pretrain_adapter import V1PretrainAdapter
    from models.v2_horizon_attention import V2HorizonAttentionForecaster
    options=dict(input_dim=7,output_dim=7,seq_len=96,pred_len=96,patch_length=16,
        patch_stride=8,d_model=128,n_heads=4,n_layers=3,d_ff=256,dropout=.1,
        activation='gelu',epsilon=1e-6,adaptation_mode='qkv',adapter_rank=8,
        adapter_alpha=16.,adapter_dropout=0.)
    v1=parameter_inventory(V1PretrainAdapter(**options))
    v2=parameter_inventory(V2HorizonAttentionForecaster(**options,relation_rank=32,
        use_horizon_embedding=True,cache_kv=True))
    assert v1['summary']==dict(total=691936,trainable=278496,frozen=413440)
    assert v2['summary']==dict(total=471445,trainable=58005,frozen=413440)
    assert v1['categories']['Forecast Head']['trainable']==260064
    assert v2['categories']['Forecast Head']['trainable']==2709
    assert v1['categories']['Adapter']==v2['categories']['Adapter']
    assert v2['categories']['Horizon Attention']['trainable']==36864
    for inventory in (v1,v2):
        assert sum(row['total'] for row in inventory['categories'].values())==inventory['summary']['total']
        assert len(inventory['parameters'])==len({row['name'] for row in inventory['parameters']})


def test_uniform_attention_diagnostics_match_analytic_values():
    from diagnose_v2 import attention_statistics
    stats=attention_statistics(np.full((2,6,2),.5))
    assert stats['entropy_nats']['mean']==pytest.approx(math.log(2))
    assert stats['entropy_normalized']['mean']==pytest.approx(1)
    assert stats['cosine_off_diagonal']['mean']==pytest.approx(1)
    assert stats['pairwise_tv']['mean']==pytest.approx(0)
    assert stats['flags']['uniform_fraction']==1
    assert stats['flags']['collapse_sample_fraction']==1
    assert stats['flags']['sharp_fraction']==0


def test_sharp_distinct_horizons_do_not_count_as_collapse():
    from diagnose_v2 import attention_statistics
    stats=attention_statistics(np.eye(3)[None,:,:])
    assert stats['entropy_nats']['mean']==0
    assert stats['cosine_off_diagonal']['mean']==0
    assert stats['pairwise_tv']['mean']==1
    assert stats['flags']['sharp_fraction']==1
    assert stats['flags']['collapse_sample_fraction']==0
    assert stats['flags']['uniform_fraction']==0


@pytest.mark.parametrize('values', [np.ones((3,2)),np.zeros((1,3,2)),
    np.full((1,3,2),np.nan),np.array([[[-1.,2.]]])])
def test_bad_attention_diagnostics_fail(values):
    from diagnose_v2 import attention_statistics
    with pytest.raises(ValueError):
        attention_statistics(values)


def test_single_token_attention_is_explicit_degeneracy():
    from diagnose_v2 import attention_statistics
    stats=attention_statistics(np.ones((2,3,1)))
    assert stats['degenerate_single_token'] is True
    assert stats['entropy_normalized']['mean']==0


def test_group_pair_quantiles_preserve_sample_variation():
    from diagnose_v2 import attention_statistics
    a=np.array([[[1.,0.],[1.,0.],[1.,0.]],
                [[1.,0.],[0.,1.],[1.,0.]]])
    stats=attention_statistics(a)
    distribution=stats['group_pairs']['short:middle']['cosine']
    assert distribution['mean']==pytest.approx(.5)
    assert distribution['p10']==pytest.approx(.1)
    assert distribution['p90']==pytest.approx(.9)


def test_source_identity_hashes_the_actual_diagnosis_script():
    from diagnose_v2 import source_identity
    source=Path('diagnose_v2.py').resolve()
    identity=source_identity(source)
    assert identity['source_file']==str(source)
    assert identity['source_file_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(identity['source_commit'])==40
    assert isinstance(identity['source_dirty'],bool)


def test_structural_explanation_uses_actual_inventory_values():
    from diagnose_v2 import structural_explanation
    v1={'summary':{'trainable':100},'categories':{
        'Adapter':{'trainable':10},'Forecast Head':{'trainable':90},
        'Horizon Attention':{'trainable':0}}}
    v2={'summary':{'trainable':85},'categories':{
        'Adapter':{'trainable':10},'Forecast Head':{'trainable':20},
        'Horizon Attention':{'trainable':55}}}
    text=structural_explanation(v1,v2)
    assert 'V1 trainable 100 = Adapter 10 + Head 90.' in text
    assert 'V2 trainable 85 = Adapter 10 + Head 20 + HA 55.' in text
    assert 'Head delta -70; HA delta +55; net delta -15.' in text
