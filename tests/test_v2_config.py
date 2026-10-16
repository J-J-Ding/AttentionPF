from copy import deepcopy

import pytest

from test_v1_config import valid_v1_config
from utils.config import load_config, validate_config


def valid_v2_config():
    config = deepcopy(valid_v1_config())
    config['model'].update(name='v2', d_model=128)
    config['horizon_attention'] = dict(mode='low_rank', rank=32,
                                      use_horizon_embedding=True, cache_kv=True)
    return config


def test_v2_config_and_full_rank_are_valid(project_root):
    config = valid_v2_config()
    validate_config(config)
    config['horizon_attention'].update(mode='full_rank', rank=128)
    validate_config(config)
    assert load_config(project_root / 'configs/v2.yaml')['data']['pred_len'] == 96


@pytest.mark.parametrize('field,value', [
    ('mode', 'bad'), ('rank', 0), ('rank', 128), ('rank', 129),
    ('rank', True), ('rank', 3.5), ('rank', '32'),
    ('use_horizon_embedding', 1), ('cache_kv', 'true'), ('surprise', 1),
])
def test_v2_rejects_invalid_attention_config(field, value):
    config = valid_v2_config()
    config['horizon_attention'][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_v2_rejects_missing_and_inconsistent_rank():
    config = valid_v2_config()
    config['horizon_attention']['mode'] = 'full_rank'
    with pytest.raises(ValueError, match='rank'):
        validate_config(config)
    del config['horizon_attention']['rank']
    with pytest.raises(ValueError, match='rank'):
        validate_config(config)
