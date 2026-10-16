import math
from unittest.mock import patch

import pytest
import torch


def attention(dim=8, rank=3, pred=4, embedding=True):
    from models.v2_horizon_attention import HorizonTemporalAttention
    return HorizonTemporalAttention(d_model=dim, pred_len=pred, relation_rank=rank,
                                    use_horizon_embedding=embedding)


@pytest.mark.parametrize('batch,tokens,dim,rank,pred', [(2,5,8,3,4), (1,1,4,2,1)])
def test_horizon_shapes_pooling_and_normalization(batch,tokens,dim,rank,pred):
    module = attention(dim,rank,pred)
    history = torch.randn(batch,tokens,dim)
    keys, values = module.project_history(history)
    torch.testing.assert_close(module.pool_history(history), history.mean(1))
    assert keys.shape == (batch,tokens,rank)
    assert values.shape == (batch,tokens,dim)
    assert module.build_queries(history.mean(1)).shape == (batch,pred,rank)
    with patch.object(module, 'attend', wraps=module.attend) as spy:
        context, weights = module(history)
        assert spy.call_count == 1
    assert context.shape == (batch,pred,dim)
    assert weights.shape == (batch,pred,tokens)
    assert torch.all(weights >= 0)
    torch.testing.assert_close(weights.sum(-1), torch.ones(batch,pred))
    assert all(layer.bias is None for layer in
               (module.query_projection,module.key_projection,module.value_projection))


def test_attention_matches_manual_scaled_dot_product():
    module = attention(2,2,1)
    query = torch.tensor([[[1.,2.]]])
    keys = torch.tensor([[[1.,0.],[0.,1.]]])
    values = torch.tensor([[[3.,4.],[7.,8.]]])
    context, weights = module.attend(query,keys,values)
    first = 1 / (1 + math.exp(1 / math.sqrt(2)))
    torch.testing.assert_close(weights, torch.tensor([[[first,1-first]]]))
    torch.testing.assert_close(context, torch.tensor([[[3*first+7*(1-first),4*first+8*(1-first)]]]))


def test_horizon_embedding_is_only_horizon_identity():
    torch.manual_seed(7)
    module = attention()
    torch.manual_seed(7)
    other = attention()
    torch.testing.assert_close(module.horizon_embedding,other.horizon_embedding)
    history = torch.randn(2,5,8)
    before = module.build_queries(history.mean(1)).detach().clone()
    with torch.no_grad():
        module.horizon_embedding[1].add_(3)
    after = module.build_queries(history.mean(1))
    torch.testing.assert_close(before[:,0], after[:,0])
    assert not torch.allclose(before[:,1], after[:,1])
    no_embedding = attention(embedding=False)
    assert no_embedding.horizon_embedding is None
    assert 'horizon_embedding' not in no_embedding.state_dict()
    contexts,weights = no_embedding(history)
    torch.testing.assert_close(contexts,contexts[:,:1].expand_as(contexts))
    torch.testing.assert_close(weights,weights[:,:1].expand_as(weights))


@pytest.mark.parametrize('indices', [torch.tensor([]),torch.tensor([0,0]),torch.tensor([-1]),
    torch.tensor([4]),torch.tensor([1.]),torch.tensor([[0]])])
def test_bad_horizon_indices_rejected(indices):
    with pytest.raises(ValueError,match='horizon'):
        attention()(torch.randn(2,5,8),indices)


def test_selected_horizons_keep_request_order():
    module = attention()
    history = torch.randn(2,5,8)
    full,weights = module(history)
    selected,selected_weights = module(history,torch.tensor([3,0]))
    torch.testing.assert_close(selected,full[:,[3,0]])
    torch.testing.assert_close(selected_weights,weights[:,[3,0]])


@pytest.mark.parametrize('history', [torch.zeros(2,8),torch.zeros(0,5,8),
    torch.zeros(2,0,8),torch.zeros(2,5,7),torch.full((2,5,8),float('nan')),
    torch.full((2,5,8),float('inf'))])
def test_invalid_history_rejected(history):
    with pytest.raises((ValueError,FloatingPointError)):
        attention()(history)


def test_full_rank_uses_more_parameters_than_low_rank():
    low,full=attention(8,3),attention(8,8)
    assert sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in low.parameters()) == 80
