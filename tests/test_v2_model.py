import pytest
import torch


def tiny_model(mode='qkv', output=1, pred=4, embedding=True):
    from models.v2_horizon_attention import V2HorizonAttentionForecaster
    return V2HorizonAttentionForecaster(input_dim=7,output_dim=output,seq_len=16,
        pred_len=pred,patch_length=8,patch_stride=8,d_model=8,n_heads=2,n_layers=1,
        d_ff=16,dropout=0.,activation='gelu',epsilon=1e-6,adaptation_mode=mode,
        adapter_rank=2,adapter_alpha=4.,adapter_dropout=0.,relation_rank=3,
        use_horizon_embedding=embedding,cache_kv=True)


@pytest.mark.parametrize('output,pred', [(1,4),(7,4),(1,1),(7,1)])
def test_v2_output_shape_and_structural_intervals(output,pred):
    model=tiny_model(output=output,pred=pred)
    prediction=model(torch.randn(1,16,7))
    assert prediction.shape == (1,pred,output,3)
    assert torch.isfinite(prediction).all()
    assert (prediction[...,0]<prediction[...,1]).all()
    assert (prediction[...,1]<prediction[...,2]).all()
    assert model.output_projection.out_features == output*3


@pytest.mark.parametrize('mode', ['full','qk','qkv'])
def test_v2_gradients_respect_adapter_boundary(mode):
    model=tiny_model(mode)
    model(torch.randn(2,16,7)).square().mean().backward()
    for name,param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None,name
        else:
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
    assert all(p.requires_grad for p in model.horizon_attention.parameters())
    assert all(p.requires_grad for p in model.output_projection.parameters())


def test_v2_without_embedding_has_no_horizon_bypass():
    model=tiny_model(embedding=False).eval()
    prediction,weights=model.forward_with_attention(torch.randn(2,16,7))
    torch.testing.assert_close(prediction,prediction[:,:1].expand_as(prediction))
    torch.testing.assert_close(weights,weights[:,:1].expand_as(weights))
    assert not any('horizon_embedding' in key for key in model.state_dict())


def test_v2_embedding_changes_horizon_predictions():
    model=tiny_model().eval()
    with torch.no_grad():
        model.horizon_attention.horizon_embedding[1].add_(10)
    prediction,weights=model.forward_with_attention(torch.randn(2,16,7))
    assert not torch.allclose(weights[:,0],weights[:,1])
    assert not torch.allclose(prediction[:,0],prediction[:,1])


@pytest.mark.parametrize('x', [torch.zeros(2,16,6),torch.zeros(0,16,7),
    torch.full((2,16,7),float('nan'))])
def test_v2_bad_input_rejected(x):
    with pytest.raises((ValueError,FloatingPointError)):
        tiny_model()(x)


def test_forecast_cache_equivalence_and_no_persistence():
    model=tiny_model().eval()
    x=torch.randn(2,16,7)
    with torch.inference_mode():
        expected=model(x)
        cache=model.build_inference_cache(x)
        first,_=model.predict_from_cache(cache,torch.tensor([0,1]))
        second,_=model.predict_from_cache(cache,torch.tensor([2,3]))
        torch.testing.assert_close(torch.cat([first,second],1),expected)
    assert not any('cache' in key for key in model.state_dict())


@pytest.mark.parametrize('part', ['backbone','horizon_attention','output_projection'])
def test_forecast_cache_rejects_updated_weights(part):
    model=tiny_model().eval()
    with torch.no_grad():
        cache=model.build_inference_cache(torch.randn(2,16,7))
        next(getattr(model,part).parameters()).add_(.1)
        with pytest.raises(ValueError,match='stale'):
            model.predict_from_cache(cache)


def test_forecast_cache_rejects_reloaded_weights():
    model=tiny_model().eval()
    with torch.no_grad():
        cache=model.build_inference_cache(torch.randn(2,16,7))
        model.load_state_dict(model.state_dict())
        with pytest.raises(ValueError,match='stale'):
            model.predict_from_cache(cache)


@pytest.mark.parametrize('fused', [False,True])
def test_cache_invalidated_by_real_adamw_step(fused):
    model=tiny_model().eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,fused=fused)
    x=torch.randn(2,16,7)
    with torch.no_grad():
        cache=model.build_inference_cache(x)
    model(x).square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        with pytest.raises(ValueError,match='stale'):
            model.predict_from_cache(cache)
        fresh=model.build_inference_cache(x)
        prediction,_=model.predict_from_cache(fresh)
        torch.testing.assert_close(prediction,model(x))


def test_unrelated_optimizer_does_not_invalidate_cache():
    model=tiny_model().eval()
    parameter=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([parameter],fused=True)
    x=torch.randn(2,16,7)
    with torch.no_grad():
        cache=model.build_inference_cache(x)
    parameter.square().sum().backward()
    optimizer.step()
    with torch.no_grad():
        prediction,_=model.predict_from_cache(cache)
        torch.testing.assert_close(prediction,model(x))
