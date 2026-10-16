from copy import deepcopy

import pytest

from utils.config import load_config


def configs():
    return load_config('configs/v1.yaml'),load_config('configs/v2.yaml')


def result_fixture(run_id, model_name, root):
    metrics=dict(mae=1.,rmse=2.,picp=.9,mpiw=3.,pinaw=.1,coverage_error=0.,interval_score=4.)
    paths=dict(best_checkpoint=root/run_id/'best.pt',resolved_config=root/run_id/'config.resolved.yaml',
               metrics_json=root/run_id/'metrics.json')
    return dict(model_name=model_name,experiment_id=run_id,best_epoch=1,best_val_loss=.1,
        total_params=100,trainable_params=20,trainable_ratio=.2,peak_gpu_memory_bytes=None,
        train_seconds=1.,inference_seconds=.1,inference_latency_seconds=.01,
        pretrain_checkpoint='pretrain/best.pt',pretrain_checkpoint_id='pretrain',
        adaptation_mode='qkv',adapter_rank=8,adapter_alpha=16.,
        horizon_attention_mode='low_rank',horizon_attention_rank=32,
        horizon_attention_use_horizon_embedding=True,horizon_attention_cache_kv=True,
        kv_cache_benchmark=dict(cache_build_seconds=.1,cached_chunk_seconds=.1,uncached_chunk_seconds=.2,chunks=2),
        test_metrics=dict(overall=metrics,by_horizon=[dict(horizon=1,**metrics)]),paths=paths)


def test_ablation_has_four_independent_runs_with_isolated_changes():
    from v2_ablation import build_ablation_configs
    v1,v2=configs()
    runs=build_ablation_configs(v1,v2)
    assert len(runs)==4
    assert [model for _,model,_ in runs]==['v1','v2','v2','v2']
    for _,_,config in runs:
        for section in ('data','pretrain','adaptation','training','optimizer','loss'):
            assert config[section]==v2[section]
    assert runs[1][2]['horizon_attention']['rank']==32
    assert runs[2][2]['horizon_attention']['use_horizon_embedding'] is False
    assert runs[3][2]['horizon_attention']['rank']==128
    assert v2['horizon_attention']['use_horizon_embedding'] is True


@pytest.mark.parametrize('section,key,value', [
    ('data','pred_len',24),('adaptation','mode','qk'),('optimizer','lr',.01),
    ('pretrain','checkpoint','other.pt'),('experiment','seed',1)])
def test_mismatched_ablation_configs_fail_before_training(section,key,value):
    from v2_ablation import build_ablation_configs
    v1,v2=configs()
    v1[section][key]=value
    with pytest.raises(ValueError,match=section):
        build_ablation_configs(v1,v2)


def test_ablation_executes_four_times_and_aliases_low_rank(tmp_path,monkeypatch):
    import v2_ablation
    v1,v2=configs()
    for config in (v1,v2):
        config['experiment']['output_root']=str(tmp_path)
    calls=[]
    def execute(model,config,*,render_single_report):
        assert render_single_report is False
        calls.append(config)
        return result_fixture(f'run-{len(calls)}',model,tmp_path)
    monkeypatch.setattr(v2_ablation,'execute',execute)
    results=v2_ablation.run_v2_ablation(v1,v2)
    assert len(calls)==4
    assert results['V2']['experiment_id']==results['V2 low-rank QK']['experiment_id']
    text=(tmp_path/'reports/V2_result.md').read_text(encoding='utf-8')
    assert 'alias' in text
    for label in ('V1','V2','V2 without horizon embedding','V2 full-rank QK','V2 low-rank QK'):
        assert label in text
    assert 'By Horizon' in text and 'Interval Score' in text


def test_ablation_failure_preserves_previous_report(tmp_path,monkeypatch):
    import v2_ablation
    v1,v2=configs()
    for config in (v1,v2):
        config['experiment']['output_root']=str(tmp_path)
    target=tmp_path/'reports/V2_result.md'
    target.parent.mkdir()
    target.write_text('previous report',encoding='utf-8')
    def fail(*args,**kwargs):
        raise RuntimeError('training failed')
    monkeypatch.setattr(v2_ablation,'execute',fail)
    with pytest.raises(RuntimeError):
        v2_ablation.run_v2_ablation(v1,v2)
    assert target.read_text(encoding='utf-8')=='previous report'


def test_report_rejects_missing_metrics_and_bad_alias(tmp_path):
    from v2_ablation import render_v2_ablation_report
    results={label:result_fixture(str(i),'v1' if i==0 else 'v2',tmp_path)
             for i,label in enumerate(('V1','V2','V2 without horizon embedding','V2 full-rank QK'))}
    results['V2 low-rank QK']=results['V2']
    broken=deepcopy(results)
    del broken['V2']['test_metrics']['overall']['mae']
    with pytest.raises(ValueError):
        render_v2_ablation_report(broken,tmp_path/'bad.md')
    broken=deepcopy(results)
    broken['V2 low-rank QK']=result_fixture('wrong','v2',tmp_path)
    with pytest.raises(ValueError,match='alias'):
        render_v2_ablation_report(broken,tmp_path/'bad.md')


def test_report_writes_in_destination_without_secure_temp_directory(tmp_path,monkeypatch):
    import tempfile
    from v2_ablation import render_v2_ablation_report
    results={label:result_fixture(str(i),'v1' if i==0 else 'v2',tmp_path)
             for i,label in enumerate(('V1','V2','V2 without horizon embedding','V2 full-rank QK'))}
    results['V2 low-rank QK']=results['V2']
    def restricted_temp(*args,**kwargs):
        raise PermissionError('secure temporary directories unavailable in sandbox')
    monkeypatch.setattr(tempfile,'TemporaryDirectory',restricted_temp)
    render_v2_ablation_report(results,tmp_path/'V2_result.md')
    assert 'Overall comparison' in (tmp_path/'V2_result.md').read_text(encoding='utf-8')
