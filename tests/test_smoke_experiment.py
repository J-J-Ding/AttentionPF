from copy import deepcopy

import numpy as np
import pytest

from run import execute
from pretrain import run_pretraining
from utils.config import load_config


pytestmark = pytest.mark.slow


def test_real_etth1_end_to_end_smoke(project_root, tmp_path):
    config = deepcopy(load_config(project_root / "configs/v0.yaml"))
    config["experiment"].update(
        {
            "name": "v0_smoke",
            "device": "cpu",
            "output_root": str(tmp_path),
        }
    )
    config["data"].update(
        {"seq_len": 16, "pred_len": 2, "stride": 500, "batch_size": 4,
         "features": "MS", "target_cols": ["OT"]}
    )
    config["model"].update(
        {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0}
    )
    config["training"].update({"epochs": 1, "early_stopping_patience": 1})

    result = execute("v0", config)

    paths = result["paths"]
    for key in (
        "best_checkpoint",
        "last_checkpoint",
        "metrics_json",
        "predictions_npz",
        "resolved_config",
        "log",
        "metrics_jsonl",
        "report",
    ):
        assert paths[key].is_file(), key
    saved = np.load(paths["predictions_npz"])
    prediction = saved["predictions"]
    assert prediction.ndim == 4
    assert prediction.shape[1:] == (2, 1, 3)
    assert np.all(prediction[..., 0] < prediction[..., 1])
    assert np.all(prediction[..., 1] < prediction[..., 2])
    report = paths["report"].read_text(encoding="utf-8")
    assert "V0 ETTh1 Interval Forecast Result" in report
    assert "By Horizon" in report
    assert paths["figures_dir"] == paths["metrics_json"].parent / "figures"
    assert len(paths["figures"]) == 6
    assert all(path.is_file() for path in paths["figures"].values())


def test_real_etth1_v1_two_stage_smoke(project_root, tmp_path):
    config = deepcopy(load_config(project_root / "configs" / "v1.yaml"))
    config["experiment"].update(
        {"name": "v1_smoke", "device": "cpu", "output_root": str(tmp_path)}
    )
    config["data"].update(
        {"seq_len": 16, "pred_len": 2, "stride": 500, "batch_size": 4,
         "features": "MS", "target_cols": ["OT"]}
    )
    config["patch"].update({"length": 8, "stride": 8})
    config["model"].update(
        {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0}
    )
    config["adaptation"].update({"mode": "qk", "rank": 2, "alpha": 4.0})
    config["pretrain"].update({"epochs": 1, "early_stopping_patience": 1})
    config["training"].update({"epochs": 1, "early_stopping_patience": 1})
    config["visualization"].update({"formats": ["png"], "dpi": 50})

    stage_i = run_pretraining(config)
    config["pretrain"]["checkpoint"] = str(stage_i["paths"]["best_checkpoint"])
    stage_ii = execute("v1", config)

    assert stage_i["paths"]["best_checkpoint"].is_file()
    assert stage_ii["paths"]["report"].name == "V1_result.md"
    assert stage_ii["paths"]["report"].is_file()
    saved = np.load(stage_ii["paths"]["predictions_npz"])["predictions"]
    assert saved.shape[1:] == (2, 1, 3)
    assert np.all(saved[..., 0] < saved[..., 1])
    assert np.all(saved[..., 1] < saved[..., 2])


def test_real_etth1_v2_horizon_attention_smoke(project_root, tmp_path):
    config=deepcopy(load_config(project_root/'configs/v1.yaml'))
    config['experiment'].update(name='v2_smoke',device='cpu',output_root=str(tmp_path))
    config['data'].update(seq_len=16,pred_len=2,stride=500,batch_size=4,
                          features='MS',target_cols=['OT'])
    config['model'].update(d_model=8,n_heads=2,n_layers=1,d_ff=16,dropout=0.)
    config['patch'].update(length=8,stride=8)
    config['adaptation'].update(mode='qkv',rank=2,alpha=4.)
    config['pretrain'].update(checkpoint=None,epochs=1,early_stopping_patience=1)
    config['training'].update(epochs=1,early_stopping_patience=1)
    config['visualization'].update(formats=['png'],dpi=50)
    stage_i=run_pretraining(config)
    config['pretrain']['checkpoint']=str(stage_i['paths']['best_checkpoint'])
    config['model']['name']='v2'
    config['horizon_attention']=dict(mode='low_rank',rank=3,use_horizon_embedding=True,cache_kv=True)
    result=execute('v2',config)
    prediction=np.load(result['paths']['predictions_npz'])['predictions']
    assert prediction.shape[1:]==(2,1,3)
    assert np.all(prediction[...,0]<prediction[...,1])
    assert np.all(prediction[...,1]<prediction[...,2])
    weights=np.load(result['paths']['attention_map'])
    assert weights.shape==(2,2)
    np.testing.assert_allclose(weights.sum(-1),1.,atol=1e-6)
    assert result['paths']['report'].name=='V2_result.md'
    assert 'Horizon attention' in result['paths']['report'].read_text(encoding='utf-8')
    assert result['kv_cache_benchmark']['chunks']==1
