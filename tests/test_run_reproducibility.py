from copy import deepcopy

import torch

import run
from utils.config import load_config


def test_execute_seeds_before_model_construction(project_root, monkeypatch):
    captured = []

    def capture(model, *args, **kwargs):
        captured.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return {"paths": {"report": project_root / "unused.md"}}

    monkeypatch.setattr(run, "run_experiment", capture)
    monkeypatch.setattr(run, "render_v0_report", lambda *args: None)
    base = load_config(project_root / "configs/v0.yaml")
    base["experiment"].update({"device": "cpu", "output_root": str(project_root)})
    base["model"].update({"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16})

    torch.manual_seed(1)
    run.execute("v0", deepcopy(base))
    torch.manual_seed(999)
    run.execute("v0", deepcopy(base))

    assert captured[0].keys() == captured[1].keys()
    assert all(torch.equal(captured[0][key], captured[1][key]) for key in captured[0])
