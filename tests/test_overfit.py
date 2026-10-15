from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from dataload.dataloader import create_dataloaders
from epx_main import train_one_epoch
from losses.quantile_loss import QuantileLoss
from models import build_model
from utils.config import load_config
from utils.seed import set_seed


pytestmark = pytest.mark.slow


def test_v0_overfits_one_real_etth1_batch(project_root):
    set_seed(42, deterministic=True)
    config = deepcopy(load_config(project_root / "configs/v0.yaml"))
    config["experiment"]["device"] = "cpu"
    config["data"].update({"seq_len": 16, "pred_len": 2, "batch_size": 4})
    config["model"].update(
        {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0.0}
    )
    bundle = create_dataloaders(config)
    loader = DataLoader(Subset(bundle.train_loader.dataset, range(4)), batch_size=4, shuffle=False)
    model = build_model("v0", config, bundle.input_dim, bundle.output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    loss_fn = QuantileLoss((0.05, 0.5, 0.95))
    x, y = next(iter(loader))
    with torch.no_grad():
        initial_loss = loss_fn(model(x), y).item()

    for _ in range(30):
        train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"), config)

    with torch.no_grad():
        prediction = model(x)
        final_loss = loss_fn(prediction, y).item()

    assert final_loss < initial_loss * 0.6
    assert torch.all(prediction[..., 0] < prediction[..., 1])
    assert torch.all(prediction[..., 1] < prediction[..., 2])
