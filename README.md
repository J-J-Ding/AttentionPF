# AttentionPF

AttentionPF is a reproducible time-series interval forecasting project. V0
provides the shared experiment framework and an encoder-only Transformer. V1
adds masked patch pretraining and full/QK/QKV low-rank adaptation. Every model
structurally guarantees `Lower < Point < Upper` without sorting.

## Environment

The project is tested with the Conda environment `py312`:

```powershell
conda activate py312
pip install -r requirements.txt
```

## Data

The supplied dataset is stored at `datasets/raw/ETTh1.csv`. Its numeric columns
must remain ordered as:

```text
HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
```

`date` is used only to sort and validate time. In the default `MS` mode, all
seven numeric columns are model inputs and the already-last column, `OT`, is
the only target. Set `data.features: M` and make `target_cols` equal
`feature_cols` to run multivariate-to-multivariate forecasting.

## Run V0

```powershell
python run.py --model v0 --config configs/v0.yaml
```

Optional runtime overrides:

```powershell
python run.py --model v0 --config configs/v0.yaml --device cpu
python run.py --model v0 --config configs/v0.yaml --resume-from checkpoints/<experiment_id>/last.pt
```

Each run creates unique subdirectories under `logs/`, `checkpoints/`, and
`results/`. The latest human-readable summary is written to
`reports/V0_result.md`; each result directory retains its resolved config,
metrics JSON, prediction NPZ, and publication-ready figures:

```text
results/<experiment_id>/
├── config.resolved.yaml
├── metrics.json
├── predictions.npz
└── figures/
    ├── training_curves.png/.pdf
    ├── forecast_interval.png/.pdf
    └── horizon_metrics.png/.pdf
```

PNG files use the configured publication DPI (300 by default), while PDF
files are vector graphics suitable for paper typesetting. Configure figure
generation in `configs/v0.yaml` under `visualization`; set `enabled: false`
to skip it. `channel_index` selects the plotted target in M mode. MS mode has
one output channel, so its default value is `0`.

## Run V1

V1 has two explicit stages. Stage I uses only windows from the official train
split and internally divides them chronologically into pretrain train/validation
subsets. Random Mask and Tail Mask reconstruct masked patches with a single
linear head:

```powershell
python pretrain.py --config configs/v1.yaml
```

The command prints the selected `checkpoints/<pretrain_id>/best.pt` path. Put
that exact path in `pretrain.checkpoint` in a copied V1 config before Stage II;
forecasting refuses to start without it. Stage II always uses the shared
`epx_main.py` training, validation, testing, metrics, checkpoint, logging, and
figure pipeline:

```powershell
python run.py --model v1 --config configs/v1.yaml
```

Choose the ablation only through `adaptation.mode`:

- `full`: V1-A, full fine-tuning without adapters.
- `qk`: V1-B, frozen backbone with Q/K adapters.
- `qkv`: V1-C, frozen backbone with Q/K/V adapters.

Keep the same Stage I checkpoint, seed, data split, optimizer budget, and all
non-adaptation settings when comparing these modes. Each supervised run writes
all artifacts under one experiment-specific `results/` directory and updates
`reports/V1_result.md`. The report records pretraining provenance, mode/rank/
alpha, parameter counts and ratio, timing, GPU memory, overall metrics, and
metrics by forecast horizon.

## Tests

After activating `py312`:

```powershell
python -m pytest tests -v -p no:cacheprovider
python -m pytest tests -v -m "not slow" -p no:cacheprovider
python -m pytest tests -v -m slow -p no:cacheprovider
```

Tests use the real ETTh1 dataset. The slow group includes a one-batch overfit
check and an end-to-end smoke experiment.

Deterministic algorithms and all relevant random seeds are enabled by default.
Exact GPU reproducibility still depends on PyTorch and CUDA support for the
selected device and operators.

## Extending models

Later versions add a model module, configuration, and one registry entry in
`models/__init__.py`. They reuse `epx_main.py`, data loading, losses, metrics,
logging, checkpoints, and output protocol unchanged.

See `specs/V0_project_and_baseline_spec.md` and
`specs/V1_pretrain_adapter_spec.md` for the complete version contracts.

