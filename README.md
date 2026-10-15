# AttentionPF

AttentionPF is a reproducible time-series interval forecasting project. V0
provides the shared experiment framework and an encoder-only Transformer whose
head structurally guarantees `Lower < Point < Upper` without sorting.

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
metrics JSON, and prediction NPZ.

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

See `specs/V0_project_and_baseline_spec.md` for the complete V0 contract.

