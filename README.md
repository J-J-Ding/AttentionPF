# AttentionPF

## V2: Horizon-Aware Temporal Attention

V2 loads the V1 Stage-I patch backbone and keeps its `full/qk/qkv` adaptation
rules. Mean-pooled history plus a learned embedding for each forecast horizon
forms queries; Q/K use a relation space (`R=32` by default), while Values stay
`D` dimensional. A shared per-horizon head returns strictly ordered
`[Lower, Point, Upper]` predictions with shape `[B,P,C_out,3]`.

```powershell
python run.py --model v2 --config configs/v2.yaml
python v2_ablation.py --v1-config configs/v1.yaml --v2-config configs/v2.yaml
```

Set `pretrain.checkpoint` to a compatible V1 Stage-I best checkpoint. The checked-in
V2 configuration matches the formal V1 ETTh1 M run (`L=P=96`, seed 2026,
`adaptation.mode=qkv`). Full-rank QK uses `mode: full_rank, rank: 128`;
`use_horizon_embedding: false` removes horizon identity. All other shared settings
must match for controlled ablations. The runner validates them before training.

Each run saves its resolved config, metrics, predictions and standard figures in
`results/<experiment_id>/`. V2 additionally saves the sample-weighted mean attention
map `[P,N]` as `attention_map.npy`. The ablation runner writes the comparison to
`reports/V2_result.md` after all four runs succeed; the standard V2 and low-rank
rows share one experiment ID. Individual summaries stay in each experiment's
`report.md`, preserving `reports/V1_result.md`.

KV cache is an inference optimization. In `model.eval()` and `torch.inference_mode()`,
use `cache = model.build_inference_cache(x)` followed by
`model.predict_from_cache(cache, horizon_indices)` with unique 1D LongTensor indices
on the model device. A cache belongs to its history snapshot and model instance;
rebuild it for every new input window or weight update. Cached tensors are not
saved in checkpoints. Timing metadata compares repeated horizon chunks over a fixed
encoded history and includes cache construction separately; it does not imply that
chunking is faster than one vectorized all-horizon forward.

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

