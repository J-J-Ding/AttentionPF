# V1 ETTh1 Masked-Pretraining and Adapter Results

Formal run date: 2026-10-16. All V1 variants use seed 2026, ETTh1 MS mode,
`L=96`, `P=24`, the same chronological split, and the same Stage-I checkpoint.
Predictions were verified finite with shape `[3461,24,1,3]` and strict
`Lower < Point < Upper` for every element.

## Stage I

- Experiment: `v1_etth1_ms_seed2026_20261016-111425-357175_559f59`
- Best checkpoint: `checkpoints/v1_etth1_ms_seed2026_20261016-111425-357175_559f59/best.pt`
- Best epoch: 11 (early stop at epoch 21)
- Best validation reconstruction loss: 0.160457
- Training time: 119.012 s
- Masking: Random 0.40 / Tail 0.25, batch-level random probability 0.50
- Patch contract: length 16, stride 8, no padding

## Overall comparison

| Model | Mode | Total params | Trainable params | Ratio | GPU MiB | Train s | Infer s | MAE | RMSE | PICP | MPIW | PINAW | Coverage error | Interval score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | scratch | 407,752 | 407,752 | 1.0000 | 140.35 | 61.054 | 0.407 | 1.580214 | 2.019042 | 0.907758 | 6.950738 | 0.384443 | 0.007758 | 8.673865 |
| V1-A | full | 422,728 | 422,600 | 0.9997 | 77.38 | 197.445 | 0.518 | 2.092744 | 2.741725 | 0.546603 | 3.609302 | 0.199630 | 0.353397 | 19.652120 |
| V1-B | qk | 435,016 | 21,576 | 0.0496 | 72.60 | 60.740 | 0.623 | 1.773217 | 2.265113 | 0.922277 | 8.457443 | 0.467779 | 0.022277 | 9.948425 |
| V1-C | qkv | 441,160 | 27,720 | 0.0628 | 72.76 | 66.953 | 0.579 | 1.644857 | 2.178360 | 0.946595 | 8.036197 | 0.444480 | 0.046595 | 9.270189 |

V1-B trains 4.96% and V1-C 6.28% of their parameters. Among V1 variants,
V1-C has the best point errors and interval score; V1-B is closer to the
nominal 90% coverage. V0 remains stronger on this single seed for MAE, RMSE,
coverage error, and interval score, so these results do not support claiming
that V1 universally improves the baseline.

## Selected horizons

The table samples early, middle, and final horizons. Complete 24-horizon values
are retained in each `metrics.json` and visualized in
`figures/horizon_metrics.*` under the experiment directory.

| Mode | Horizon | MAE | RMSE | PICP | PINAW | Interval score |
|---|---:|---:|---:|---:|---:|---:|
| full | 1 | 1.328743 | 1.721658 | 0.740537 | 0.205466 | 8.757151 |
| full | 12 | 2.174413 | 2.811070 | 0.496099 | 0.192399 | 21.078778 |
| full | 24 | 2.485824 | 3.211957 | 0.499856 | 0.219135 | 25.273603 |
| qk | 1 | 1.420553 | 1.779419 | 0.888183 | 0.350373 | 8.152073 |
| qk | 12 | 2.127384 | 2.630421 | 0.863045 | 0.446294 | 11.044871 |
| qk | 24 | 2.063792 | 2.602294 | 0.929500 | 0.526570 | 11.056001 |
| qkv | 1 | 1.323238 | 1.665746 | 0.943080 | 0.327302 | 6.726811 |
| qkv | 12 | 1.565866 | 2.054523 | 0.944814 | 0.438798 | 9.144994 |
| qkv | 24 | 1.858759 | 2.456599 | 0.934990 | 0.491985 | 10.394950 |

## Artifact provenance

| Variant | Experiment ID | Best epoch | Best val loss | Metrics and figures |
|---|---|---:|---:|---|
| V1-A | `v1_full_etth1_ms_seed2026_20261016-111645-834608_743b85` | 21 | 0.068073 | `results/v1_full_etth1_ms_seed2026_20261016-111645-834608_743b85/` |
| V1-B | `v1_qk_etth1_ms_seed2026_20261016-112011-142591_50cbac` | 1 | 0.052823 | `results/v1_qk_etth1_ms_seed2026_20261016-112011-142591_50cbac/` |
| V1-C | `v1_qkv_etth1_ms_seed2026_20261016-112115-465185_1d832f` | 1 | 0.069949 | `results/v1_qkv_etth1_ms_seed2026_20261016-112115-465185_1d832f/` |

Adapter hyperparameters are rank 8, alpha 16, and dropout 0.0. All three runs
reuse the same forecast head structure, optimizer budget, early stopping,
metrics, checkpoint implementation, and plotting code.
