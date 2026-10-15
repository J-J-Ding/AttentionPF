# V0 Transformer Interval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable V0 experiment framework and an encoder-only Transformer that produces structurally non-crossing prediction intervals on the provided ETTh1 dataset.

**Architecture:** ETTh1 is loaded, chronologically split, standardized from train rows only, and windowed into a shared `DataBundle`. A version-independent experiment engine trains any registered `BaseForecastModel`; V0 embeds timesteps, encodes the history, reads the final state, and applies a center-plus-positive-width head. Configuration, metrics, checkpoints, logs, predictions, and reports are reusable by V1–V5.

**Tech Stack:** Python 3.10+, PyTorch, pandas, NumPy, PyYAML, pytest

**Spec:** `specs/V0_project_and_baseline_spec.md`

## Global Constraints

- Input is `[B, L, C_in]`; target is `[B, P, C_out]`; prediction is `[B, P, C_out, 3]`.
- Prediction index order is exactly `[Lower, Point, Upper]`.
- Use `Point = raw_point`, `Lower = Point - softplus(raw_lower_gap) - epsilon`, and `Upper = Point + softplus(raw_upper_gap) + epsilon` with `epsilon = 1.0e-6` by default.
- Never use sorting, value swapping, or corrective post-processing to enforce interval order.
- M uses all input columns as targets; MS uses only the already-last input column and preserves its channel dimension.
- The `date` column is used for ordering and validation only, never as model input.
- Split chronologically; fit the scaler on train rows only; validation/test targets must remain within their own split.
- Default quantiles are `0.05`, `0.50`, and `0.95`, representing a 90% interval.
- V0 uses timestep embedding, standard Transformer Encoder layers, final-timestep readout, and one post-encoder linear forecast head.
- All versions must reuse `epx_main.py`; model code contains no training loop or data-loading logic.
- Main command is `python run.py --model v0 --config configs/v0.yaml`.
- Tests and smoke execution use the provided real `ETTh1.csv`; do not invent a synthetic forecasting dataset.
- Runtime artifacts are written to unique experiment directories and never overwrite prior experiment outputs.

## Review Focus

- A split too short for `seq_len + pred_len` must fail with the split name and required length; Task 3 pins this behavior.
- MS configuration whose declared target is not the last feature must fail before training; Task 2 pins this behavior.
- Very negative raw gap logits must still produce strictly ordered finite intervals on ETTh1-scaled values; Task 4 pins this behavior.
- Inverse scaling `[B,P,1,3]` in MS must use only the final feature's statistics without accidental broadcasting; Task 2 pins this behavior.
- A checkpoint saved on one device must load through an explicit `map_location`, and an incompatible model/output shape must fail clearly; Task 6 pins this behavior.

---

## File Map

- `datasets/raw/ETTh1.csv`: provided source dataset in its canonical project location.
- `datasets/README.md`: expected ETT schema, target ordering, and data-placement rules.
- `dataload/scaler.py`: serializable train-only standardizer.
- `dataload/dataset.py`: CSV validation, chronological split boundaries, and window dataset.
- `dataload/dataloader.py`: seeded loaders and `DataBundle` construction.
- `models/base_model.py`: common forecast model contract.
- `models/v0_transformer.py`: timestep embedding, Transformer Encoder, and non-crossing head.
- `models/__init__.py`: model registry and construction.
- `losses/quantile_loss.py`: strict-shape pinball loss.
- `losses/interval_loss.py`: optional interval score loss extension point.
- `metrics/point_metrics.py`: MAE, MSE, and RMSE.
- `metrics/interval_metrics.py`: PICP, MPIW, PINAW, coverage error, and interval score.
- `utils/config.py`: YAML loading and strict semantic validation.
- `utils/seed.py`: process and DataLoader reproducibility.
- `utils/logger.py`: console/file/JSONL logging and experiment directories.
- `utils/checkpoint.py`: atomic save, best/last state, and portable restore.
- `epx_main.py`: shared train, validate, test, predict, and experiment lifecycle.
- `run.py`: CLI and dependency assembly only.
- `configs/v0.yaml`: executable ETTh1 MS default configuration.
- `reports/V0_result.md`: generated latest-V0 human-readable report.
- `tests/`: real-ETTh1 unit, integration, overfit, and smoke tests.

### Task 1: Project skeleton and canonical ETTh1 placement

**Files:**
- Move: `ETTh1.csv` -> `datasets/raw/ETTh1.csv`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `datasets/README.md`
- Create: package `__init__.py` files under `dataload/`, `models/`, `losses/`, `metrics/`, and `utils/`
- Create: `tests/conftest.py`
- Test: `tests/test_project_structure.py`

**Interfaces:**
- Consumes: provided root-level `ETTh1.csv` with columns `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`.
- Produces: pytest fixture `etth1_path() -> pathlib.Path`; canonical dataset path `datasets/raw/ETTh1.csv`.

- [ ] **Step 1: Write the project-structure test**

Assert the canonical CSV exists, the old root path does not, the header exactly matches the eight provided columns, every required package imports, and runtime directories exist.

- [ ] **Step 2: Run the structure test and verify it fails**

Run: `pytest tests/test_project_structure.py -v`

Expected: FAIL because the canonical project structure does not yet exist.

- [ ] **Step 3: Create the directory/package skeleton and move the verified CSV**

Before moving, resolve both source and destination as paths inside the workspace, verify the source filename and CSV header, then move it to `datasets/raw/ETTh1.csv`. Create empty runtime directories with tracked `.gitkeep` files and ignore generated contents.

- [ ] **Step 4: Add dependencies and minimal project/data documentation**

Pin compatible minimum versions in `requirements.txt`; document that users supply ETT-style CSV files and that MS targets must already be the final numeric feature.

- [ ] **Step 5: Run the structure test**

Run: `pytest tests/test_project_structure.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the skeleton when Git is available**

```bash
git add .gitignore requirements.txt README.md datasets dataload models losses metrics utils tests/test_project_structure.py tests/conftest.py
git commit -m "chore: initialize V0 project structure"
```

## Task 2: Strict configuration and train-only scaler

**Files:**
- Create: `configs/v0.yaml`
- Create: `utils/config.py`
- Create: `dataload/scaler.py`
- Test: `tests/test_config.py`
- Test: `tests/test_scaler.py`

**Interfaces:**
- Consumes: YAML path and NumPy arrays whose final axis is feature/channel.
- Produces: `load_config(path: str | Path) -> dict`; `validate_config(config: dict) -> None`; `StandardScaler.fit(data)`, `transform(data)`, `inverse_transform(data, channel_indices=None)`, `state_dict()`, and `load_state_dict(state)`.

- [ ] **Step 1: Write failing config tests**

Test the real `configs/v0.yaml` contract plus failures for unknown keys, invalid ratios, nonpositive lengths, `d_model % n_heads != 0`, invalid quantile order/coverage, M target mismatch, and MS target not equal to `feature_cols[-1]`.

- [ ] **Step 2: Run config tests and verify failure**

Run: `pytest tests/test_config.py -v`

Expected: FAIL because config loading and validation do not exist.

- [ ] **Step 3: Implement strict YAML loading and validation**

Use the exact config groups and defaults from the Spec. Unknown keys raise `ValueError` with their dotted path; do not silently normalize an invalid M/MS declaration.

- [ ] **Step 4: Run config tests**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing scaler tests against ETTh1**

Fit only the first chronological 70% of ETTh1 numeric rows; assert train means are near zero, state round-trips, full M inverse transform restores all channels, and MS `[B,P,1,3]` inverse transform uses feature index `-1` for every interval member without shape expansion.

- [ ] **Step 6: Run scaler tests and verify failure**

Run: `pytest tests/test_scaler.py -v`

Expected: FAIL because `StandardScaler` does not exist.

- [ ] **Step 7: Implement `StandardScaler`**

Store float64 statistics for fitting stability, transform to requested floating dtype, guard zero variance with `epsilon_s`, and support explicit channel indices for M/MS inverse transforms.

- [ ] **Step 8: Run config and scaler tests**

Run: `pytest tests/test_config.py tests/test_scaler.py -v`

Expected: PASS.

- [ ] **Step 9: Commit config and scaler**

```bash
git add configs/v0.yaml utils/config.py dataload/scaler.py tests/test_config.py tests/test_scaler.py
git commit -m "feat: add strict V0 config and scaler"
```

## Task 3: ETTh1 chronological datasets and DataLoaders

**Files:**
- Create: `dataload/dataset.py`
- Create: `dataload/dataloader.py`
- Modify: `dataload/__init__.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: validated `config["data"]` and seed.
- Produces: `TimeSeriesWindowDataset(values, target_start, target_end, seq_len, pred_len, stride, output_indices)`; `DataBundle(train_loader, val_loader, test_loader, scaler, input_dim, output_dim, split_metadata)`; `create_dataloaders(config: dict) -> DataBundle`.

- [ ] **Step 1: Write failing ETTh1 schema and chronological split tests**

Assert parsed timestamps are unique/increasing after ordering, numeric values are finite, split target ranges are disjoint and exhaustive, scaler statistics equal only the train rows, and the date column never appears in tensors.

- [ ] **Step 2: Write failing M/MS window tests**

With ETTh1 and short configured windows, assert exact dataset lengths from boundary formulas; M items are `[L,7]`/`[P,7]`, MS items are `[L,7]`/`[P,1]`, and MS `y` equals the scaled future OT slice `[:, -1:]`.

- [ ] **Step 3: Write failing boundary and error tests**

Assert validation/test may use up to `seq_len` prior rows only as history, every target index stays inside its split, and a split shorter than `seq_len + pred_len` raises an error naming the split and required length. Derive any malformed-file fixture from a small copy of real ETTh1 rows rather than generating a synthetic series.

- [ ] **Step 4: Run dataset tests and verify failure**

Run: `pytest tests/test_dataset.py -v`

Expected: FAIL because dataset and loader construction do not exist.

- [ ] **Step 5: Implement CSV validation, split metadata, and window indexing**

Represent each sample by its forecast start index; compute history as `[forecast_start-L:forecast_start]` and forecast as `[forecast_start:forecast_start+P]`. Restrict forecast starts to each split, while allowing prior-split history at validation/test boundaries.

- [ ] **Step 6: Implement reproducible DataLoaders**

Shuffle train only. Use a seeded `torch.Generator` and worker seeding utility. Return dimensions and scaler in `DataBundle` so `run.py` contains no data-mode branches.

- [ ] **Step 7: Run dataset tests**

Run: `pytest tests/test_dataset.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the data pipeline**

```bash
git add dataload tests/test_dataset.py
git commit -m "feat: add leakage-safe ETTh1 data pipeline"
```

## Task 4: Base model and structurally non-crossing V0 Transformer

**Files:**
- Create: `models/base_model.py`
- Create: `models/v0_transformer.py`
- Modify: `models/__init__.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `x: torch.Tensor[B,L,C_in]` and validated model/interval dimensions.
- Produces: `BaseForecastModel.forward(x) -> Tensor[B,P,C_out,3]`; `V0Transformer`; `MODEL_REGISTRY`; `build_model(name, config, input_dim, output_dim) -> BaseForecastModel`.

- [ ] **Step 1: Write failing M/MS forward and registry tests**

Instantiate small V0 models and assert M output `[2,24,7,3]`, MS output `[2,24,1,3]`, registry construction works, parameter count is positive, and wrong rank/channel inputs fail clearly.

- [ ] **Step 2: Write failing non-crossing and gradient tests**

Assert finite outputs satisfy strict elementwise ordering, including a head fed very negative gap logits on the ETTh1-scaled numeric range; assert backward produces finite gradients. Add a source/behavior guard showing the forward path does not call sort or exchange bounds.

- [ ] **Step 3: Run model tests and verify failure**

Run: `pytest tests/test_model.py -v`

Expected: FAIL because the model classes do not exist.

- [ ] **Step 4: Implement `BaseForecastModel` and sinusoidal encoding**

Use `batch_first=True`; register positional values as a nonpersistent or state-safe buffer; validate rank, sequence length, and channel count.

- [ ] **Step 5: Implement `V0Transformer` and post-encoder interval head**

Apply `Linear(C_in,d_model)`, positional encoding, standard encoder layers, `H[:, -1, :]`, and `Linear(d_model,P*C_out*3)`. Interpret raw slot zero as point and slots one/two as lower/upper gaps before stacking `[lower, point, upper]`.

- [ ] **Step 6: Implement model registry/building**

Registry lookup rejects unknown names with available choices and passes only explicit dimensions/hyperparameters into the model.

- [ ] **Step 7: Run model tests**

Run: `pytest tests/test_model.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the V0 model**

```bash
git add models tests/test_model.py
git commit -m "feat: add non-crossing V0 transformer"
```

## Task 5: Quantile losses and original-scale metrics

**Files:**
- Create: `losses/quantile_loss.py`
- Create: `losses/interval_loss.py`
- Modify: `losses/__init__.py`
- Create: `metrics/point_metrics.py`
- Create: `metrics/interval_metrics.py`
- Modify: `metrics/__init__.py`
- Test: `tests/test_losses.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: prediction `[B,P,C,3]`, target `[B,P,C]`, and nominal coverage.
- Produces: `QuantileLoss(quantiles).forward(pred, target) -> scalar`; `interval_score_loss`; point/interval metric functions; `compute_metrics(pred, target, coverage) -> dict` with `overall` and `by_horizon`.

- [ ] **Step 1: Write failing pinball and interval-loss tests**

Compare small hand-calculated tensors, assert exact zero for an exact standalone loss input, reject accidental broadcasting, and verify finite gradients.

- [ ] **Step 2: Run loss tests and verify failure**

Run: `pytest tests/test_losses.py -v`

Expected: FAIL because losses do not exist.

- [ ] **Step 3: Implement strict-shape losses**

Map quantiles `0.05/0.50/0.95` to output slots `0/1/2`; keep optional interval loss separate and disabled by default.

- [ ] **Step 4: Write failing metric tests**

Assert hand-calculated MAE/MSE/RMSE, closed-boundary PICP, MPIW, PINAW, Coverage Error, and Winkler score; require overall plus exactly `P` horizon entries; verify weighted aggregation and explicit zero-range PINAW behavior.

- [ ] **Step 5: Run metric tests and verify failure**

Run: `pytest tests/test_metrics.py -v`

Expected: FAIL because metrics do not exist.

- [ ] **Step 6: Implement metric functions**

Compute from complete concatenated original-scale arrays to avoid mean-of-batch-means errors. Return JSON-serializable floats/lists and a documented `null` PINAW for zero target range.

- [ ] **Step 7: Run loss and metric tests**

Run: `pytest tests/test_losses.py tests/test_metrics.py -v`

Expected: PASS.

- [ ] **Step 8: Commit losses and metrics**

```bash
git add losses metrics tests/test_losses.py tests/test_metrics.py
git commit -m "feat: add interval losses and metrics"
```

## Task 6: Reproducibility, logging, and portable checkpoints

**Files:**
- Create: `utils/seed.py`
- Create: `utils/logger.py`
- Create: `utils/checkpoint.py`
- Modify: `utils/__init__.py`
- Test: `tests/test_utils.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: resolved config, experiment identity, model/optimizer/scheduler/scaler states.
- Produces: `set_seed(seed, deterministic)`, `seed_worker(worker_id)`, `create_experiment_paths(config)`, `setup_logger(paths)`, `save_checkpoint(path, state)`, and `load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu")`.

- [ ] **Step 1: Write failing seed/path/logger tests**

Assert repeated CPU random sequences match, experiment paths are unique, repeated logger setup does not duplicate handlers, and JSONL records parse as JSON.

- [ ] **Step 2: Write failing checkpoint tests**

Assert atomic save leaves no temp file, model/scaler/config/RNG round-trip, explicit `map_location` is honored, incompatible format version fails, and loading a checkpoint into a different output shape raises a clear compatibility error.

- [ ] **Step 3: Run utility tests and verify failure**

Run: `pytest tests/test_utils.py tests/test_checkpoint.py -v`

Expected: FAIL because utilities do not exist.

- [ ] **Step 4: Implement seed and logger utilities**

Capture Python, NumPy, PyTorch CPU/CUDA seeds; log environment/device metadata; create unique timestamp-plus-suffix paths without overwriting existing runs.

- [ ] **Step 5: Implement checkpoint utilities**

Use a same-directory temporary file followed by `os.replace`; store `format_version=1`; validate model name and relevant config dimensions before applying state dictionaries.

- [ ] **Step 6: Run utility tests**

Run: `pytest tests/test_utils.py tests/test_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 7: Commit experiment utilities**

```bash
git add utils tests/test_utils.py tests/test_checkpoint.py
git commit -m "feat: add reproducible experiment utilities"
```

## Task 7: Shared experiment engine

**Files:**
- Create: `epx_main.py`
- Test: `tests/test_experiment.py`
- Test: `tests/test_overfit.py`

**Interfaces:**
- Consumes: registered model, `DataBundle` loaders/scaler, validated config, optimizer, and loss.
- Produces: `train_one_epoch(...) -> dict`; `validate(...) -> dict`; `test(...) -> dict`; `predict(...) -> np.ndarray`; `run_experiment(model, train_loader, val_loader, test_loader, config, scaler) -> dict`.

- [ ] **Step 1: Write failing single-epoch lifecycle tests on ETTh1**

Use deterministic subsets created by selecting windows from the real ETTh1-backed Dataset. Assert train updates parameters; validate/test do not; output shape is retained; nonfinite predictions/loss fail immediately; and validation metrics are computed after inverse scaling.

- [ ] **Step 2: Run lifecycle tests and verify failure**

Run: `pytest tests/test_experiment.py -v`

Expected: FAIL because `epx_main.py` does not exist.

- [ ] **Step 3: Implement train/validate/test/predict**

Use `model.train()` only in training and `torch.inference_mode()` elsewhere. Validate every output shape, aggregate loss by target element count, clip gradients by config, and concatenate original-scale predictions/targets before metrics.

- [ ] **Step 4: Write the failing one-batch overfit test**

Take one real ETTh1 MS batch, disable dropout, fix the seed, train a small V0 model repeatedly, and assert final quantile loss is materially below its initial value while all predictions remain non-crossing.

- [ ] **Step 5: Run the overfit test and verify its initial failure**

Run: `pytest tests/test_overfit.py -v -m slow`

Expected: FAIL until the optimizer/lifecycle connection is correct.

- [ ] **Step 6: Implement `run_experiment`**

Create optimizer/scheduler from config, track `val_loss`, atomically save `best.pt`/`last.pt`, stop by patience, restore `best.pt`, test exactly once, save `.npz`, and return timings, memory, parameter counts, metrics, and paths.

- [ ] **Step 7: Run engine and overfit tests**

Run: `pytest tests/test_experiment.py tests/test_overfit.py -v -m "slow or not slow"`

Expected: PASS.

- [ ] **Step 8: Commit the shared engine**

```bash
git add epx_main.py tests/test_experiment.py tests/test_overfit.py
git commit -m "feat: add shared experiment lifecycle"
```

## Task 8: CLI assembly, report generation, and real ETTh1 smoke run

**Files:**
- Create: `run.py`
- Create: `reports/V0_result.md`
- Create: `tests/test_cli.py`
- Create: `tests/test_smoke_experiment.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `--model`, `--config`, optional `--device`, and optional `--resume-from`.
- Produces: exit status, experiment artifact paths, `results/<experiment_id>/metrics.json`, `predictions.npz`, and `reports/V0_result.md`.

- [ ] **Step 1: Write failing CLI tests**

Assert argument parsing, model/config name agreement, unknown model error choices, and that assembly delegates to `create_dataloaders`, `build_model`, and `run_experiment` without embedding training branches.

- [ ] **Step 2: Implement CLI and report renderer**

Keep `run.py` limited to parsing, validated construction, and delegation. Render the V0 report from the returned structured result, including total/trainable parameters, peak GPU memory or N/A, train time, inference latency, MAE, RMSE, PICP, MPIW, PINAW, Coverage Error, Interval Score, and by-horizon metrics.

- [ ] **Step 3: Run CLI tests**

Run: `pytest tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 4: Write and run the real ETTh1 smoke test**

Run one CPU epoch with a small model and bounded train/validation/test window subsets from `datasets/raw/ETTh1.csv`; assert best checkpoint reload, final test/prediction shapes, strict interval order, metrics JSON, prediction NPZ, JSONL logs, config snapshot, and report fields.

Run: `pytest tests/test_smoke_experiment.py -v -m slow`

Expected: PASS within a practical CPU test duration.

- [ ] **Step 5: Run the documented main command on ETTh1**

Run: `python run.py --model v0 --config configs/v0.yaml`

Expected: completes Load Data -> Train -> Validation -> Save Best Checkpoint -> Test -> Prediction -> Metrics -> Save Results, using `datasets/raw/ETTh1.csv`.

- [ ] **Step 6: Complete README usage and artifact documentation**

Document environment setup, ETTh1 schema/order, M/MS selection, main command, output locations, test commands, deterministic limitations, and how later model registry entries reuse the framework.

- [ ] **Step 7: Commit the runnable V0 baseline**

```bash
git add run.py README.md reports/V0_result.md tests/test_cli.py tests/test_smoke_experiment.py
git commit -m "feat: deliver runnable V0 ETTh1 baseline"
```

## Task 9: Full verification and acceptance evidence

**Files:**
- Modify only files whose tests expose defects.

**Interfaces:**
- Consumes: complete V0 project.
- Produces: passing test suite and inspectable ETTh1 experiment artifacts.

- [ ] **Step 1: Run fast tests**

Run: `pytest -v -m "not slow"`

Expected: all tests pass.

- [ ] **Step 2: Run slow/real-data tests**

Run: `pytest -v -m slow`

Expected: one-batch overfit and smoke tests pass.

- [ ] **Step 3: Run the complete suite together**

Run: `pytest -v`

Expected: all tests pass with no warnings that indicate shape broadcasting, data leakage, duplicate logging, or unsafe checkpoint loading.

- [ ] **Step 4: Inspect generated artifacts**

Verify resolved config, `best.pt`, `last.pt`, JSONL log, metrics JSON, predictions NPZ, and `reports/V0_result.md`; load the NPZ and assert `[N,P,C_out,3]` plus strict non-crossing.

- [ ] **Step 5: Check project constraints mechanically**

Search model code for sorting/swap patterns, search Python files for hard-coded absolute data paths, and confirm the only training loop is in `epx_main.py`.

- [ ] **Step 6: Record final environment and verification commands in the handoff**

Include exact passed commands, test counts, ETTh1 experiment ID, runtime, and any platform-specific deterministic caveat.

- [ ] **Step 7: Commit verification fixes if Git is available**

```bash
git add -A
git commit -m "test: verify V0 ETTh1 baseline"
```

