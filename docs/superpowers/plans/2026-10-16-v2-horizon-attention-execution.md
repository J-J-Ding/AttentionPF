# V2 execution record

Plan: `2026-10-16-v2-horizon-attention.md`.
Spec: `specs/V2_horizon_attention_spec.md`.
Branch: `feature/v2`; implementation base: `687ed3c`.

## Verified implementation

- Tasks 1–7 implemented and committed individually; each new component had an observed RED test followed by GREEN.
- Initial full suite: 228 passed in 35.60 seconds.
- Fresh whole-branch review: no critical findings; one important cache invalidation issue fixed.
- Cache-fix suite: **232 passed in 38.25 seconds**, including V0/V1/V2 smoke, checkpoint contracts, real optimizer cache invalidation and repeated deterministic CPU training.
- Formal ablation completed on RTX 4070 Laptop GPU: four independent runs with ETTh1 M, L=P=96, seed 2026, the shared Stage-I checkpoint and QKV adaptation. Five comparison rows are recorded in `reports/V2_result.md` (standard V2 and low-rank share one run).
- Saved predictions for every run were `[3389,96,7,3]`, finite and strictly non-crossing. Metrics were recomputed from saved artifacts and agreed to numerical precision; all 96 horizons, resolved configs, checkpoint epochs, scaler statistics and parameter counts were checked. V2 attention maps were `[96,11]`, finite and row-normalized. The no-embedding variant had identical predictions at every horizon, as specified.
- Cache timings were measured again with the optimizer-invalidation fix; model test evaluation was not repeated.
- Final suite after the Windows report fix: **233 passed in 44.61 seconds**. Normal sandbox read access to the generated report was confirmed; `results/` contains no test-generated directories.

## Decisions and deviations

1. Used PowerShell/apply_patch progress bookkeeping in this Windows worktree instead of bash-only skill scripts. The same task/commit/test record is retained here; risk is bookkeeping portability only.
2. Added an optional V1 output projection factory with unchanged default behavior. V2 therefore reuses V1 initialization without constructing/discarding its flattened head, as required by the spec. V1 baseline reproduction and regression tests protect compatibility; risk is the small additional constructor API.
3. Strengthened cache ownership with a per-instance token and parameter identity/version signatures. After review demonstrated fused AdamW bypassing Tensor version counters, added optimizer-step generations in a weak-reference registry. Normal and fused optimizer steps invalidate old caches, fresh caches remain usable, and unrelated optimizers do not invalidate them. Risk is dependence on PyTorch's optimizer hook API; no parameter tensors are copied per horizon request.
4. Split new tests into focused model/experiment/ablation files. Migrated existing output-producing test fixtures to pytest temporary directories, preserving production experiment artifacts.
5. Ran targeted task tests and a complete suite, then reran the complete suite after the review fix. Avoided three duplicate broad runs; all required test categories executed.
6. Explicit cache calls remain FP32/default-dtype APIs with strict device/dtype checks; mixed precision is outside the current training workflow. A future mixed-precision caller may require additional compatibility work.
7. This experiment uses one fixed seed, as planned. It supports a single-seed comparison and cannot establish statistical significance.
8. The original report writer used a secure temporary directory, whose Windows ACL prevented sandbox writes and made its atomically replaced report inaccessible. Reproduced with a failure-injection regression test; refactored shared formatting into `format_report` and use a unique exclusive destination-sibling file for atomic replacement with inherited workspace permissions. Existing four-run artifacts were reused, avoiding retraining or extra test-set evaluations.

## Review outcome

The independent reviewer reproduced stale cache acceptance after fused AdamW with a prediction discrepancy of 0.147475. A real fused-optimizer regression test failed before the fix and passed after it; all 232 tests passed afterward.

One defensive report-validation improvement is deferred: the report validator checks contiguous horizon numbering against the supplied list length rather than an expected prediction length, and does not require cache timing metadata when caching is enabled. The normal experiment pipeline supplies complete data; final formal artifacts are audited independently against `pred_len=96` and cache metadata before delivery. A later caller supplying truncated synthetic results could require stronger validation.
