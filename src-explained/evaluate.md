# `src/evaluate.py`

## What this module is for
Holdout evaluation with uncertainty attached: bootstrap intervals, paired comparisons,
calibration, threshold metrics, lift and per-segment performance. The notebook's §6 calls it.

## Public surface
- `holdout_metrics(y, p) -> {auc, pr_auc, brier, ece}`.
- `bootstrap_ci(y, p, metric=roc_auc_score, n_boot=2000) -> (point, low, high)`.
- `paired_delta(y, p_new, p_old, metric, n_boot) -> {delta, ci_low, ci_high, p_not_better}`.
- `calibration_table(y, p, bins=10) -> DataFrame(predicted, actual, n)`; `expected_calibration_error(y, p) -> float`.
- `at_threshold(y, p, thr) -> dict` (confusion counts, precision, recall).
- `lift_table(y, p, bins=10) -> DataFrame` per decile: churn rate, lift, cumulative share of churners.
- `segment_performance(X, y, p, by) -> DataFrame`: AUC and mean prediction against actual churn per level.

## Walkthrough
`_resample_idx` is the core. It draws stratified bootstrap resamples, sampling churners and
stayers separately with replacement and keeping both counts. `bootstrap_ci` scores one model on
each resample. `paired_delta` scores both models on the *same* resample and keeps the difference.
`p_not_better` is the share of resamples where the new model does not win, a bootstrap analogue of
a one-sided p-value.

## The non-obvious bits
- **Stratified resampling** guarantees both classes in every resample, so AUC is always defined.
  It slightly understates variance compared with a plain bootstrap; with 407 churners the
  difference is negligible.
- **Same seed for both models** in `paired_delta`: that is what makes it paired.
- `calibration_table` uses `qcut` on predictions (equal-count bins) with `duplicates="drop"`, so
  heavily tied predictions don't crash it.
- ECE weights each bin by its size; with equal-count bins that is a plain average.

## How it can break
`segment_performance` returns NaN AUC for a segment with only one class instead of raising. Tests
pin that identical predictions give a zero delta with a zero-width interval, and that perfectly
calibrated predictions give ECE = 0.
