# `src/models.py`

## What this module is for
The model pipelines, cross-validation and tuning. It holds two families on purpose: the source
notebook reproduced (`source_*`) and this project's versions. It uses `config` and
`features.engineer`; the notebook calls it in §4–6.

## Public surface
- `folds() -> StratifiedKFold`: 5 folds, shuffled, seed 42, shared by every model.
- `source_preprocessor()`, `source_baselines() -> {name: Pipeline}`: LR, RF, LightGBM, as in the source.
- `tree_preprocessor(engineered, exclude=())`, `tree_pipeline(clf, engineered=True, exclude=()) -> Pipeline`.
- `lgbm(**params)`, `xgb(**params)`: classifiers with the project's fixed defaults.
- `scorecard_pipeline() -> Pipeline`: spline logistic regression.
- `cv_evaluate(pipe, X, y) -> {per_fold, mean, std, oof}`; `cv_table(results) -> DataFrame`.
- `lgbm_search_space(trial) -> dict`; `tune_lgbm(X, y, n_trials=40, engineered=True) -> (params, study)`.

## Walkthrough
**Source family.** `StandardScaler` on the six numeric columns, `OneHotEncoder(drop='first')` on
country and gender, binaries passed through: 11 columns, as in the source. A test pins that width.

**Tree family.** No scaling. One-hot with `drop='if_binary'`, so gender becomes one column
(`Gender_Male`) while every country keeps its own (`Geography_Germany`, …).
`verbose_feature_names_out=False` plus `set_output("pandas")` mean SHAP later sees real names
instead of `num__x3`. `exclude` removes raw columns entirely; the notebook uses it to price gender.

**Scorecard.** A `ColumnTransformer` with a 5-knot cubic spline on Age, standard scaling on the
remaining numerics, one-hot on country, gender and product count, and the binaries passed through.
Its docstring states the EDA finding behind each choice.

**CV.** `cv_evaluate` clones the pipeline for each fold, so no fitted state leaks between folds.
It records AUC, PR-AUC, Brier and log-loss per fold and fills an out-of-fold vector.

**Tuning.** The objective is the mean 5-fold CV AUC of a tree pipeline. Optuna's TPE sampler is
seeded, and `subsample_freq=1` is fixed both inside the search space and in the returned params.

## The non-obvious bits
- **`subsample_freq=1`** is the fix for the source's search, where `subsample` did nothing.
- **`n_jobs=4`**, not −1: parallel LightGBM inside 5 folds × 40 trials oversubscribes cores on a
  laptop and makes runtimes erratic. It doesn't change results.
- XGBoost uses `tree_method="hist"` for speed, and `eval_metric` is set to silence a warning.
- The scorecard's `LogisticRegression(C=1.0)` is mildly regularised. That matters because the
  4-product dummy perfectly predicts churn (60/60), and without a penalty its coefficient would
  run to infinity.

## How it can break
- New category levels at prediction time: `handle_unknown="ignore"` encodes them as all zeros
  rather than crashing.
- Tuning is the slow step (~60 s). Reducing `n_trials` changes the chosen parameters and the
  quoted numbers, and the notebook's assert cell will say so.
- Tests pin the source width and the gender exclusion. CV and tuning are exercised by the notebook,
  not unit-tested.
