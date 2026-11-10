# `src/features.py`

## What this module is for
It goes from the raw table to model inputs: drop identifiers, separate the target, make the
holdout split, and add four engineered columns. It is used by the notebook directly and by
`models.py` inside its pipelines.

## Public surface
- `split_xy(df) -> (X, y)`: X has the 10 raw feature columns; y is `Exited` as int.
- `holdout_split(X, y, test_size=0.2) -> X_train, X_test, y_train, y_test`: stratified, seed 42.
- `engineer(X) -> X'`: X plus `ZeroBalance`, `BalancePerProduct`, `TenureByAge`, `MultiProduct3Plus`.
- `ENGINEERED`: the list of those four names.

## Walkthrough
`split_xy` selects `config.RAW_FEATURES`, so `RowNumber`, `CustomerId` and `Surname` can never
leak in. `holdout_split` is the source notebook's split exactly, so the comparison with the source
is on identical customers. `engineer` adds four columns, each motivated in the module docstring by
something §3 of the notebook shows: the zero-balance spike, deposits per product, tenure relative
to age, and the 3+ product cliff.

## The non-obvious bits
- `engineer` is **row-wise**: no statistic is learned from data, so it can run before or after the
  split without leakage, and it can sit inside a pipeline as a `FunctionTransformer`.
- It copies its input. The test checks that the caller's frame is not mutated, because a mutated
  `X_test` would quietly change later cells.
- The features help the scorecard but not LightGBM (CV 0.856 vs 0.860 without them). That is
  reported, not hidden: trees find these splits on their own.

## How it can break
`BalancePerProduct` divides by `NumOfProducts`, which is ≥ 1 in this data. A zero would give inf.
`validate` does not check this range, so a new data source with 0 products would need a guard.
