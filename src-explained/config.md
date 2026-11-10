# `src/config.py`

## What this module is for
Every constant the analysis depends on, in one place: paths, the seed, the data source and its
pinned checksum, the schema, and the business assumptions. Every other module imports it; it
imports nothing from the project.

## Public surface
- `SEED = 42`. Used for the split, the folds, Optuna's sampler, k-means and the bootstraps.
- Paths: `PROJECT`, `RAW_DIR`, `OUTPUTS`, `FIGURES` (all derived from this file's location).
- Source: `KAGGLE_DATASET`, `RAW_FILE`, `RAW_SHA256`.
- Schema: `TARGET`, `ID_COLS`, `NUM_COLS`, `CAT_COLS`, `BIN_COLS`, `RAW_FEATURES`.
- Assumptions: `DEPOSIT_SPREAD = 0.015`, `FEE_PER_PRODUCT = 60.0`, `HORIZON_YEARS = 3`,
  `CONTACT_COST = 30.0`, `INCENTIVE = 100.0`, `SAVE_RATE = 0.20`.

## Walkthrough
The file is split visibly into *facts about the data* and *business assumptions*. That split is
the point: a client may change the assumptions, and should never need to touch the facts.
`PROJECT` is computed from `__file__`, so the same code works in the repo, in a temp folder and on
Kaggle (where `src/` is written to `/kaggle/working/src/`).

## The non-obvious bits
- `NumOfProducts` is listed under `NUM_COLS` because that is how the source notebook treats it.
  The scorecard overrides this by one-hot encoding it (see `models.md`).
- `INCENTIVE` is commented as paid to *every* targeted customer. That is the reason "call
  everyone" is not optimal: the waiver leaks to stayers.
- `RAW_SHA256` is pinned, with a date. Changing the data means changing this line on purpose.

## How it can break
Changing an assumption changes every euro figure. The notebook's final assert cell then fails,
which is intended: the prose quotes the base case. Changing `SEED` changes the split, and every
number moves.
