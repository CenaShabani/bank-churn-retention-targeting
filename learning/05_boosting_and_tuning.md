# 05 · Gradient boosting, and what tuning can and can't buy

**The one idea:** boosting builds many small trees, each correcting the last one's errors; tuning
sets how big, how many and how cautious those trees are, and here it bought one point of AUC.

## The anchor
A team of editors passing a draft along, each fixing what the previous one missed. The learning
rate is how boldly each editor rewrites. Small changes and more editors is slower but harder to
overdo.

## The hyperparameters searched
`learning_rate`, `n_estimators` (how many trees), `max_depth` / `num_leaves` (how complex each
tree is), `min_child_samples` (the smallest group a leaf may describe), `subsample` /
`colsample_bytree` (each tree sees a random share of rows / columns), `reg_lambda` (a penalty on
large leaf values). The search settled on a slow learning rate (0.011) with 403 moderately shallow trees.

## Optuna (TPE)
Rather than a grid, Optuna's Tree-structured Parzen Estimator looks at which settings have scored
well so far and proposes the next trial where improvement is likely. 40 trials with a seeded
sampler, so the same trials run in the same order every time.

## The bug in the source
LightGBM samples rows only when `subsample_freq > 0`. The source tuned `subsample` without setting
it, so every value it tried trained the identical model. This project sets `subsample_freq=1`.

## What it bought
CV AUC 0.858 → 0.867; holdout 0.860 → 0.868. Real (lesson 06) but small, and lesson 10 shows it
changes no decision.

## Where it becomes code
`src/models.py: lgbm, lgbm_search_space, tune_lgbm`.
