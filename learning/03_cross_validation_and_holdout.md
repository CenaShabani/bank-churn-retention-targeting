# 03 · Cross-validation for choosing, the holdout for reporting

**The one idea:** every choice (model, features, hyperparameters, threshold) is made by
cross-validation on the training data; the holdout is scored once, at the end, so its number is
not flattered by having been chosen.

## The anchor
Practice exams for studying, the real exam once. If you retake the real exam until you pass, your
pass says little about what you know.

## How it's set up
- **80/20 stratified split, seed 42**, identical to the source notebook. Stratified means both
  parts keep the 20.4% churn rate.
- **5-fold CV inside the 80%**: train on 4/5, score the remaining 1/5, five times. Every training
  customer gets one *out-of-fold* prediction from a model that never saw them.
- **One fold definition shared by every model** (`models.folds()`), so scores are paired fold by
  fold and differences aren't fold luck.
- Optuna's 40 trials each run a full 5-fold CV. The holdout plays no part in any of it.
- The break-even cut-off in §8 uses the *training* set's mean customer value, also fixed before
  the holdout is looked at.

## What the holdout then tells you
The tuned model's CV AUC (0.867) and holdout AUC (0.868) agree. Had the holdout come in much lower,
the tuning would have overfit the CV folds.

## Where it becomes code
`src/features.py: holdout_split` and `src/models.py: folds, cv_evaluate, tune_lgbm`.
