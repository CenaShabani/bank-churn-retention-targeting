"""Model pipelines, cross-validation and tuning.

Two families live here, on purpose:
- `source_*` reproduces the Kaggle notebook this project starts from, line for line in spirit
  (StandardScaler + one-hot with drop='first', default LR / RF / LightGBM). It is the baseline
  every improvement is measured against, on the same split and the same folds.
- everything else is this project's version: raw-unit features for trees (so SHAP plots read in
  euros and years, not z-scores), engineered features, a tuned LightGBM, and an interpretable
  logistic "scorecard" that a bank's model-risk team could actually sign off.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from . import config
from .features import engineer

N_FOLDS = 5


def folds() -> StratifiedKFold:
    """One fold definition shared by every model, so their scores are paired fold by fold."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)


# --- The source notebook, reproduced -------------------------------------------------------
def source_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), config.NUM_COLS),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]),
             config.CAT_COLS),
        ],
        remainder="passthrough",
    )


def source_baselines() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(random_state=config.SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=config.SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=config.SEED, n_jobs=-1, verbose=-1),
    }
    return {name: Pipeline([("preprocessor", source_preprocessor()), ("classifier", m)])
            for name, m in models.items()}


# --- This project's pipelines -------------------------------------------------------------
def tree_preprocessor(engineered: bool, exclude: tuple[str, ...] = ()) -> ColumnTransformer:
    """No scaling (trees are invariant to it, and scaled inputs make SHAP plots unreadable).
    One-hot keeps every country as its own named column: Geography_Germany, not cat__x0_2.
    `exclude` drops raw columns entirely, e.g. ("Gender",) to price what a protected attribute adds."""
    num = config.NUM_COLS + config.BIN_COLS + (
        ["ZeroBalance", "BalancePerProduct", "TenureByAge", "MultiProduct3Plus"] if engineered else [])
    num = [c for c in num if c not in exclude]
    cat = [c for c in config.CAT_COLS if c not in exclude]
    ct = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), cat)],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def tree_pipeline(clf, engineered: bool = True, exclude: tuple[str, ...] = ()) -> Pipeline:
    steps = [("engineer", FunctionTransformer(engineer))] if engineered else []
    return Pipeline(steps + [("prep", tree_preprocessor(engineered, exclude)), ("clf", clf)])


def lgbm(**params) -> LGBMClassifier:
    base = dict(random_state=config.SEED, n_jobs=4, verbose=-1)
    return LGBMClassifier(**{**base, **params})


def xgb(**params) -> XGBClassifier:
    base = dict(random_state=config.SEED, n_jobs=4, eval_metric="logloss", tree_method="hist")
    return XGBClassifier(**{**base, **params})


def scorecard_pipeline() -> Pipeline:
    """A logistic regression that sees the shapes the EDA found, so it can compete honestly.

    - Age through a cubic spline: churn rises to a peak in the 50s and falls again; a single
      linear coefficient on Age cannot bend.
    - NumOfProducts as categories: churn is U-shaped (1 product 28%, 2 products 8%, 3-4 near
      100%); as a number it would be forced to be monotone.
    - ZeroBalance as its own flag for the spike at exactly zero.
    Every coefficient is still a readable log-odds effect, which is what a model-risk reviewer
    in a bank asks for.
    """
    ct = ColumnTransformer([
        ("age", SplineTransformer(n_knots=5, degree=3), ["Age"]),
        ("num", StandardScaler(), ["CreditScore", "Balance", "EstimatedSalary", "Tenure", "TenureByAge"]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["Geography", "Gender", "NumOfProducts"]),
        ("bin", "passthrough", ["HasCrCard", "IsActiveMember", "ZeroBalance"]),
    ])
    return Pipeline([("engineer", FunctionTransformer(engineer)), ("prep", ct),
                     ("clf", LogisticRegression(C=1.0, max_iter=5000))])


# --- Evaluation by cross-validation -------------------------------------------------------
def cv_evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit on 4/5, score the held-out 1/5, five times. Returns per-fold metrics and the
    out-of-fold (OOF) probabilities: every training row predicted by a model that never saw it."""
    oof = np.zeros(len(y))
    rows = []
    for tr, va in folds().split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        yv = y.iloc[va]
        rows.append(dict(auc=roc_auc_score(yv, p), pr_auc=average_precision_score(yv, p),
                         brier=brier_score_loss(yv, p), log_loss=log_loss(yv, p)))
    per_fold = pd.DataFrame(rows)
    return dict(per_fold=per_fold, mean=per_fold.mean(), std=per_fold.std(ddof=0), oof=oof)


def cv_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model: mean and std of each metric across folds."""
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for k in ["auc", "pr_auc", "brier", "log_loss"]:
            row[k] = r["mean"][k]
            row[f"{k}_sd"] = r["std"][k]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# --- Tuning -------------------------------------------------------------------------------
def lgbm_search_space(trial: optuna.Trial) -> dict:
    """The source notebook's space, plus regularisation, with one fix: LightGBM ignores
    `subsample` unless `subsample_freq` > 0. The source tuned a parameter that did nothing."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def tune_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 40, engineered: bool = True):
    """Bayesian search (Optuna's TPE) for the hyperparameters with the best 5-fold CV AUC.
    Seeded sampler, so the same trials run in the same order every time."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        pipe = tree_pipeline(lgbm(**lgbm_search_space(trial)), engineered=engineered)
        return cv_evaluate(pipe, X, y)["mean"]["auc"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials)
    params = {**study.best_params, "subsample_freq": 1}
    return params, study
