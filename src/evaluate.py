"""Holdout evaluation with uncertainty attached.

A single holdout AUC of 0.86 says nothing about whether 0.86 differs from 0.85. Every
comparison here is a *paired bootstrap*: resample the same holdout customers, score both
models on each resample, and look at the distribution of the difference. Pairing matters:
both models face the same easy and hard customers, so the noise they share cancels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)

from . import config


def holdout_metrics(y, p) -> dict:
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), ece=expected_calibration_error(y, p))


def _resample_idx(y: np.ndarray, n_boot: int, seed: int):
    """Stratified resamples: each keeps the holdout's churner/stayer counts, so no resample
    is ever all one class and AUC is always defined."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, p, metric=roc_auc_score, n_boot: int = 2000, seed: int = config.SEED):
    y, p = np.asarray(y), np.asarray(p)
    vals = np.array([metric(y[i], p[i]) for i in _resample_idx(y, n_boot, seed)])
    return float(metric(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(y, p_new, p_old, metric=roc_auc_score, n_boot: int = 2000,
                 seed: int = config.SEED) -> dict:
    """metric(new) - metric(old), with a 95% CI and the share of resamples where new is not better."""
    y, a, b = np.asarray(y), np.asarray(p_new), np.asarray(p_old)
    d = np.array([metric(y[i], a[i]) - metric(y[i], b[i]) for i in _resample_idx(y, n_boot, seed)])
    return dict(delta=float(metric(y, a) - metric(y, b)), ci_low=float(np.percentile(d, 2.5)),
                ci_high=float(np.percentile(d, 97.5)), p_not_better=float((d <= 0).mean()))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Customers grouped into deciles of predicted risk: mean predicted vs actual churn rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    t = df.groupby("bin").agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Average |predicted - actual| over risk deciles, weighted by size. 0.02 = off by 2 points."""
    t = calibration_table(y, p, bins)
    return float((t["n"] * (t["predicted"] - t["actual"]).abs()).sum() / t["n"].sum())


def at_threshold(y, p, thr: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (np.asarray(p) >= thr).astype(int)).ravel()
    return dict(threshold=thr, flagged=int(tp + fp), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


def lift_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Rank customers by risk, cut into deciles: churn rate and lift per decile, and the
    cumulative share of all churners caught by contacting down to that decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    df["decile"] = np.arange(len(df)) * bins // len(df) + 1
    t = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), mean_p=("p", "mean"))
    t["churn_rate"] = t["churners"] / t["customers"]
    t["lift"] = t["churn_rate"] / df["y"].mean()
    t["cum_share_of_churners"] = t["churners"].cumsum() / df["y"].sum()
    return t


def segment_performance(X: pd.DataFrame, y, p, by: str) -> pd.DataFrame:
    """Does the model work equally well for every group? AUC within the group, and whether its
    average prediction matches the group's real churn rate."""
    df = X[[by]].copy()
    df["y"], df["p"] = np.asarray(y), np.asarray(p)
    rows = []
    for level, g in df.groupby(by):
        rows.append(dict(segment=f"{by} = {level}", customers=len(g), churn_rate=g["y"].mean(),
                         mean_predicted=g["p"].mean(),
                         auc=roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan))
    return pd.DataFrame(rows)
