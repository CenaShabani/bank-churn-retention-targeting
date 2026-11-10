"""SHAP explanations, from global importance down to a reason code per customer.

SHAP splits one prediction into additive pieces, one per feature: base rate + sum of pieces =
the model's output for that customer (in log-odds). Three uses here:
1. global: which features move predictions most, on average (mean |SHAP|);
2. shape: how a feature's value maps to its push on risk (dependence plots, in raw units);
3. action: the top reasons for *one* customer (reason codes), and groups of high-risk
   customers who are at risk for the *same* reasons (personas), which is what a retention team
   can design a playbook around.

SHAP explains the model, not the world. "Inactive raises predicted risk" is a fact about the
model; "activating a customer would keep them" is a causal claim SHAP cannot make.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from . import config


def model_inputs(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """What the classifier actually sees: X pushed through every step but the last."""
    return pipe[:-1].transform(X)


def shap_explanation(pipe: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP values (log-odds units) for the pipeline's final tree model."""
    Xm = model_inputs(pipe, X)
    ex = shap.TreeExplainer(pipe[-1])(Xm)
    if ex.values.ndim == 3:            # some versions return one slice per class
        ex = ex[:, :, 1]
    return ex


def global_importance(ex: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(ex.values).mean(0), index=ex.feature_names).sort_values(ascending=False)


def _readable(feature: str, value: float) -> str:
    if feature.startswith("Geography_"):
        return f"lives in {feature.split('_', 1)[1]}" if value == 1 else f"not in {feature.split('_', 1)[1]}"
    if feature == "Gender_Male":
        return "male" if value == 1 else "female"
    if feature == "IsActiveMember":
        return "inactive member" if value == 0 else "active member"
    if feature == "ZeroBalance":
        return "zero balance" if value == 1 else "holds a balance"
    if feature == "MultiProduct3Plus":
        return "3+ products" if value == 1 else "fewer than 3 products"
    if feature == "HasCrCard":
        return "has a credit card" if value == 1 else "no credit card"
    if feature in ("Balance", "BalancePerProduct", "EstimatedSalary"):
        return f"{feature} = EUR {value:,.0f}"
    if feature == "TenureByAge":
        return f"TenureByAge = {value:.2f}"
    return f"{feature} = {value:g}"


def reason_codes(ex: shap.Explanation, top: int = 3) -> list[str]:
    """For each customer, the `top` features pushing their risk up most, in plain words."""
    out = []
    for vals, data in zip(ex.values, ex.data):
        order = np.argsort(-vals)[:top]
        out.append("; ".join(f"{_readable(ex.feature_names[j], data[j])} (+{vals[j]:.2f})"
                             for j in order if vals[j] > 0))
    return out


def personas(ex: shap.Explanation, mask: np.ndarray, k_range=range(3, 7)) -> tuple[np.ndarray, int, dict]:
    """Cluster the *explanations* of high-risk customers, not their raw features.

    Two customers with the same churn risk can be at risk for different reasons (one is old and
    inactive, another holds three products). Clustering SHAP vectors groups customers by
    *why* the model flags them, and SHAP values share one unit (log-odds), so no scaling is
    needed. k is chosen by silhouette score, not by eye.
    """
    V = ex.values[mask]
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=20, random_state=config.SEED).fit_predict(V)
        scores[k] = silhouette_score(V, labels, random_state=config.SEED)
    best = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best, n_init=20, random_state=config.SEED).fit_predict(V)
    return labels, best, scores


def persona_profile(ex: shap.Explanation, mask: np.ndarray, labels: np.ndarray,
                    y: np.ndarray, p: np.ndarray, top: int = 3) -> pd.DataFrame:
    """One row per persona: size, real and predicted churn, its top drivers, and a few traits."""
    names = list(ex.feature_names)
    V, D = ex.values[mask], pd.DataFrame(ex.data[mask], columns=names)
    yy, pp = np.asarray(y)[mask], np.asarray(p)[mask]
    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        mean_shap = pd.Series(V[m].mean(0), index=names).sort_values(ascending=False)
        d = D[m]
        rows.append(dict(
            persona=int(c), customers=int(m.sum()), actual_churn=yy[m].mean(), predicted_churn=pp[m].mean(),
            top_drivers=", ".join(f"{f} (+{v:.2f})" for f, v in mean_shap.head(top).items() if v > 0),
            median_age=d["Age"].median(), share_inactive=1 - d["IsActiveMember"].mean(),
            share_germany=d.get("Geography_Germany", pd.Series(0)).mean(),
            share_1_product=(d["NumOfProducts"] == 1).mean(),
            share_3plus_products=(d["NumOfProducts"] >= 3).mean(),
            share_zero_balance=(d["Balance"] == 0).mean(), share_female=1 - d["Gender_Male"].mean(),
        ))
    return pd.DataFrame(rows).sort_values("customers", ascending=False).reset_index(drop=True)


def name_persona(row: pd.Series) -> str:
    """A plain-English label from a persona's profile. Rules, not judgement, so the label
    follows the data if the clusters ever change; anything unmatched is called what it is."""
    if row["share_3plus_products"] >= 0.5:
        return "3+ products"
    if row["share_germany"] >= 0.5 and row["share_1_product"] >= 0.5:
        return "German single-product"
    if row["median_age"] >= 45 and row["share_inactive"] >= 0.5:
        return "Older, disengaged"
    return f"Persona {int(row['persona'])}"
