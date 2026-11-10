"""The statistics behind the EDA: churn rates with honest intervals, and tests that say which
differences are real.

Why intervals: a churn rate on 60 customers and one on 5,000 should not be drawn as equally
certain. Wilson intervals stay sensible near 0% and 100%, where the textbook normal interval
breaks (the 4-product group churns at exactly 100%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def wilson(k: np.ndarray | int, n: np.ndarray | int, z: float = 1.96):
    """95% Wilson score interval for k successes out of n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def churn_by(df: pd.DataFrame, col: str | pd.Series, name: str | None = None) -> pd.DataFrame:
    """Churn rate per level of `col`, with its count and Wilson interval."""
    key = df[col] if isinstance(col, str) else col
    g = df.groupby(key, observed=True)[config.TARGET].agg(churned="sum", customers="size")
    g["churn_rate"] = g["churned"] / g["customers"]
    g["ci_low"], g["ci_high"] = wilson(g["churned"], g["customers"])
    g.index.name = name or (col if isinstance(col, str) else col.name)
    return g


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, c = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def association_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: the right test for its type, its p-value, and an effect size.

    Categorical / binary / small-integer features: chi-square test of independence, effect size
    Cramer's V. Continuous features: Mann-Whitney U (no normality assumption; Balance is a spike
    plus a bump, nothing like normal), effect size the rank-biserial correlation, which reads as
    "how much more likely a churner ranks higher than a stayer", from -1 to 1.

    With 10,000 rows almost everything is 'significant'; the effect size is what to read.
    """
    y = df[config.TARGET]
    rows = []
    for col in config.CAT_COLS + config.BIN_COLS + ["NumOfProducts", "Tenure"]:
        tab = pd.crosstab(df[col], y)
        chi2, p, *_ = stats.chi2_contingency(tab, correction=False)
        rows.append(dict(feature=col, test="chi-square", p_value=p, effect="Cramer's V",
                         effect_size=cramers_v(tab)))
    for col in ["Age", "Balance", "CreditScore", "EstimatedSalary"]:
        a, b = df.loc[y == 1, col], df.loc[y == 0, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 2 * u / (len(a) * len(b)) - 1
        rows.append(dict(feature=col, test="Mann-Whitney U", p_value=p, effect="rank-biserial",
                         effect_size=rbc))
    out = pd.DataFrame(rows)
    out["abs_effect"] = out["effect_size"].abs()
    return out.sort_values("abs_effect", ascending=False).drop(columns="abs_effect").reset_index(drop=True)


def age_bands(age: pd.Series) -> pd.Series:
    return pd.cut(age, [17, 29, 39, 49, 59, 69, 100],
                  labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"])


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """The checks a client would ask about before trusting anything downstream."""
    feats = df[config.RAW_FEATURES]
    return pd.DataFrame({
        "check": ["rows", "missing values", "duplicate customers (CustomerId)",
                  "duplicate feature rows", "customers with Balance = 0",
                  "German customers with Balance = 0", "EstimatedSalary below 1,000"],
        "value": [len(df), int(feats.isna().sum().sum()), int(df["CustomerId"].duplicated().sum()),
                  int(feats.duplicated().sum()), int((df["Balance"] == 0).sum()),
                  int(((df["Geography"] == "Germany") & (df["Balance"] == 0)).sum()),
                  int((df["EstimatedSalary"] < 1000).sum())],
    })
