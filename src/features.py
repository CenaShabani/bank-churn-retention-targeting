"""From the raw table to model inputs: drop identifiers, split, and add the engineered features.

The engineered features each answer something the EDA showed, so each one has a reason:
- ZeroBalance       36% of customers hold exactly 0, a spike no smooth curve over Balance captures.
- BalancePerProduct deposits spread over products: a heavy single-product saver differs from a
                    light multi-product one.
- TenureByAge       years as a customer relative to adult life, a loyalty proxy that tenure alone
                    (flat churn across 0-10 years) does not give.
- MultiProduct3Plus 3-4 products churn at 83-100%; a flag lets a linear model see the cliff.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

ENGINEERED = ["ZeroBalance", "BalancePerProduct", "TenureByAge", "MultiProduct3Plus"]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Identifiers out (CustomerId and Surname carry no behaviour, only leakage risk), target apart."""
    X = df[config.RAW_FEATURES].copy()
    y = df[config.TARGET].astype(int)
    return X, y


def holdout_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20):
    """The same 80/20 stratified split the source notebook uses, same seed.

    Everything that chooses anything (model, hyperparameters, threshold) happens inside the 80%.
    The 20% is scored once, at the end.
    """
    return train_test_split(X, y, test_size=test_size, random_state=config.SEED, stratify=y)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered columns. Pure function of each row, so no train/test leakage."""
    out = X.copy()
    out["ZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["BalancePerProduct"] = out["Balance"] / out["NumOfProducts"]
    out["TenureByAge"] = out["Tenure"] / out["Age"]
    out["MultiProduct3Plus"] = (out["NumOfProducts"] >= 3).astype(int)
    return out
