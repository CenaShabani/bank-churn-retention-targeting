"""Every constant the analysis depends on, in one place.

Two kinds live here and they are kept visibly apart:
- *facts about the data* (column names, the target, the source), which come from the dataset;
- *business assumptions* (what a retained customer is worth, what an offer costs), which do
  not come from the data at all and are the first thing a client should be allowed to change.
"""
from __future__ import annotations

from pathlib import Path

SEED = 42

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
OUTPUTS = PROJECT / "outputs"
FIGURES = OUTPUTS / "figures"

# --- Source ------------------------------------------------------------------------------
# The public parent of the Kaggle "Customer Churn Prediction Challenge" data (same 13
# columns). CC0, so it can be redistributed and attached to a Kaggle notebook.
KAGGLE_DATASET = "barelydedicated/bank-customer-churn-modeling"
RAW_FILE = "Churn_Modelling.csv"
RAW_SHA256 = "3996cd1fa372e0db0cd9c0ebac35bbd4e8e3c65fb942bb010c826e7b1eeef0a0"  # pinned 2026-10-12

# --- Schema ------------------------------------------------------------------------------
TARGET = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]          # identifiers, never features
NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CAT_COLS = ["Geography", "Gender"]
BIN_COLS = ["HasCrCard", "IsActiveMember"]
RAW_FEATURES = NUM_COLS + CAT_COLS + BIN_COLS

# --- Business assumptions (NOT from the data) ---------------------------------------------
# A bank earns a spread on the deposits it holds and a fee on each product. These are
# deliberately round, deliberately conservative, and swept in the sensitivity analysis.
DEPOSIT_SPREAD = 0.015       # 1.5% a year on the balance held
FEE_PER_PRODUCT = 60.0       # EUR a year per product held
HORIZON_YEARS = 3            # how long a saved customer is assumed to stay
CONTACT_COST = 30.0          # EUR per targeted customer: an advisor's call
INCENTIVE = 100.0            # EUR fee waiver offered on the call, paid to EVERY customer targeted,
                             # including the ones who were never going to leave
SAVE_RATE = 0.20             # share of would-be churners the offer actually keeps
