"""Unit tests for the src modules. Small hand-checkable inputs, so a failure points at the logic,
not at the data. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import acquisition, business, config, eda, evaluate, explain, features, models  # noqa: E402


def toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "RowNumber": np.arange(n), "CustomerId": np.arange(n) + 1000, "Surname": ["x"] * n,
        "CreditScore": rng.integers(400, 850, n), "Geography": rng.choice(["France", "Germany", "Spain"], n),
        "Gender": rng.choice(["Male", "Female"], n), "Age": rng.integers(18, 80, n),
        "Tenure": rng.integers(0, 11, n), "Balance": np.where(rng.random(n) < 0.3, 0.0, rng.uniform(1e4, 2e5, n)),
        "NumOfProducts": rng.integers(1, 5, n), "HasCrCard": rng.integers(0, 2, n),
        "IsActiveMember": rng.integers(0, 2, n), "EstimatedSalary": rng.uniform(1e3, 2e5, n),
        "Exited": np.tile([0, 1], n // 2),
    })


# --- acquisition ----------------------------------------------------------------------------
def test_validate_accepts_clean_data():
    acquisition.validate(toy())


def test_validate_rejects_duplicate_customer():
    df = toy()
    df.loc[1, "CustomerId"] = df.loc[0, "CustomerId"]
    with pytest.raises(ValueError, match="duplicate"):
        acquisition.validate(df)


def test_validate_rejects_missing_column():
    with pytest.raises(ValueError, match="missing"):
        acquisition.validate(toy().drop(columns="Balance"))


# --- eda ------------------------------------------------------------------------------------
def test_wilson_contains_rate_and_stays_in_unit_interval():
    lo, hi = eda.wilson(np.array([0, 5, 60]), np.array([60, 10, 60]))
    assert np.all(lo >= -1e-12) and np.all(hi <= 1 + 1e-12)
    assert lo[1] < 0.5 < hi[1]
    assert hi[2] == pytest.approx(1.0) and lo[2] < 1.0      # 60/60 still gets an interval


def test_churn_by_counts():
    t = eda.churn_by(toy(), "Geography")
    assert t["customers"].sum() == 40
    assert np.allclose(t["churn_rate"], t["churned"] / t["customers"])


# --- features -------------------------------------------------------------------------------
def test_engineer_values():
    X = pd.DataFrame({"Balance": [0.0, 1000.0], "NumOfProducts": [1, 4], "Tenure": [2, 5], "Age": [20, 50]})
    out = features.engineer(X)
    assert out["ZeroBalance"].tolist() == [1, 0]
    assert out["BalancePerProduct"].tolist() == [0.0, 250.0]
    assert out["TenureByAge"].tolist() == [0.1, 0.1]
    assert out["MultiProduct3Plus"].tolist() == [0, 1]
    assert "ZeroBalance" not in X                          # input not mutated


def test_split_xy_drops_identifiers():
    X, y = features.split_xy(toy())
    assert not set(config.ID_COLS) & set(X.columns)
    assert config.TARGET not in X


# --- models ---------------------------------------------------------------------------------
def test_source_preprocessor_matches_source_width():
    X, _ = features.split_xy(toy())
    assert models.source_preprocessor().fit_transform(X).shape[1] == 11   # as in the source notebook


def test_tree_pipeline_can_exclude_gender():
    X, y = features.split_xy(toy())
    pipe = models.tree_pipeline(models.lgbm(n_estimators=5), exclude=("Gender",)).fit(X, y)
    cols = pipe[:-1].transform(X).columns
    assert not any(c.startswith("Gender") for c in cols)
    assert "Geography_Germany" in cols


# --- evaluate -------------------------------------------------------------------------------
def test_paired_delta_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 50)
    p = rng.random(100)
    d = evaluate.paired_delta(y, p, p, n_boot=50)
    assert d["delta"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_perfect_calibration_has_zero_ece():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 100)
    y = np.concatenate([np.r_[np.ones(int(round(q * 100))), np.zeros(100 - int(round(q * 100)))]
                        for q in np.linspace(0.05, 0.95, 10)])
    assert evaluate.expected_calibration_error(y, p) == pytest.approx(0, abs=1e-9)


# --- business -------------------------------------------------------------------------------
def test_realised_profit_by_hand():
    y = np.array([1, 0, 1])
    v = np.array([1000.0, 1000.0, 50.0])
    t = np.array([True, True, False])
    # 1 churner saved at 20% of 1000 = 200, minus 2 calls at 130
    assert business.realised_profit(y, v, t, save_rate=0.2, cost=130) == pytest.approx(200 - 260)


def test_break_even_threshold():
    assert business.break_even_threshold(1000, save_rate=0.2) == pytest.approx(business.cost_per_target() / 200)


def test_hindsight_policy_only_targets_worthwhile_churners():
    y = np.array([1, 1, 0])
    v = np.array([10_000.0, 100.0, 10_000.0])
    pol = business.policies(y, np.array([0.9, 0.9, 0.9]), v, mean_value_train=5000)
    assert pol["Hindsight ceiling (knows who churns)"].tolist() == [True, False, False]


def test_profit_curve_ends_at_call_everyone():
    rng = np.random.default_rng(2)
    y, p, v = rng.integers(0, 2, 50), rng.random(50), rng.uniform(100, 5000, 50)
    c = business.profit_curve(y, p, v)
    assert c["profit"].iloc[-1] == pytest.approx(business.realised_profit(y, v, np.ones(50, bool)))
    assert c["profit"].iloc[0] == 0


def test_ab_test_size_textbook_value():
    # 50% -> 40%, alpha 0.05, power 0.8: ~385-390 per arm in standard tables
    assert 380 <= business.ab_test_size(0.5, 0.2) <= 392


# --- explain --------------------------------------------------------------------------------
def test_reason_codes_pick_largest_positive_contributions():
    import shap
    ex = shap.Explanation(values=np.array([[0.5, -1.0, 0.2]]), data=np.array([[0, 3, 52]]),
                          feature_names=["IsActiveMember", "NumOfProducts", "Age"])
    code = explain.reason_codes(ex, top=3)[0]
    assert code.startswith("inactive member (+0.50)")
    assert "Age = 52 (+0.20)" in code and "NumOfProducts" not in code


def test_name_persona_rules():
    base = dict(persona=0, share_3plus_products=0.0, share_germany=0.0, share_1_product=0.0,
                median_age=30, share_inactive=0.0)
    assert explain.name_persona(pd.Series({**base, "share_3plus_products": 1.0})) == "3+ products"
    assert explain.name_persona(pd.Series({**base, "share_germany": 0.7, "share_1_product": 1.0})) == "German single-product"
    assert explain.name_persona(pd.Series({**base, "median_age": 50, "share_inactive": 0.6})) == "Older, disengaged"
    assert explain.name_persona(pd.Series(base)) == "Persona 0"
