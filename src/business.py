"""From a churn probability to a decision: whom should the bank call, and is it worth it?

The economics of one targeted customer i:

    benefit_i = p_i * save_rate * value_i         (only churners can be saved)
    cost      = contact_cost + incentive          (paid for everyone targeted, churner or not)
    target i  if  benefit_i > cost

Two things fall out of that inequality, and they are the business half of this project:
1. The right cut-off is not 0.5. It is cost / (save_rate * value), which for a typical customer
   is far lower: missing a churner costs much more than calling a stayer.
2. The cut-off is different for every customer, because value_i is. A 40% risk on a EUR 150k
   depositor is worth a call; a 60% risk on an empty account may not be. Ranking by probability
   alone ignores that.

All money inputs are assumptions (config.py), not data. The sensitivity grid shows where the
recommendation would flip, which is the honest way to present a number built on assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def customer_value(X: pd.DataFrame, multiplier: float = 1.0) -> np.ndarray:
    """What keeping customer i is worth over the horizon: deposit spread on the balance plus a
    fee per product held, for HORIZON_YEARS years."""
    annual = config.DEPOSIT_SPREAD * X["Balance"] + config.FEE_PER_PRODUCT * X["NumOfProducts"]
    return (multiplier * config.HORIZON_YEARS * annual).to_numpy()


def cost_per_target() -> float:
    return config.CONTACT_COST + config.INCENTIVE


def expected_net(p, value, save_rate=config.SAVE_RATE, cost=None) -> np.ndarray:
    cost = cost_per_target() if cost is None else cost
    return np.asarray(p) * save_rate * np.asarray(value) - cost


def realised_profit(y, value, targeted, save_rate=config.SAVE_RATE, cost=None) -> float:
    """What the campaign would actually have earned on these customers, using what really
    happened (y), not what the model expected."""
    cost = cost_per_target() if cost is None else cost
    y, value, t = np.asarray(y), np.asarray(value), np.asarray(targeted, bool)
    return float((y[t] * save_rate * value[t]).sum() - cost * t.sum())


def break_even_threshold(mean_value: float, save_rate=config.SAVE_RATE) -> float:
    """The single probability cut-off at which an *average* customer is worth a call."""
    return cost_per_target() / (save_rate * mean_value)


def policies(y, p, value, mean_value_train: float, save_rate=config.SAVE_RATE, cost=None) -> dict:
    """The targeting rules compared on the holdout. Each is fixed before looking at y."""
    cost = cost_per_target() if cost is None else cost
    p, value, y = np.asarray(p), np.asarray(value), np.asarray(y)
    thr = cost / (save_rate * mean_value_train)
    return {
        "Call no one": np.zeros_like(p, bool),
        "Call everyone": np.ones_like(p, bool),
        "Source notebook: p >= 0.5": p >= 0.5,
        f"Break-even cut-off: p >= {thr:.2f}": p >= thr,
        "Value-aware: p x save x value > cost": expected_net(p, value, save_rate, cost) > 0,
        "Hindsight ceiling (knows who churns)": (y == 1) & (save_rate * value > cost),
    }


def policy_table(y, p, value, mean_value_train: float, save_rate=config.SAVE_RATE, cost=None) -> pd.DataFrame:
    rows = []
    for name, t in policies(y, p, value, mean_value_train, save_rate, cost).items():
        rows.append(dict(policy=name, targeted=int(t.sum()), churners_reached=int(np.asarray(y)[t].sum()),
                         profit=realised_profit(y, value, t, save_rate, cost)))
    return pd.DataFrame(rows)


def bootstrap_policy_delta(y, value, t_new, t_old, n_boot=2000, seed=config.SEED,
                           save_rate=config.SAVE_RATE, cost=None) -> dict:
    """Profit(new) - profit(old) over resampled holdout customers, with a 95% interval."""
    cost = cost_per_target() if cost is None else cost
    y, v = np.asarray(y), np.asarray(value)
    gain = y * save_rate * v - cost                        # per-customer profit if targeted
    diff = gain * (np.asarray(t_new, float) - np.asarray(t_old, float))
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(n_boot)])
    return dict(delta=float(diff.sum()), ci_low=float(np.percentile(boots, 2.5)),
                ci_high=float(np.percentile(boots, 97.5)), p_not_better=float((boots <= 0).mean()))


def profit_curve(y, score, value, save_rate=config.SAVE_RATE, cost=None) -> pd.DataFrame:
    """Contact customers in descending order of `score`; realised profit after each one."""
    cost = cost_per_target() if cost is None else cost
    order = np.argsort(-np.asarray(score), kind="stable")
    gain = np.asarray(y)[order] * save_rate * np.asarray(value)[order] - cost
    cum = np.concatenate([[0.0], np.cumsum(gain)])
    n = np.arange(len(cum))
    return pd.DataFrame({"contacted": n, "share_contacted": n / len(order), "profit": cum})


def value_at_risk_concentration(p, value) -> pd.DataFrame:
    """Expected value lost (p x value), sorted: what share of it sits in the top x% of customers."""
    evl = np.sort(np.asarray(p) * np.asarray(value))[::-1]
    share = np.cumsum(evl) / evl.sum()
    return pd.DataFrame({"share_of_customers": np.arange(1, len(evl) + 1) / len(evl),
                         "share_of_value_at_risk": share})


def sensitivity(y, p, value_fn, mean_value_fn, save_rates, multipliers) -> pd.DataFrame:
    """Re-run the decision under other assumptions. value_fn(m) gives holdout values at
    multiplier m; mean_value_fn(m) the training-set mean used for the break-even cut-off."""
    rows = []
    for s in save_rates:
        for m in multipliers:
            v = value_fn(m)
            tab = policy_table(y, p, v, mean_value_fn(m), save_rate=s).set_index("policy")["profit"]
            names = list(tab.index)
            rows.append(dict(save_rate=s, value_multiplier=m,
                             value_aware=tab[names[4]], break_even=tab[names[3]],
                             source_05=tab[names[2]], call_everyone=tab[names[1]],
                             share_targeted_value_aware=float(
                                 (expected_net(p, v, s) > 0).mean())))
    out = pd.DataFrame(rows)
    out["gain_vs_source"] = out["value_aware"] - out["source_05"]
    return out


def ab_test_size(p_control: float, relative_reduction: float, alpha=0.05, power=0.80) -> int:
    """Customers per arm to detect the campaign's effect in a randomised holdout test
    (two-sided two-proportion z-test)."""
    p1, p2 = p_control, p_control * (1 - relative_reduction)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
         / (p1 - p2) ** 2)
    return int(np.ceil(n))
