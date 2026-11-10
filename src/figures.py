"""Every figure in the project. One function per figure, each returning the matplotlib Figure;
`save` writes it to outputs/figures/ for the deck. Nothing is drawn anywhere else.

Colour roles are fixed so a reader learns them once (same roles as the rest of the portfolio):
  BLUE   = this project's approach (tuned model, value-aware rule)
  ORANGE = the source notebook / common practice it is compared against
  AQUA   = the hindsight ceiling, or a third series
  GREY   = reference lines and context
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

BLUE, ORANGE, AQUA, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#c8423b"
INK, INK_2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 10,
    "legend.frameon": False,
})


def save(fig, name: str):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    return fig


def _eur(x, _=None):
    return f"€{x/1000:,.0f}k" if abs(x) >= 1000 else f"€{x:,.0f}"


def _pct(x, _=None):
    return f"{x:.0%}"


def churn_by_segment(tables: dict[str, pd.DataFrame], overall: float):
    """A panel per feature: churn rate per level with its 95% Wilson interval and count."""
    fig, axes = plt.subplots(1, len(tables), figsize=(3.3 * len(tables), 3.6), sharey=True)
    for ax, (title, t) in zip(axes, tables.items()):
        x = np.arange(len(t))
        err = np.clip(np.vstack([t["churn_rate"] - t["ci_low"], t["ci_high"] - t["churn_rate"]]), 0, None)  # 60/60 rounds to -1e-16
        colors = [RED if r > overall * 1.5 else BLUE for r in t["churn_rate"]]
        ax.bar(x, t["churn_rate"], color=colors, alpha=0.85, width=0.7)
        ax.errorbar(x, t["churn_rate"], yerr=err, fmt="none", ecolor=INK_2, capsize=3, lw=1)
        ax.axhline(overall, color=GREY, ls="--", lw=1)
        ax.set_xticks(x, [str(i) for i in t.index], rotation=0)
        for xi, (n, r) in enumerate(zip(t["customers"], t["ci_high"])):
            ax.text(xi, min(r + 0.03, 1.02), f"n={n:,}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(_pct)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("churn rate (95% CI)")
    fig.suptitle(f"Churn is driven by product count, age, country and activity, not by tenure"
                 f"   (dashed line = overall {overall:.0%})",
                 x=0.01, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    return fig


def effect_sizes(assoc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    t = assoc.iloc[::-1]
    colors = [BLUE if p < 0.001 else GREY for p in t["p_value"]]
    ax.barh(t["feature"], t["effect_size"].abs(), color=colors)
    for i, (e, kind) in enumerate(zip(t["effect_size"].abs(), t["effect"])):
        ax.text(e + 0.005, i, f"{e:.2f} ({kind})", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, t["effect_size"].abs().max() * 1.45)
    ax.set(xlabel="association with churn (effect size, 0 = none)",
           title="Effect sizes: Age and product count dominate; salary, card and tenure are noise")
    ax.text(0.99, 0.02, "blue: p < 0.001   grey: not significant at 0.001", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK_2)
    return fig


def model_comparison(cv: pd.DataFrame, source_names: set[str]):
    t = cv.sort_values("auc")
    fig, ax = plt.subplots(figsize=(7, 3.9))
    colors = [ORANGE if m in source_names else BLUE for m in t["model"]]
    ax.barh(t["model"], t["auc"], xerr=t["auc_sd"], color=colors, alpha=0.9, capsize=3,
            error_kw=dict(ecolor=INK_2, lw=1))
    for i, (a, s) in enumerate(zip(t["auc"], t["auc_sd"])):
        ax.text(a + s + 0.003, i, f"{a:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0.70, 0.90)
    ax.set(xlabel="5-fold CV ROC-AUC (bar = mean, line = ±1 sd across folds)",
           title="Model choice moves AUC by about a point once the features are right")
    ax.text(0.99, 0.03, "orange = source notebook as published\nblue = this project",
            transform=ax.transAxes, fontsize=8, color=INK_2, ha="right")
    return fig


def calibration(tables: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="perfect calibration")
    for (name, t), c in zip(tables.items(), [ORANGE, BLUE, AQUA]):
        ax.plot(t["predicted"], t["actual"], "o-", color=c, lw=2, ms=5, label=name)
    ax.set(xlabel="predicted churn probability (decile mean)", ylabel="actual churn rate",
           title="The probabilities can be used as probabilities")
    ax.xaxis.set_major_formatter(_pct); ax.yaxis.set_major_formatter(_pct)
    ax.legend(loc="upper left")
    return fig


def gains(lift: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x = np.concatenate([[0], lift.index / len(lift)])
    y = np.concatenate([[0], lift["cum_share_of_churners"]])
    ax.plot(x, y, "o-", color=BLUE, lw=2, ms=5, label="model ranking")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="random calling")
    for q in (0.2, 0.3):
        v = np.interp(q, x, y)
        ax.annotate(f"call the top {q:.0%}\n→ reach {v:.0%} of churners", (q, v), xytext=(18, -34),
                    textcoords="offset points", fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set(xlabel="share of customers called (highest risk first)", ylabel="share of all churners reached",
           title="The ranking concentrates churners at the top")
    ax.xaxis.set_major_formatter(_pct); ax.yaxis.set_major_formatter(_pct)
    ax.legend(loc="lower right")
    return fig


def shap_beeswarm(ex, max_display: int = 12):
    import shap
    shap.plots.beeswarm(ex, max_display=max_display, show=False, plot_size=(7.5, 4.8))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP: what pushes each customer's predicted risk up or down",
                          loc="left", fontweight="bold")
    return fig


def shap_dependence(ex, features: list[str], color_by: dict[str, str]):
    """SHAP value against the raw feature value, in real units, coloured by a second feature."""
    names = list(ex.feature_names)
    fig, axes = plt.subplots(1, len(features), figsize=(4.4 * len(features), 3.7))
    for ax, f in zip(axes, features):
        j = names.index(f)
        x, s = ex.data[:, j], ex.values[:, j]
        c = ex.data[:, names.index(color_by[f])]
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(x)) if f == "NumOfProducts" else 0
        sc = ax.scatter(x + jitter, s, c=c, cmap="coolwarm", s=7, alpha=0.6, lw=0)
        ax.axhline(0, color=GREY, lw=1)
        ax.set(xlabel=f, ylabel="SHAP value (log-odds of churn)" if ax is axes[0] else "",
               title=f"{f}  (colour: {color_by[f]})")
        if f == "Balance":
            ax.xaxis.set_major_formatter(_eur)
        cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.05)
        cb.set_label(color_by[f], fontsize=8)
    fig.tight_layout()
    return fig


def shap_waterfall(ex_row, title: str):
    import shap
    shap.plots.waterfall(ex_row, max_display=8, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.2, 4.4)
    fig.axes[0].set_title(title, loc="left", fontweight="bold", fontsize=11)
    return fig


def persona_heatmap(ex, mask, labels, profile: pd.DataFrame, names: dict[int, str], top_n: int = 8):
    """Mean SHAP value per persona (rows) and feature (columns): why each group is at risk."""
    V = pd.DataFrame(ex.values[mask], columns=ex.feature_names)
    V["persona"] = labels
    M = V.groupby("persona").mean()
    cols = M.abs().max().sort_values(ascending=False).head(top_n).index
    M = M.loc[profile["persona"], cols]
    fig, ax = plt.subplots(figsize=(1.05 * top_n + 2.5, 0.75 * len(M) + 1.6))
    lim = np.abs(M.values).max()
    im = ax.imshow(M.values, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(M)), [f"{names[p]}\n{n} customers, {c:.0%} churned" for p, n, c in
                                   zip(profile["persona"], profile["customers"], profile["actual_churn"])])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(M.values[i, j]) > lim * 0.55 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04).set_label("mean SHAP (log-odds)", fontsize=8)
    ax.set_title("High-risk customers fall into groups at risk for different reasons", loc="left")
    return fig


def profit_curves(curves: dict[str, pd.DataFrame], markers: dict[str, tuple[int, float]], ceiling: float):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for (name, c), col in zip(curves.items(), [ORANGE, BLUE]):
        ax.plot(c["contacted"], c["profit"], color=col, lw=2, label=name)
        k = int(c["profit"].idxmax())
        ax.plot(c.loc[k, "contacted"], c.loc[k, "profit"], "o", color=col, ms=6)
    ax.axhline(ceiling, color=AQUA, ls="--", lw=1.2, label="hindsight ceiling")
    ax.axhline(0, color=GREY, lw=1)
    for (name, (n, prof)), (dx, dy) in zip(markers.items(), [(10, -40), (25, -48), (-20, 30)]):
        ax.plot(n, prof, "D", color=INK, ms=6)
        ax.annotate(f"{name}\n{_eur(prof)}", (n, prof), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.yaxis.set_major_formatter(_eur)
    ax.set(xlabel="customers called (holdout of 2,000), in ranking order", ylabel="realised campaign profit",
           title="Rank by expected value, not by probability, and stop where the curve peaks")
    ax.legend(loc="lower right")
    return fig


def policy_bars(table: pd.DataFrame):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = []
    for n in t["policy"]:
        colors.append(AQUA if "Hindsight" in n else ORANGE if "0.5" in n else BLUE if "Value" in n else GREY)
    ax.barh(t["policy"], t["profit"], color=colors)
    for i, (pr, k) in enumerate(zip(t["profit"], t["targeted"])):
        ax.text(max(pr, 0) + 3000, i, f"{_eur(pr)}  ({k:,} called)", va="center", fontsize=8.5)
    ax.xaxis.set_major_formatter(_eur)
    ax.set_xlim(0, t["profit"].max() * 1.35)
    ax.set(xlabel="realised profit on the 2,000 holdout customers",
           title="Same model, different rule: the cut-off is worth more than the tuning")
    return fig


def sensitivity_heatmap(grid: pd.DataFrame):
    piv = grid.pivot(index="save_rate", columns="value_multiplier", values="gain_vs_source")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    lim = np.abs(piv.values).max()
    # every cell is a gain; a diverging map would invite looking for losses that are not there
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=lim * 1.15, aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1]), [f"×{m:g}" for m in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{s:.0%}" for s in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, _eur(piv.values[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if piv.values[i, j] > lim * 0.6 else INK)
    ax.grid(False)
    ax.set(xlabel="customer value vs. assumption (×1 = base case)", ylabel="save rate of the offer",
           title="Value-aware rule minus the 0.5 cut-off, under other assumptions")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
    cb.ax.yaxis.set_major_formatter(_eur)
    return fig


def value_concentration(conc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(conc["share_of_customers"], conc["share_of_value_at_risk"], color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    for q in (0.1, 0.2):
        v = np.interp(q, conc["share_of_customers"], conc["share_of_value_at_risk"])
        ax.plot(q, v, "o", color=BLUE)
        ax.annotate(f"top {q:.0%} of customers\nhold {v:.0%} of value at risk", (q, v),
                    xytext=(14, -30), textcoords="offset points", fontsize=8.5)
    ax.set(xlabel="share of customers (highest expected loss first)",
           ylabel="share of expected value lost", title="Value at risk is concentrated")
    ax.xaxis.set_major_formatter(_pct); ax.yaxis.set_major_formatter(_pct)
    return fig
