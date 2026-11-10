# `src/eda.py`

## What this module is for
The statistics behind the EDA: churn rates with intervals, the association tests with effect
sizes, age bands, and the data-quality table. Only the notebook's §2–3 call it.

## Public surface
- `wilson(k, n, z=1.96) -> (low, high)`: vectorised Wilson score interval.
- `churn_by(df, col, name=None) -> DataFrame` indexed by level, with `churned, customers,
  churn_rate, ci_low, ci_high`. `col` may be a column name or a Series (e.g. age bands).
- `cramers_v(table) -> float`.
- `association_table(df) -> DataFrame`: `feature, test, p_value, effect, effect_size`, sorted by |effect|.
- `age_bands(age) -> Categorical`; `data_quality(df) -> DataFrame` of `check, value`.

## Walkthrough
`association_table` chooses the test by feature type. For categorical, binary and small-integer
features it uses a chi-square test of independence with Cramér's V. For continuous ones it uses
Mann-Whitney U with the rank-biserial correlation, computed as 2U/(n₁n₂) − 1. Results are sorted by
absolute effect size, not p-value, because at n = 10,000 the p-values barely discriminate.

## The non-obvious bits
- **Wilson, not Wald**: the 4-product group is 60/60. Wald gives [1, 1]; Wilson gives [0.94, 1].
- Floating point puts Wilson's upper bound at 1 − 1e-16 for p = 1, below the rate itself. The
  figure code clips the error bar at 0 for that reason (`figures.churn_by_segment`).
- **Mann-Whitney, not a t-test**: Balance is a spike at 0 plus a bump near €120k, nothing like
  normal.
- `Tenure` and `NumOfProducts` are treated as categories in the chi-square test, so a U-shape
  counts as association. A correlation coefficient would miss it.
- `correction=False` in chi-square: the Yates correction is for 2×2 tables with small counts, and
  would make Cramér's V inconsistent across tables of different shapes.

## How it can break
`churn_by` with a level holding zero customers would divide by zero; `groupby(observed=True)`
prevents empty categorical levels from appearing. Tests pin Wilson's bounds and `churn_by`'s counts.
