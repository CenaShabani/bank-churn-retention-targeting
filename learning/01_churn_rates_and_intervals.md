# 01 · A churn rate needs an interval, and a test needs an effect size

**The one idea:** a rate computed on 60 customers is far less certain than one on 5,000, and with
10,000 rows almost every difference is "significant", so read the effect size instead.

## The anchor
Flip a coin 4 times and get 4 heads: you wouldn't call the coin biased. Get 4,000 heads in 4,000
flips and you would. The rate is the same; the evidence is not.

## Intervals: Wilson, not the textbook formula
The textbook interval p ± 1.96·√(p(1−p)/n) collapses to zero width at p = 1. The 60 customers with
four products all churned, so it would claim certainty. The **Wilson score interval** stays sensible
near 0 and 1: for 60/60 it gives [0.94, 1.00]. Every bar in the segment figure carries one.

## Effect sizes
A p-value answers "could this difference be chance?" With 10,000 customers, tiny differences pass.
The effect size answers "how big is it?":
- **Cramér's V** for categories (0 = no association, 1 = perfect). Product count: 0.39.
- **Rank-biserial correlation** for numbers: the probability that a random churner ranks above a
  random stayer, rescaled to −1…1. Age: 0.46, so a churner is older than a stayer 73% of the time.

Tenure, salary and credit card come out near zero. SHAP later agrees, and two independent methods
agreeing is worth more than either alone.

## Where it becomes code
`src/eda.py`: `wilson`, `churn_by`, `cramers_v`, `association_table`.
