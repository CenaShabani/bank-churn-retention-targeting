# 02 · Confounding: when two variables tell the same story

**The one idea:** if every German customer holds a balance, a "balance effect" and a "Germany
effect" can't be fully separated, and a model will split the credit between them arbitrarily.

## The anchor
Ice-cream sales and drownings rise together. Neither causes the other; summer causes both. Here,
"Germany" is part of the summer behind the balance pattern.

## In this data
- 36% of customers have a balance of exactly 0. None of them are German.
- Pooled, zero-balance customers churn *less* (14% vs 24%).
- Within France and Spain the gap is smaller (4–6 points). Part of the pooled gap is the high-churn
  country being absent from the zero-balance group.

## Why it matters downstream
SHAP will report how *this model* divided the credit between `Balance` and `Geography_Germany`. A
different model on the same data could divide it differently and fit just as well. So the SHAP
section reads "the model uses", never "balance causes". The causal question, whether changing a
customer's balance would change their churn, needs an experiment.

## Where it becomes code
`src/eda.py: data_quality` (the "German customers with Balance = 0" row) and the confound table in
§3 of the notebook. `src/figures.py: shap_dependence` colours Balance by Germany for exactly this
reason.
