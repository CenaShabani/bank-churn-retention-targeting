# `src/explain.py`

## What this module is for
SHAP explanations at three levels: global importance, a reason code per customer, and personas
built by clustering explanations. The notebook's §7 and the call list in §8 use it.

## Public surface
- `model_inputs(pipe, X) -> DataFrame`: X pushed through every pipeline step except the classifier.
- `shap_explanation(pipe, X) -> shap.Explanation`: TreeSHAP values in log-odds, with `data` and `feature_names`.
- `global_importance(ex) -> Series`: mean |SHAP|, sorted.
- `reason_codes(ex, top=3) -> list[str]`: plain-English top positive drivers per customer.
- `personas(ex, mask, k_range=range(3, 7)) -> (labels, best_k, silhouette_by_k)`.
- `persona_profile(ex, mask, labels, y, p) -> DataFrame`; `name_persona(row) -> str`.

## Walkthrough
`shap_explanation` explains the *classifier* on the *transformed* inputs, which are raw-unit
columns because the tree pipeline doesn't scale. That is why the plots read in years and euros.
`reason_codes` takes each customer's positive SHAP values in descending order and renders them
through `_readable`, so `IsActiveMember = 0` becomes "inactive member" and `Geography_Germany = 1`
becomes "lives in Germany". `personas` runs k-means (20 restarts, seeded) on the SHAP rows of the
masked customers for each k and keeps the k with the best silhouette. `persona_profile` reports
size, actual and predicted churn, the top mean-SHAP drivers and a few traits. `name_persona`
labels the profile by fixed rules.

## The non-obvious bits
- **The `ndim == 3` branch**: some SHAP versions return one slice per class for binary models;
  slice 1 is churn.
- **No scaling before k-means**, because every SHAP column is in log-odds. Clustering raw features
  would require choosing a scaling, and that choice would shape the clusters.
- **Names by rule, not by eye**: `name_persona` falls back to "Persona k" when no rule fits, so a
  re-run with different clusters can't inherit a wrong label.
- `persona_profile` reads `Gender_Male` and `Geography_Germany` by name, so it assumes the tree
  pipeline with gender included.

## How it can break
If the final model isn't a tree model, `TreeExplainer` raises; the scorecard would need
`LinearExplainer`. Tests pin `reason_codes` (it picks only positive drivers, in order) and each
`name_persona` rule.
