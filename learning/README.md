# Learning: bank churn, from prediction to decision

A short course on the ideas behind this project. Read in order; each lesson takes a few minutes and
points at the `src/` file where the idea becomes code. For a walk through the code itself, see
`../src-explained/`.

| # | Lesson | The one idea |
|---|---|---|
| 00 | [A prediction is not a decision](00_prediction_vs_decision.md) | A churn probability needs money attached before anyone can act on it. |
| 01 | [Churn rates and intervals](01_churn_rates_and_intervals.md) | Give every rate an interval, and read effect sizes, not p-values, at n = 10,000. |
| 02 | [Confounding](02_confounding.md) | No German customer has a zero balance, so the two effects can't be fully separated. |
| 03 | [Cross-validation and the holdout](03_cross_validation_and_holdout.md) | Choose with CV; report with a holdout scored once. |
| 04 | [Shapes, splines and categories](04_shapes_splines_and_categories.md) | Logistic regression scores 0.76 when fed straight lines and 0.85 when given the right shapes. |
| 05 | [Boosting and tuning](05_boosting_and_tuning.md) | Tuning bought one point of AUC, and fixed a no-op in the source's search. |
| 06 | [Comparing models honestly](06_comparing_models_honestly.md) | Bootstrap the *difference* on paired resamples, not two overlapping intervals. |
| 07 | [Calibration](07_calibration.md) | Ranking well isn't enough once probabilities are multiplied by euros. |
| 08 | [SHAP](08_shap.md) | A fair split of one prediction among the features; it explains the model, not the world. |
| 09 | [Personas](09_personas.md) | Cluster the *reasons* for risk to get groups that need different actions. |
| 10 | [Expected-value targeting](10_expected_value_targeting.md) | Call when p × save rate × value > cost: +43% on the same model. |
| 11 | [Sensitivity and testing](11_sensitivity_and_testing.md) | Present assumed money as a direction and a range, then measure it with a test. |

**The spine**, for a reader who won't read all twelve: **00** (why the decision matters),
**04** (why shapes beat tuning) and **10** (the rule that carries the result).
