# 08 · SHAP: splitting one prediction among the features

**The one idea:** SHAP gives each feature a share of a single prediction, so that base rate +
shares = this customer's predicted risk (in log-odds), and the shares are fair in a precise
game-theory sense (Shapley values).

## The anchor
Three people share a taxi to different stops. A fair split of the fare averages each person's extra
cost over every possible order in which they could have joined. SHAP does that for features.

## Three ways it's used
1. **Global:** mean |SHAP| ranks features. Age, products and activity lead; salary, tenure and card
   are near zero, matching lesson 01's effect sizes.
2. **Shape:** a dependence plot puts the feature's value on x and its SHAP value on y. Because the
   trees train on raw features, x is in years and euros. The source standardised first, so its axes
   were z-scores.
3. **Local:** a waterfall for one customer, compressed into a reason code such as "3 products (+1.77);
   age 57 (+1.60); inactive (+0.45)", which an advisor can open a call with.

## Two things it is not
- **Not causal.** It explains the model. Lesson 02's confound shows the model's split of credit is
  partly arbitrary.
- **Not monotone by default.** The source read the beeswarm as "more products → more churn". The
  dependence plot shows 1 product (+0.39), 2 (−0.72), 3–4 (+1.9 to +2.2). A colour-coded beeswarm
  hides U-shapes; a dependence plot shows them.

## Where it becomes code
`src/explain.py: shap_explanation, global_importance, reason_codes`; figures in
`src/figures.py: shap_beeswarm, shap_dependence, shap_waterfall`.
