# 04 · Why the shape of a relationship decides which model can learn it

**The one idea:** logistic regression scored 0.76 not because it is weak but because it was told
that age and product count act in straight lines, and they don't. Told the right shapes, it
scores 0.85.

## The anchor
Fit a ruler to a hill. However you tilt it, it misses the top. Bend the ruler (a spline), or measure
each side separately (categories), and it fits.

## The two shapes
- **Age is a hump**: 8% churn under 30, 56% in the 50s, 10% past 70. A single coefficient can only
  say "older = riskier" or "older = safer". A **cubic spline** (`SplineTransformer`, 5 knots)
  replaces Age with a few smooth basis curves, so the regression can rise and fall while each piece
  keeps a coefficient.
- **Product count is U-shaped**: 28% / 8% / 83% / 100%. As a number it forces a monotone effect. As
  **categories** (one-hot), each count gets its own effect.

## Why trees didn't need this
A decision tree splits: "NumOfProducts ≤ 2?", "Age > 42?". Splits carve out humps and cliffs
naturally, which is why the source's tree models did fine and its regression didn't. It is also why
the engineered features added nothing to LightGBM but help the scorecard.

## Why keep the scorecard at all
0.853 vs 0.868 AUC, and every coefficient is a readable log-odds effect. In a bank, a model-risk
reviewer signs that off more easily than 400 trees.

## Where it becomes code
`src/models.py: scorecard_pipeline` (its docstring lists each shape and its fix), and
`src/features.py` for the flags.
