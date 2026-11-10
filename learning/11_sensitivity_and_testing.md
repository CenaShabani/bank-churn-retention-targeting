# 11 · Assumed numbers: show where the answer flips, then go and measure

**The one idea:** a money figure built on assumptions is presented as a direction plus a range,
and the assumption that matters most is replaced by a randomised test.

## The anchor
An engineer doesn't know the exact load a bridge will carry, so they check that the design holds
across the plausible range, and then they load-test it.

## Sensitivity
Sweep save rate (10–40%) and customer value (×0.5 to ×2): 25 scenarios. The value-aware rule beats
the 0.5 cut-off in all 25, by €7k to €505k. The grid repeats along its diagonals because the rule
depends only on save rate × value, so the client needs to estimate one number: the expected value
saved per churner reached.

## The test
Everything leans on the 20% save rate, and on the assumption that the call *causes* retention,
which SHAP can't establish. So: randomise the call list into "call" and "don't call", and compare
churn.

Sample size for a two-proportion test: churn among targeted customers is 39%. Detecting a 20%
relative cut (39% → 31%) with α = 0.05 and 80% power needs ≈584 customers per arm. Detecting a 10%
cut needs ≈2,400 per arm, about four times as many. Halving the effect roughly quadruples the
sample, because n scales with 1/effect².

## Where it becomes code
`src/business.py: sensitivity, ab_test_size`; `src/figures.py: sensitivity_heatmap`.
