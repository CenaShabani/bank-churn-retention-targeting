# 07 · Calibration: do the probabilities mean what they say?

**The one idea:** a model can rank customers perfectly and still be wrong about the level of risk;
the moment probabilities are multiplied by euros, the level matters.

## The anchor
A forecaster who says "70%" on days it rains 40% of the time may still pick the rainier days
correctly. You'd rank the days fine and still pack the wrong things.

## How it's checked
Sort holdout customers into 10 groups of 200 by predicted risk. In each group, compare the average
prediction with the share who actually left. The **expected calibration error (ECE)** is the
size-weighted average gap: 0.024 for the tuned model, about 2 points.

The top decile is slightly *under*-predicted (76% predicted, 84% actual). For the decision that is
conservative: the riskiest customers are, if anything, worth a little more than the rule assumes.

## Why AUC can't see it
AUC only uses the order of the predictions. Halve every probability and AUC is unchanged, but every
expected value in §8 halves too. Brier score and log-loss do see it, which is why the model table
reports them.

## Where it becomes code
`src/evaluate.py: calibration_table, expected_calibration_error`; `src/figures.py: calibration`.
