# 06 · Comparing two models honestly: the paired bootstrap

**The one idea:** to know whether model A beats model B on a holdout, resample the *same*
customers many times, score both on each resample, and look at the spread of the *difference*.

## The anchor
To see which of two runners is faster, race them on the same tracks. Comparing one runner's time on
a hilly course with the other's on a flat one tells you about the courses.

## Why "paired"
Some holdout customers are easy to rank and some are hard. Both models face the same ones on each
resample, so that shared difficulty cancels in the difference. The two separate AUC intervals
(0.839–0.880 and 0.847–0.888) overlap heavily and look like "no difference". The paired interval
for the difference, +0.001 to +0.015, excludes zero.

## Stratified resampling
Each resample keeps the holdout's 407 churners and 1,593 stayers (resampling within each group),
so no resample is ever all one class and AUC is always defined.

## The same idea for money
`business.bootstrap_policy_delta` applies it to profit: per-customer profit under policy A minus
policy B, resampled. That gives the +€44k to +€89k interval for the rule change and −€10k to +€6k
for the model change.

## Where it becomes code
`src/evaluate.py: _resample_idx, bootstrap_ci, paired_delta`; `src/business.py:
bootstrap_policy_delta`.
