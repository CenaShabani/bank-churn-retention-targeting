# `src/business.py`

## What this module is for
It turns probabilities into a decision and the decision into euros: customer value, targeting
policies, realised profit, bootstrap gains, profit curves, sensitivity and A/B test sizing. It
uses `config` only, and it carries the project's main result (§8).

## Public surface
- `customer_value(X, multiplier=1.0) -> ndarray`: HORIZON × (spread × balance + fee × products).
- `cost_per_target() -> float`: €130.
- `expected_net(p, value, save_rate, cost) -> ndarray`: p × s × v − c.
- `realised_profit(y, value, targeted, save_rate, cost) -> float`.
- `break_even_threshold(mean_value, save_rate) -> float`.
- `policies(y, p, value, mean_value_train) -> {name: bool mask}`; `policy_table(...) -> DataFrame`.
- `bootstrap_policy_delta(y, value, t_new, t_old) -> {delta, ci_low, ci_high, p_not_better}`.
- `profit_curve(y, score, value) -> DataFrame(contacted, share_contacted, profit)`.
- `value_at_risk_concentration(p, value)`, `sensitivity(...)`, `ab_test_size(p_control, relative_reduction)`.

## Walkthrough
The module docstring states the one inequality: target i if p_i × save × value_i > cost.
`realised_profit` scores a targeting mask against what *actually happened*. Only real churners
(y = 1) can be saved, and every targeted customer costs €130. `policies` builds six masks, all from
predictions and training-set statistics, never from y, except the hindsight ceiling, which is
labelled as such. `bootstrap_policy_delta` computes each customer's profit-if-targeted, multiplies
it by the difference of the two masks, and resamples customers. `profit_curve` sorts by any score
and accumulates realised gain. `sensitivity` re-runs `policy_table` over a grid. `ab_test_size` is
the standard two-proportion formula.

## The non-obvious bits
- **The break-even cut-off uses the training set's mean value**, so it is fixed before the holdout
  is seen.
- **Policies are looked up by position** (`names[2]`, `names[4]`) in the notebook and in
  `sensitivity`, because the break-even policy's name contains its computed threshold. Reordering
  `policies` would silently swap rows; the notebook's assert cell would catch the changed numbers.
- **Save rate × value is all that matters** to the rule, which is why the sensitivity grid is
  constant along its diagonals.
- The value model ignores discounting and the chance a saved customer leaves later. Both would
  scale value down, which is equivalent to a multiplier below 1 in the grid.

## How it can break
Negative or missing balances would give negative values; the data has none and `validate` forbids
missing values. Tests pin a hand-computed profit, the break-even formula, the hindsight mask, that
the profit curve ends at "call everyone", and a textbook sample size (50% → 40%: about 385 per arm).
