# 00 · A prediction is not a decision

**The one idea:** a churn model answers "how likely is this customer to leave?" A retention
team needs an answer to "should we call this customer?" The second needs the first plus money.

## The anchor
A weather forecast says 30% chance of rain. Whether you take an umbrella depends on what getting
wet costs you. Same forecast, different decisions for someone in a suit and someone in a swimsuit.

## In this project
The source notebook stops at the forecast: it outputs probabilities and reports a classification
at 0.5. A cut-off of 0.5 means "act when leaving is more likely than staying", which is only right if
a missed churner and a wasted call cost the same. Here they don't: a lost depositor is worth
thousands of euros, a call costs €130.

So the project has two halves:
1. **Predict** (§4–7): a probability per customer, as accurate and well calibrated as possible.
2. **Decide** (§8): combine that probability with what each customer is worth and what a call costs.

The surprise in this project is which half carries the value. Improving the model added 0.008
AUC and about €0. Improving the decision added €65k per 2,000 customers.

## Where it becomes code
- `src/models.py`: the prediction half.
- `src/business.py`: the decision half. Its module docstring states the inequality the whole
  decision rests on.
