# 10 · From probability to decision: expected-value targeting

**The one idea:** call a customer when p × save rate × value > cost; that gives every customer
their own break-even probability, and it earned 43% more than the 0.5 cut-off on the same model.

## The anchor
An insurer doesn't refuse every risky applicant; it prices risk times exposure. A 40% risk on a
mansion and a 40% risk on a shed are different decisions.

## The pieces (all in `config.py`, all assumptions)
- **value** = 3 years × (1.5% × balance + €60 × products): from €180 (empty account, one product) to over €11k.
- **cost** = €30 call + €100 fee waiver = €130, paid for *everyone* called, stayers included.
- **save rate** = 20% of would-be churners kept by the offer.

## Three rules, compared on what customers actually did
| Rule | Called | Profit |
|---|---|---|
| p ≥ 0.5 | 270 | €150k |
| p ≥ 0.175 (break-even for the average customer) | 677 | €197k |
| p × 0.2 × value > €130 | 659 | €215k |

Most of the gain comes from lowering the cut-off to where an *average* customer breaks even. The rest
comes from recognising that customers are not average: the value-aware rule reaches fewer churners
than the break-even rule (258 vs 321) and still earns more, because it skips churners not worth saving.

## Profit curves
Call customers in rank order and plot cumulative realised profit. Ranking by expected value peaks
higher (€219k after 546 calls) than ranking by probability (€199k after 695). The rule stops near
the peak *without* seeing outcomes, because it stops where the next call's expected gain turns negative.

## And the better model?
Under the same rule, the tuned model earns −€2k (CI −€10k to +€6k). Its extra AUC reorders
customers in a band where the decision doesn't change.

## Where it becomes code
`src/business.py: customer_value, expected_net, policies, policy_table, profit_curve`.
