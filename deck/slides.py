"""Slide content: titles state the claim, bodies stay short, notes are what I would say.

Every number here appears identically in notebooks/00_bank_churn_final.ipynb, where the final
cell asserts it against the computation. If the notebook's numbers move, change them here too.
"""

TITLE_CLAIM = "Same churn model, better rule:\n+43% retention profit"
TITLE_SUB = ("A Kaggle churn notebook reproduced, audited, and taken from a prediction to a decision\n"
             "10,000 bank customers · LightGBM + SHAP · value-aware targeting")
TITLE_NOTES = (
    "I took a public Kaggle notebook that predicts bank churn with LightGBM and explains it with SHAP. "
    "I reproduced it on the same split, checked it, and then asked the question it stops short of: "
    "if a retention team acts on these scores, whom should they call? The headline is that tuning the model, "
    "where most churn notebooks spend their effort, is worth about zero euros here, while changing the decision "
    "rule on the very same model raises campaign profit by 43 percent on the held-out customers.")

# (title, body, notes, figure filename or None)
SLIDES = [
    (
        "The question is whom to call, not how well we can predict",
        "• A retail bank loses ~1 in 5 customers (20.4%)\n"
        "• The retention team can call customers and offer a €100 fee waiver\n"
        "• Decision: for each customer, is the call worth its cost?\n\n"
        "Data: 10,000 real customers (FR / DE / ES), public CC0. The parent of the Kaggle competition file\n"
        "Source notebook: 80/20 split, LR / RF / LightGBM, SHAP. It stops at probabilities",
        "The source notebook answers 'how well can churn be predicted'. A client asks a different question: "
        "who do we call, and is it worth it. I used the public original dataset, ten thousand customers of a "
        "European bank, because the competition file is a noise-added derivative behind a rules gate. That "
        "means absolute AUCs differ from the source's 0.93, so every comparison I show is like for like: the "
        "source pipeline re-run on this data, same split, same folds.",
        None,
    ),
    (
        "Churn follows products, age, country and activity, and two of those are not straight lines",
        "Products U-shaped (1 → 28%, 2 → 8%, 3–4 → 83–100%) · age peaks at 56% in the 50s\n"
        "Germany 32% vs 16–17% · inactive 27% vs 14% · tenure, salary, card: no signal",
        "Before any model, the EDA shows the shapes. Product count is U-shaped: two products is the safest "
        "state, three or four is almost certain churn. Age is a hump peaking in the fifties. Those two shapes "
        "matter because a plain logistic regression treats them as straight lines, which is exactly why the "
        "source's logistic regression scores so badly. The bars carry Wilson intervals, so the 60 customers "
        "with four products are shown with the uncertainty they deserve. And there is a confound: every German "
        "customer holds a balance, so balance and country effects are tangled.",
        "01_churn_by_segment.png",
    ),
    (
        "Tuning buys one point of AUC; respecting the data's shapes buys nine",
        "• Source LightGBM: 0.858 CV AUC → tuned: 0.867\n"
        "• Holdout: 0.860 → 0.868, Δ +0.008 (95% CI +0.001 to +0.015)\n"
        "• Logistic regression 0.76 → 0.85 with a spline on age + products as categories\n"
        "• Fixed a search-space bug: subsample without subsample_freq is a no-op in LightGBM\n"
        "• Calibrated (ECE ≈ 2 pts); equal AUC for women and men",
        "Every candidate changes one thing so its effect is readable. Optuna tuning, which the source wrote but "
        "never ran, adds under a point of AUC: real, the paired bootstrap interval excludes zero, but small. "
        "The biggest jump in the table is the interpretable scorecard: the same logistic regression goes from "
        "0.76 to 0.85 just by being told that age bends and that product count is a category. That came from "
        "the EDA, not from compute. I also checked calibration, because the next step multiplies probabilities "
        "by euros, and fairness by country and gender.",
        "03_model_comparison.png",
    ),
    (
        "The second product is the most protective thing a customer holds",
        "1 product raises risk, 2 lowers it, 3–4 raise it sharply: 'more products → more churn' is wrong for 97%\n"
        "From ~45, inactive customers (blue) sit well above active ones of the same age",
        "These are SHAP dependence plots in real units, which is why I trained the trees on unscaled features. "
        "The source notebook concluded that more products increase churn. The plot shows that is only true past "
        "two; for the 97 percent of customers holding one or two products it is the opposite. The age panel "
        "shows an interaction: from about 45, inactive customers sit well above active ones of the same age. "
        "One caution I always add: SHAP explains the model, not the world. These are associations until tested.",
        "07_shap_dependence.png",
    ),
    (
        "High-risk customers are at risk for different reasons, so they need different actions",
        "Older, disengaged → re-engage  ·  German single-product → second-product offer\n"
        "3+ products (87% churned) → not a call list, an investigation  (k-means on SHAP, k by silhouette)",
        "I clustered the explanations of the 500 riskiest holdout customers rather than their raw features, so "
        "the groups are defined by why the model flags them. Three personas came out. The third one is the "
        "slide I would bring to a client: customers who bought three or more products and then almost all left. "
        "A model-driven campaign would call them first because they score highest, but a fee waiver will not fix "
        "whatever drove them out. That is a product or mis-selling question for the business, not for the call centre.",
        "09_personas.png",
    ),
    (
        "Same model, different rule: +€65k on 2,000 customers",
        "Call customer i if  p × save rate × value > cost\n\n"
        "• 0.5 cut-off (source): €150k, misses half the churners\n"
        "• Break-even cut-off p ≥ 0.18: €197k\n"
        "• Value-aware rule: €215k, +43% (95% CI +€44k to +€89k)\n"
        "• Tuned model under the same rule: −€2k (noise)\n"
        "• 70% of the hindsight ceiling, up from 49%",
        "Here is the core result. Value is three years of deposit margin plus product fees; a call costs 130 euros "
        "including the waiver, which goes to everyone called; and the offer saves 20 percent of would-be churners. "
        "The source model at 0.5 earns 150 thousand on the 2,000 holdout customers. The same model with the "
        "value-aware rule earns 215. The bootstrap interval on that gain is 44 to 89 thousand. Swapping in my "
        "tuned model under the same rule changes nothing. So the value is in the decision, not the model.",
        "10_policy_profit.png",
    ),
    (
        "Rank by expected value, not by probability, and stop where the curve peaks",
        "• Ranking by expected value peaks at €219k after 546 calls\n"
        "• Ranking by probability peaks at €199k after 695 calls\n"
        "• The rule stops near the peak without seeing outcomes\n"
        "• Top 20% of customers hold 67% of expected value lost",
        "If you call customers in order and track realised profit, ranking by expected value beats ranking by "
        "probability everywhere, and peaks higher with fewer calls. The value-aware rule lands near that peak "
        "without having seen who churned, because it stops exactly where the next call's expected gain turns "
        "negative. Value at risk is concentrated: a fifth of customers hold two thirds of it.",
        "11_profit_curves.png",
    ),
    (
        "The rule wins under every assumption tested; only the size of the win moves",
        "• 25 scenarios: save rate 10–40% × customer value ×0.5–×2\n"
        "• Gain over the 0.5 cut-off: €7k to €505k, never negative\n"
        "• Only save rate × value matters, so a client estimates one number\n"
        "• Lead with the direction, give the size as a range",
        "The money rests on assumptions, so I swept the two most uncertain. The value-aware rule beats the 0.5 "
        "cut-off in all 25 scenarios. The size of the gain depends on the assumptions; the direction does not. "
        "The grid repeats along its diagonals because the rule only uses save rate times value, which simplifies "
        "the conversation with a client: they only need to estimate one number.",
        "12_sensitivity.png",
    ),
    (
        "What I'd do: change the rule now, prove the save rate with a test",
        "1. Switch to the value-aware rule: +43% campaign profit, no new model\n"
        "2. Split the call list by persona: different scripts, different offers\n"
        "3. Send 3+ product customers to the product team\n"
        "4. Randomised test: ~580 customers per arm to detect a 20% cut in churn\n"
        "5. If model-risk sign-off binds, ship the scorecard (−0.015 AUC)",
        "The recommendation is in that order on purpose. The rule change is free and is worth the most. "
        "Personas turn one call list into three playbooks. The 3+ product group is a root-cause question. "
        "The test matters because everything rests on the assumed 20 percent save rate, and because it is the "
        "only way to learn whether the call causes retention, which SHAP cannot tell us. With roughly 39 percent "
        "churn among targeted customers, about 580 per arm detects a 20 percent relative reduction.",
        None,
    ),
    (
        "Limitations: the money is modelled, the data is a snapshot",
        "• Value, cost and save rate are assumptions; direction robust, size not\n"
        "• No dates: cannot rule out timing leakage or model when customers leave\n"
        "• SHAP and personas are associations, not causes\n"
        "• 3–4 product group is 326 customers; cause unknown from this data\n"
        "• Not the competition file: absolute AUCs differ from the source's 0.93",
        "The honest limits. The euro figures are a model of the economics, which is why I lead with the "
        "sensitivity analysis. The data has no timestamps, so a production version would need point-in-time "
        "features. The personas describe the model, not causes. And because I used the clean public parent "
        "rather than the competition file, the AUCs are not comparable to the source's leaderboard number. "
        "Every comparison inside the project is like for like.",
        None,
    ),
    (
        "Appendix: what reproducing the source notebook turned up",
        "• Evaluates at 0.5 only; no decision layer\n"
        "• SHAP on standardised features; 'more products → more churn' misread\n"
        "• Optuna commented out, ROC still labelled 'Optimized'; subsample no-op\n"
        "• Dependence-plot text describes Age, code plots EstimatedSalary\n"
        "• No intervals, paired tests or calibration; gender unexamined\n\n"
        "Reproduce: python run_analysis.py  ·  github.com/Cena-S/bank-churn-retention-targeting",
        "If asked what I thought of the source: it is a clean, standard pipeline, and none of these issues is "
        "unusual for a competition notebook. But most of them would matter in front of a client, and the first "
        "one, stopping at a 0.5 cut-off, is worth 65 thousand euros per two thousand customers here. Everything "
        "regenerates from one command, and the notebook asserts every number quoted in its prose.",
        None,
    ),
]
