# 09 · Personas: clustering *why*, not *who*

**The one idea:** cluster high-risk customers by their SHAP vectors, the reasons the model gives
for them, and you get groups that each need a different action.

## The anchor
Two patients with the same fever can have different infections. The temperature (the risk score)
says how worried to be; the diagnosis (the reasons) says what to prescribe.

## How
1. Take the top 25% of holdout customers by predicted risk (500).
2. Their SHAP matrix is 500 rows × 16 features, all in log-odds, so **no scaling is needed**. That
   is the practical advantage over clustering raw features, where euros and years would first have
   to be made comparable.
3. Run k-means for k = 3…6 and keep the k with the best **silhouette score** (how much closer each
   point is to its own cluster than to the next one). k = 3 wins (0.40).
4. Name each cluster by rules on its profile (`name_persona`), not by eye, so the name follows the
   data if the clusters change.

## The three
| Persona | Driver | Action |
|---|---|---|
| Older, disengaged (328) | Age +1.25 | re-engagement, life-stage products |
| German single-product (103) | one product +0.47, Germany +0.30 | a second product |
| 3+ products (69; 87% left) | products +1.96 | investigate, don't call |

The third is the client insight: the highest scores, and the group a fee waiver is least likely to fix.

## Where it becomes code
`src/explain.py: personas, persona_profile, name_persona`; `src/figures.py: persona_heatmap`.
