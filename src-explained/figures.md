# `src/figures.py`

## What this module is for
Every figure in the project, one function each, plus `save()` which writes PNGs to
`outputs/figures/`. The notebook calls each function and saves the result; the deck reads only
those files. No figure is drawn anywhere else.

## Public surface
`save(fig, name)` and 13 figure builders, each returning a matplotlib Figure:
`churn_by_segment`, `effect_sizes`, `model_comparison`, `calibration`, `gains`, `shap_beeswarm`,
`shap_dependence`, `shap_waterfall`, `persona_heatmap`, `profit_curves`, `policy_bars`,
`sensitivity_heatmap`, `value_concentration`.

## Walkthrough
A shared `rcParams` block sets the portfolio's look: no top or right spines, light grid, left-aligned
bold titles. Colours have fixed roles: BLUE for this project's approach, ORANGE for the source or
common practice, AQUA for the hindsight ceiling, GREY for reference lines, RED for segments churning
at more than 1.5× the average. Each function takes already computed tables. None of them computes
statistics, so a figure can never disagree with the table printed beside it.

## The non-obvious bits
- **Titles state the claim** ("Rank by expected value, not by probability…"), matching the deck
  rule that a reader of titles alone gets the argument.
- `churn_by_segment` clips error bars at 0 because Wilson's upper bound at 60/60 is 1 − 1e-16.
- `shap_dependence` jitters NumOfProducts horizontally so four stacked columns of points become visible.
- `sensitivity_heatmap` uses a sequential map from 0, not a diverging one: every cell is a gain,
  and red would invite the reader to look for losses that aren't there.
- SHAP's own plots (`beeswarm`, `waterfall`) draw onto the current figure, so those builders grab
  `plt.gcf()` and restyle the title.
- `save` uses `bbox_inches="tight"` and a white background, so the PNGs drop cleanly onto slides.

## How it can break
SHAP plotting APIs change between versions. `shap==0.48.0` is pinned for that reason. The deck
builder fails loudly if a figure file is missing.
