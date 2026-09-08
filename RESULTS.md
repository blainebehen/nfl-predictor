# NFL win predictor — results log

Data: nflverse schedules via nflreadpy, 1999–2025, 7,276 completed games.
Validation: walk-forward by construction — each prediction uses only ratings
built from prior games, so every forecast is out-of-sample.
All probabilities are from the home team's perspective.

## Scoreboard

| model                 | Acc    | L      |
|-----------------------|--------|--------|
| always predict 0.5    | 0.5000 | 0.6931 |
| always pick home team | 0.5631 | —      |
| Elo (tuned, MOV)      | 0.6406 | 0.6310 |
| Vegas closing line    | 0.6611 | 0.6140 |

Elo closes ~78% of the gap in log loss between an uninformed prediction and
the market, using nothing but who played whom and who won.

theta* = (k=20, H=50, rho=0.50, mov=True), interior on all three axes.

## Settled

**Margin of victory helps.** +0.0059 in L. Each variant tuned on its own grid
for fairness; the MOV multiplier averages ~2.5, which drops optimal k from 48
to 20.

**rho wants 0.50, not 0.33.** FiveThirtyEight's NFL Elo uses 0.33 offseason
reversion. This data prefers 0.50 consistently across the grid — more
forgetting between seasons. Interior optimum, not a boundary.

**Relocations must be merged.** nflverse uses distinct codes before and after
a franchise moves (STL to LA, SD to LAC, OAK to LV). Unmerged, three phantom
teams converge to exactly 1500 and the surviving franchises lose their
pre-move history. Merged on the grounds that ratings measure roster and
coaching continuity rather than venue — a judgment call, not an obvious one.

**Calibration is good.** Predicted vs actual agree within 0.035 in every
bucket, and within 0.017 across the three largest buckets (4,754 of 7,276
games). The largest deviation is in the smallest bucket (n=367), where 0.035
is ~1.5 SE — not significant. Caveat: measured in-sample, on the same games
the hyperparameters were tuned on; see Known limitation.

**Late-season exclusion does NOT help.** 15 of the 20 largest disagreements
with the closing line are Week 16–18 games, where seeding is locked and
starters rest — real contamination the market prices and Elo cannot see. But
excluding those games from the update (while still scoring predictions on
them) made things slightly worse: L 0.6314 vs 0.6310, Acc 0.6357 vs 0.6406.
Most late games still involve teams competing, so blanket exclusion costs
more signal than the rested-starter games add noise. The right fix is a
clinch-status feature, which needs standings and tiebreaker logic.

## Known limitation

Hyperparameters were tuned on the same 7,276 games used for evaluation, so
the reported L is the best of 375 attempts on this dataset and is
optimistically biased. With only three hyperparameters and a flat objective
surface (1st and 5th place differ by 0.0006) the effect should be small, but
it is not zero, and any feature-based model will need a genuine held-out test
period.

## Not yet tried

- Held-out test split (tune on 1999–2018, evaluate on 2019–2025)
- QB adjustment (roster data) — likely the largest remaining gain
- Rest days and travel distance
- Separate offensive and defensive ratings
- Clinch-status feature for late-season games
