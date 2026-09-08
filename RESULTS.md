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
| Elo (tuned, MOV)      | 0.6413 | 0.6309 |
| Vegas closing line    | 0.6611 | 0.6140 |

Elo closes ~79% of the gap in log loss between an uninformed prediction and
the market, using nothing but who played whom and who won.

theta* = (k=20, rolling H, rho=0.50, mov=True)

## Settled

**Margin of victory helps.** +0.0059 in L. Each variant tuned on its own grid
for fairness; the MOV multiplier averages ~2.5, which drops optimal k from 48
to 20.

**rho wants 0.50, not 0.33.** FiveThirtyEight's NFL Elo uses 0.33 offseason
reversion. This data prefers 0.50 in every run — more forgetting between
seasons. The most strongly identified of the three parameters: 0.50 wins the
top five rows of every tuning block.

**Relocations must be merged.** nflverse uses distinct codes before and after
a franchise moves (STL to LA, SD to LAC, OAK to LV). Unmerged, three phantom
teams converge to exactly 1500 and the surviving franchises lose their
pre-move history. Merged on the grounds that ratings measure roster and
coaching continuity rather than venue — a judgment call, not an obvious one.

**Calibration is good.** Predicted vs actual agree within 0.028 in every
bucket and within 0.010 in the four largest (5,776 of 7,276 games). The
largest deviation is in the smallest bucket (n=399), about 1.1 SE — not
significant.

**Late-season exclusion does NOT help.** 15 of the 20 largest disagreements
with the closing line are Week 16–18 games, where seeding is locked and
starters rest — real contamination the market prices and Elo cannot see. But
excluding those games from the update (while still scoring predictions on
them) made things slightly worse: L 0.6314 vs 0.6310, Acc 0.6357 vs 0.6406.
Most late games still involve teams competing, so blanket exclusion costs
more signal than the rested-starter games add noise. The right fix is a
clinch-status feature, which needs standings and tiebreaker logic.

## Home-field advantage collapsed after 2020

Estimating H per season from the prior 5 seasons' home-win rate:

| season | H    |
|--------|------|
| 2000   | 69.3 |
| 2008   | 52.2 |
| 2015   | 51.1 |
| 2019   | 50.1 |
| 2020   | 42.6 |
| 2021   | 36.3 |
| 2022   | 27.2 |
| 2024   | 22.7 |
| 2025   | 26.4 |

Stable near 50 for fifteen years, then a sharp drop from 2020 onward. Home
field is now worth roughly a third of its 2000 value. The timing coincides
with the empty-stadium 2020 season, and it has not recovered.

Note this is a level shift, not a gradual trend, so a rolling average is an
imperfect tool — it smears the discontinuity across several seasons. An
explicit break at 2020 would fit the actual shape better and is untested.

## Held-out evaluation

Hyperparameters are chosen as the best of hundreds of trials on the scoring
set, so an in-sample L is optimistically biased. Measured by tuning on games
through 2018 and scoring only 2019–2025 (1,960 games):

| model      | in-sample L | held-out L | gap    |
|------------|-------------|------------|--------|
| fixed H    | 0.6310      | 0.6399     | 0.0089 |
| rolling H  | 0.6309      | 0.6366     | 0.0056 |

**The main finding: a feature worth nothing in-sample was worth 0.0033 out of
sample.** Rolling H improves in-sample L by 0.0001 — indistinguishable from
noise, and it would have been discarded on that evidence. Held out, it cuts
the generalization gap by more than a third. A constant H fit across 26
seasons is a fine compromise *for those seasons*; it fails only when asked to
predict a period whose home-field advantage differs from the historical
average, and that failure is invisible in-sample.

This also confirms the era-shift explanation for the gap. The remaining
0.0056 is some mix of residual era effects and genuine selection bias.

**Hyperparameters are stable across training windows.** Tuning on 1999–2018
and on 2012–2018 selected the identical theta (k=24, H=60, rho=0.50).
Overfitting to noise would move them.

**Window length is weakly identified.** The held-out rolling grid returns
L = 0.6289 for windows of 5, 8, and 12 alike — identical to four decimals. In
sample the spread is 0.0001. The model cares that H moves, not how fast, so
the specific window value should not be read as a tuned optimum.

## Not yet tried

- Explicit level shift in H at 2020 rather than a rolling average
- QB adjustment (roster data) — likely the largest remaining gain
- Rest days and travel distance
- Separate offensive and defensive ratings
- Clinch-status feature for late-season games

## Live predictions

Week 1 2026 predictions committed before kickoff (predictions_2026.csv).
score.py joins actual results and reports Acc and L for the model and the
closing spread on the same games. Sixteen games per week is a small sample —
the record only becomes informative around midseason.
