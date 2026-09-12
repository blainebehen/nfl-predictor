# NFL win predictor — results log

Data: nflverse schedules via nflreadpy, 1999–2025, 7,276 completed games.
Validation: walk-forward by construction — each prediction uses only ratings
built from prior games, so every forecast is out-of-sample.
All probabilities are from the home team's perspective.

## Scoreboard

Held out on 2019–2025 (1,960 games), with all parameters tuned on earlier
seasons only:

| model                 | Acc    | L      |
|-----------------------|--------|--------|
| always predict 0.5    | 0.5000 | 0.6931 |
| always pick home team | 0.5631 | —      |
| Elo + MOV             | 0.6388 | 0.6374 |
| Elo + MOV + QB        | 0.6449 | 0.6302 |
| Vegas closing line    | 0.6611 | 0.6140 |

In-sample over all 7,276 games the same model reaches L 0.6249 / Acc 0.6462,
but that figure is the best of many trials on the scoring set and should be
read as optimistic.

theta* = (k=20, H=50, rho=0.50, mov=True, qb_scale=600, qb_alpha=0.02)

## Adopted

**Margin of victory.** +0.0059 in L. Each variant tuned on its own grid for
fairness; the MOV multiplier averages ~2.5, which drops optimal k from 48
to 20.

**Quarterback adjustment.** The largest single improvement in the project:
+0.0061 in-sample, +0.0071 held out, and positive in all six test windows
(see below). Each quarterback carries an EWMA of his EPA per dropback, and
the adjustment entering d is qb_scale × (Q_home − Q_away).

Starter is identified as the QB with the most attempts in each team-game.
Checked on 2024: in the 22% of team-games where two QBs appear, the second
threw a median 4% of attempts, so the rule picks the real starter almost
always. Value combines passing and rushing EPA — ignoring the run would
systematically underrate mobile quarterbacks — over dropbacks including
sacks. Coverage is 99.7% of team-games; the rest contribute no adjustment.

Sanity check on the measure: career leaders are Mahomes, Jackson, Manning,
Allen, Rodgers, Brees, Brady; trailers are Harrington, D.Carr, Sanchez. It
agrees with what any football fan would say.

**rho wants 0.50, not 0.33.** FiveThirtyEight's NFL Elo uses 0.33 offseason
reversion. This data prefers 0.50 in every run. The most strongly identified
of the parameters: 0.50 wins the top five rows of every tuning block.

**Relocations must be merged.** nflverse uses distinct codes before and after
a franchise moves (STL to LA, SD to LAC, OAK to LV). Unmerged, three phantom
teams converge to exactly 1500 and the surviving franchises lose their
pre-move history. Merged on the grounds that ratings measure roster and
coaching continuity rather than venue — a judgment call, not an obvious one.

## The QB adjustment across six windows

Each window tuned on everything strictly before it, then scored on the
window itself (qb_windows.py). diff = no-QB L − with-QB L:

| window    | n     | no QB  | with QB | diff    | tuned                |
|-----------|-------|--------|---------|---------|----------------------|
| 2004–2009 | 1,602 | 0.6249 | 0.6212  | +0.0037 | scale=200, α=0.05    |
| 2010–2015 | 1,602 | 0.6265 | 0.6183  | +0.0082 | scale=900, α=0.01    |
| 2016–2018 | 801   | 0.6271 | 0.6206  | +0.0064 | scale=900, α=0.01    |
| 2019–2021 | 821   | 0.6421 | 0.6300  | +0.0121 | scale=900, α=0.01    |
| 2022–2025 | 1,139 | 0.6339 | 0.6280  | +0.0059 | scale=600, α=0.02    |
| 2019–2025 | 1,960 | 0.6374 | 0.6302  | +0.0071 | scale=900, α=0.01    |

Every window positive, and the smallest gain exceeds the largest gain rolling
H managed anywhere. Five of six windows independently select α in {0.01,
0.02} and scale in {600, 900} — parameters replicating across independent
tunings is what a real signal looks like.

The gain is largest on 2019–2021, the window hardest for both models. That is
consistent with COVID protocols pulling starters out on short notice, which
is exactly what this feature catches and team Elo cannot.

## Rejected

**The team-baseline formulation of the QB adjustment.** The original design
compared each starter to a team baseline T, on the reasoning that team Elo
already absorbs a franchise's average QB quality, so only the deviation from
it is new information. Tuning drove T's update rate to zero at every grid
floor it was given, and at beta = 0 the baseline is a constant that cancels:

    (Q_home − T) − (Q_away − T) = Q_home − Q_away

Dropping T improved both in-sample and held-out loss and removed a parameter.
The double-counting worry was unfounded: once the adjustment enters d, team
ratings update against a prediction that already includes it, so they settle
lower for teams with good quarterbacks. Elo partitions the credit itself.

**Season-varying home-field advantage.** Estimated from the prior five
seasons' home-win rate, it looked promising — +0.0001 in-sample, +0.0033 held
out on 2019–2025. But the test window opens on the anomalous 2019–2021
seasons, so it needed checking elsewhere (holdout_windows.py):

| window    | n     | fixed L | rolling L | diff    |
|-----------|-------|---------|-----------|---------|
| 2004–2009 | 1,602 | 0.6269  | 0.6251    | +0.0017 |
| 2010–2015 | 1,602 | 0.6267  | 0.6266    | +0.0001 |
| 2016–2018 | 801   | 0.6267  | 0.6276    | −0.0008 |
| 2019–2021 | 821   | 0.6469  | 0.6418    | +0.0051 |
| 2022–2025 | 1,139 | 0.6339  | 0.6339    | +0.0000 |

The entire gain sits in one window. Weighting 2019–2021 against 2022–2025 —
(821 × .0051 + 1139 × 0) / 1960 = +0.0021 — roughly reproduces the headline
+0.0033, so the combined result is the anomaly diluted, not an independent
effect. One favourable window out of six is not evidence a method is better.
Not adopted; the code stays in elo.py so the comparison is reproducible.

Note these H comparisons predate the QB adjustment — both arms lacked it, so
the comparison was fair on its own terms, but it was run against a weaker
model than the current one.

**Rest and travel.** Both plausible a priori — a team off a bye against a
team on a short week should have an edge, and a coast-to-coast trip should
cost something. Neither survives inspection (rest_travel.py). Measured as
mean model residual S − E by bucket, so team quality is already removed:

| rest diff (days) | n     | resid   | t    |
|------------------|-------|---------|------|
| −30 to −6        | 344   | −0.0508 | −2.0 |
| −6 to −2         | 387   | −0.0187 | −0.8 |
| −2 to −0.5       | 517   | +0.0051 | +0.2 |
| −0.5 to 0.5      | 4,860 | −0.0007 | −0.1 |
| 0.5 to 2         | 453   | +0.0058 | +0.3 |
| 2 to 6           | 331   | −0.0230 | −0.9 |
| 6 to 30          | 386   | +0.0282 | +1.2 |

Scatter, not a gradient, and the leftmost cell has the wrong sign for a
fatigue story. The direct test — home off a bye against a team on short rest
— is −0.1 SE on 58 games.

Travel has one cell at +2.8 SE (1,500–2,000 miles) but the bucket above it
reverses to −1.1 on a larger sample. Six buckets tested; one clearing 2 SE is
what chance produces.

**The instructive part:** travel's raw home-win rate climbs cleanly from .548
to .615 across the first five buckets — exactly the pattern the feature
predicts. The residuals stay flat. Long trips are made by the same handful of
teams every year, so the raw gradient was team quality, not fatigue, and the
model already had it. The clearest example in this project of why raw rates
mislead.

**Late-season exclusion.** 15 of the 20 largest disagreements with the
closing line are Week 16–18 games, where seeding is locked and starters rest
— real contamination the market prices and Elo cannot see. But excluding
those games from the update (while still scoring predictions on them) made
things slightly worse: L 0.6314 vs 0.6310. Most late games still involve
teams competing, so blanket exclusion costs more signal than the
rested-starter games add noise. A clinch-status feature would be the
targeted fix.

## Correction: home-field advantage dipped, it did not collapse

An earlier version of this file claimed home-field advantage collapsed after
2020 and stayed low, based on the smoothed rolling-H table. **That was
wrong.** The raw per-season home-win rates (hfa_trend.py):

| period    | home-win rate |
|-----------|---------------|
| 1999–2018 | 0.573         |
| 2019–2021 | 0.512         |
| 2022–2025 | 0.552         |

2019–2021 came in at .521, .498, .516, then 2022–2025 recovered to .563,
.565, .547, .533. A three-season dip, not a regime change. The smoothed table
showed 2024 at H = 22.7 because the five-season window was still carrying
2019–2021 long after those seasons had passed.

Two lessons: look at the raw series before theorising about a smoothed one,
and 1999–2018 is noisier than it appears — individual seasons range from .539
to .614 with no trend, and per-season standard error is about 0.030.

A stepped-H variant (separate H either side of a 2020 break) was written and
discarded once the raw rates showed there is no break to model.

## Calibration

Predicted vs actual agree within 0.035 in every bucket and within 0.017
across the three largest (4,754 of 7,276 games). The largest deviation is in
the smallest bucket (n=367), about 1.5 SE — not significant. Measured
in-sample.

## Not yet tried

- Rolling team EPA (offense and defense) as features
- Separate offensive and defensive ratings
- Clinch-status feature for late-season games
- A joint tune of team and QB parameters — currently staged, with k/H/rho
  fixed at theta* while the QB grid runs
- Against-the-spread evaluation is written (ats.py) but not yet finalised
- What drove the 2019–2021 dip. Empty stadiums explain 2020 but not the
  seasons either side of it.
- Why 2019–2021 was harder to predict overall — both models degrade by
  ~0.015 in L on that window

## Live predictions

Week 1 2026 predictions committed before kickoff (predictions_2026.csv),
using theta* including the QB adjustment. An earlier pre-QB version of the
same week is in git history; it was replaced before any game was played, so
no forecast was revised with knowledge of a result.

score.py joins actual results and reports Acc and L for the model and the
closing spread on the same games, with a standard error on accuracy attached.
Sixteen games has an SE of about 12 points, so a 10-6 week and a 6-10 week
are both consistent with a 64% model. The record only becomes informative
around midseason.

### Inferring who starts

The QB adjustment needs to know the starter for an unplayed game, which is
not in the data. predict.py infers it from two sources: the current season's
roster says who is on the team, and the previous season's REG week 1-16
attempts say which of them starts. Attempts are counted wherever the player
threw them, so a quarterback who changed teams keeps his history — Rodgers on
Pittsburgh's roster is matched to his New York attempts.

**Weeks 17-18 must be excluded here too.** The naive rule — whoever started
most recently — picks whoever took snaps in Week 18, which for a team with
its seed locked is a third-stringer. That put C.Oladokun under center for
Kansas City and J.Stidham for Denver, moving those games by 80+ rating
points. Same rested-starter contamination found earlier in the disagreement
diagnostic, surfacing in a completely different part of the code.

One case the rule cannot fix: a backup who outthrew the starter because the
starter was injured. San Francisco picks M.Jones over B.Purdy and Cincinnati
picks J.Flacco over J.Burrow on raw attempts. Those need manual correction
via STARTER_OVERRIDES, with find_qb.py to look up player ids. Rookies who won
a camp battle have no attempt history and hit the same problem; predict.py
prints a warning naming any team whose presumed starter has zero prior
attempts.
