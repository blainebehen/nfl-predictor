# NFL win predictor — results log

Data: nflverse schedules via nflreadpy, 1999–2025, 7,276 completed games.
Validation: walk-forward by construction — each prediction uses only ratings
built from prior games, so every forecast is out-of-sample.
All probabilities are from the home team's perspective.

## Scoreboard

All from a single run of elo.py over 7,278 completed games, so the rows are
directly comparable:

| model                 | Acc    | L      |
|-----------------------|--------|--------|
| always predict 0.5    | 0.5000 | 0.6931 |
| always pick home team | 0.5631 | —      |
| Elo + MOV             | 0.6406 | 0.6310 |
| + QB adjustment       | 0.6433 | 0.6289 |
| + team EPA            | 0.6441 | 0.6266 |
| Vegas closing line    | 0.6610 | 0.6140 |

These are in-sample: hyperparameters were chosen as the best of hundreds of
trials on the same games, so they are optimistically biased. Held out on
2019–2025, with everything tuned on earlier seasons only:

| model       | held-out L |
|-------------|------------|
| Elo + MOV   | 0.6400     |
| + QB        | 0.6344     |
| + team EPA  | 0.6318     |

The model closes about 80% of the log-loss distance between an uninformed
prediction and the market. QB and EPA together account for roughly a third
of what Elo + MOV alone left on the table.

theta* = (k=20, H=50, rho=0.50, mov=True, qb_scale=600, qb_alpha=0.02,
          epa_scale=200, epa_alpha=0.15)

Held-out figures drift by a thousandth or two between runs as nflverse
refreshes and the current season accumulates. Compare features within a
single run, not across runs.

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

**Team EPA.** Elo updates on who won, scaled by margin of victory. EPA per
play measures how a team actually moved the ball, which is less noisy than
the scoreboard — a team that gains 6.5 yards a play and loses on a late
turnover played better than the result says.

Offence is passing + rushing EPA per play (receiving would double-count the
same plays); defence is not shipped and is recovered as the opponent's
offensive EPA in the same game. Both are EWMAs read before a game and
updated after. The adjustment is epa_scale × (net_home − net_away), where net
is offence minus defence allowed.

The residual test (epa_test.py) gives a monotone gradient across octiles of
the differential, with both tails past 3 SE:

| octile | resid   | t    |
|--------|---------|------|
| 1      | −0.0457 | −2.9 |
| 2      | −0.0622 | −3.9 |
| 3      | −0.0092 | −0.5 |
| 4      | −0.0059 | −0.4 |
| 5      | −0.0137 | −0.8 |
| 6      | −0.0014 | −0.1 |
| 7      | +0.0546 | +3.7 |
| 8      | +0.0526 | +4.0 |

Across six windows on top of the QB model (epa_windows.py): +0.0029,
+0.0020, +0.0013, +0.0011, +0.0034, and +0.0027 on the combined 2019–2025
window. Every window positive, and all six independently selected scale=200
— stronger evidence than the loss numbers, since a feature fitting noise
would produce scattered parameters.

The profile differs from the QB adjustment in an informative way. QB ranged
+0.0037 to +0.0121, large and variable, biggest in the COVID seasons when
starters moved around. EPA is +0.0011 to +0.0034, small and uniform. That is
what a refinement looks like against new information: EPA is not telling the
model something it did not know, it is telling it the same thing with less
noise. Consistent with the +0.796 correlation between the EPA differential
and the model's own probability.

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

**Everything else in the schedule data.** residual_scan.py runs every
nflverse schedule column through the same residual test: divisional game,
roof type, playing surface, day of week, week of season, temperature, wind,
and the over/under total. Across roughly fifty buckets the largest |t| is
1.8. That is fewer excursions than chance alone would produce, so schedule
metadata can be treated as exhausted.

One column is not flat. Bucketing by the closing spread gives a perfectly
monotone residual gradient with |t| up to 7.3:

| closing spread | n     | raw   | resid   | t    |
|----------------|-------|-------|---------|------|
| −30 to −7      | 521   | 0.234 | −0.0783 | −4.1 |
| −7 to −3       | 1,334 | 0.345 | −0.0900 | −6.9 |
| −3 to 0        | 690   | 0.468 | −0.0346 | −1.8 |
| 0 to 3         | 1,447 | 0.543 | −0.0096 | −0.7 |
| 3 to 7         | 1,938 | 0.675 | +0.0397 | +3.7 |
| 7 to 30        | 1,348 | 0.815 | +0.0759 | +7.3 |

This is not a feature. It says the home team beats expectation when the
market favours them and underperforms when it does not — a restatement of
"the market is better than this model," already measured directly as L
0.6302 against 0.6140. Any model worse than the market produces exactly this
gradient. It diagnoses the gap rather than closing it.

Blending the model's probability with the market-implied one would improve
log loss materially, and for a pure forecasting goal that is standard
practice. It is excluded here because it makes "can this model beat the
market" circular.

**Rest and travel.** Both are reasonable ideas with a clear mechanism behind
them — a team off a bye against a team on a short week should have an edge,
and a coast-to-coast trip should cost something. Neither survives inspection
(rest_travel.py). Measured as
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

| predicted | n     | predicted | actual |
|-----------|-------|-----------|--------|
| 0.0–0.3   | 467   | 0.237     | 0.269  |
| 0.3–0.4   | 748   | 0.354     | 0.346  |
| 0.4–0.5   | 1,281 | 0.454     | 0.444  |
| 0.5–0.6   | 1,576 | 0.551     | 0.554  |
| 0.6–0.7   | 1,546 | 0.648     | 0.636  |
| 0.7–0.8   | 1,151 | 0.745     | 0.740  |
| 0.8–1.0   | 509   | 0.845     | 0.875  |

Agreement is within 0.032 in every bucket and within 0.012 across the five
largest (6,302 of 7,278 games). Both largest deviations sit in the two
smallest buckets and are about 1.4 SE — not significant. Measured in-sample.

## Not yet tried

- Joint tuning of team, QB and EPA parameters — currently staged
- A logistic regression with all terms as features, rather than additive
  adjustments to a rating
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
