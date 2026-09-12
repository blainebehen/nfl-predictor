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

## Clinch status — a candidate, not yet adopted

The targeted fix named at the end of the late-season-exclusion entry
below. Two findings pointed at the same gap: 15 of the 20 largest
disagreements with the closing line are Week 16–18 games, and the naive
starter rule in predict.py picked third-stringers for Kansas City and
Denver because those teams had their seeds locked. Both are teams with
nothing to play for resting starters — real contamination the market
prices and a rating system cannot see.

Stake is computed per team entering a game, from games completed strictly
before it, using win totals only (`clinch.py`). Tiebreakers are ignored,
which makes the flags conservative. Sanity check on 2024 Week 18: the five
teams flagged live are exactly the five that still had something to play
for, and Kansas City — which had locked the AFC's top seed and sat
Mahomes — is flagged dead.

**The first version failed, and the failure was the interesting part.**
Bucketing the residual by a stake differential gave a perfectly monotone
raw gradient and no residual gradient at all:

| stake diff | n   | raw   | resid   | t    |
|------------|-----|-------|---------|------|
| −1.0       | 169 | 0.426 | +0.0444 | +1.2 |
| −0.5       | 229 | 0.463 | −0.0294 | −1.0 |
| 0.0        | 560 | 0.573 | −0.0030 | −0.2 |
| +0.5       | 244 | 0.656 | +0.0423 | +1.5 |
| +1.0       | 164 | 0.732 | +0.0018 | +0.1 |

That is the travel result again: a clean raw climb, flat residuals, max
|t| 1.5. On the reading this project has used twice before, it stops here.

It did not stop here because the feature was built wrong. Stake 0.0 pooled
two situations that are not alike — *eliminated* and *seed locked*. Only
the second rests starters; the first is just a bad team playing its normal
lineup, which the ratings already know about. And the pool was 98%
eliminated teams, so whatever the locked teams carried was buried. Split
apart, they move in opposite directions and had been cancelling:

| flag                  | n   | shortfall | t    |
|-----------------------|-----|-----------|------|
| eliminated            | 572 | −0.0290   | −1.6 |
| clinched a berth      | 473 | +0.0418   | +2.0 |
| locked the top seed   |  19 | +0.4234   | +4.1 |
| clinched, pooled      | 484 | +0.0575   | +2.8 |

Shortfall is signed so positive means the flagged team did worse than the
model predicted, pooled across home and away — two independent samples
that agree. The top-seed cell is enormous and n=19, which is the
single-cell excursion this project rejects on sight; the usable evidence
is the pooled clinched row. By era: +0.0743 (t +2.5) on 1999–2011 and
+0.0427 (t +1.5) on 2012–2025 — same sign, weakening.

So the model term flags clinched teams only, and docks
`clinch_scale × (clinched_away − clinched_home)` rating points. It is
inert outside REG weeks 15+, so even a real effect can only move
whole-sample L slightly.

**Six windows, scale tuned inside each training span (`clinch_windows.py`):**

| window    | n     | theta* | +clinch | diff    | tuned | late-only diff |
|-----------|-------|--------|---------|---------|-------|----------------|
| 2004–2009 | 1,602 | 0.6227 | 0.6227  | +0.0000 |   0   | +0.0000 on 288 |
| 2010–2015 | 1,602 | 0.6182 | 0.6176  | +0.0005 |  75   | +0.0025 on 288 |
| 2016–2018 |   801 | 0.6252 | 0.6254  | −0.0002 |  50   | −0.0007 on 144 |
| 2019–2021 |   821 | 0.6334 | 0.6325  | +0.0009 |  50   | +0.0048 on 160 |
| 2022–2025 | 1,139 | 0.6304 | 0.6303  | +0.0001 |  50   | +0.0004 on 255 |
| 2019–2025 | 1,960 | 0.6317 | 0.6312  | +0.0004 |  50   | +0.0021 on 415 |

Four of six positive, one exactly zero, one negative by 0.0002, and the
selected scale wandering across 0, 50 and 75. Held at a fixed 50 instead
(`clinch_fixed.py`) it is five of six, with a smooth in-sample loss curve
on late games — 0.6132, 0.6118, 0.6109, **0.6104**, 0.6109, 0.6141,
0.6235 across scales 0 to 160 — a clean interior minimum and symmetric
rise either side, which a feature fitting noise does not produce.
Spillover onto the 5,912 games the adjustment never touches is +0.0000.
That fixed-scale run is not a validation, though: 50 was chosen knowing
the whole sample. It says the effect exists, not that it could have been
found in advance.

## Clinch status, second pass: the flag was still wrong

"Clinched a berth" is not the population that rests starters. A team that
has secured a place but is still fighting for seeding has everything to
play for. The population is any team that **cannot improve its position
by winning** — whatever seed that is.

`clinch_wide.py` splits the clinched teams three ways, each a strict
subset of the last: `in` (seed still live), `nogain` (cannot climb, could
still fall), `locked` (rank cannot move in either direction). The
shortfall is a clean dose-response:

| flag                          | n   | shortfall | t    |
|-------------------------------|-----|-----------|------|
| in — clinched, seed still live| 341 | +0.0146   | +0.6 |
| nogain — cannot improve       | 129 | +0.0748   | +1.9 |
| locked — rank frozen          |  54 | +0.2426   | +3.8 |
| nogain + locked               | 175 | +0.1300   | +3.8 |

n counts games where exactly one side carries the flag, so the combined
row is smaller than the sum — nine games have it on both sides and carry
no contrast. By era on nogain+locked: +0.1407 (t +3.0) on 1999–2011,
+0.1182 (t +2.3) on 2012–2025 — same sign, similar size, much steadier
than the diluted version.

So the +0.0575 above was the real effect buried under 341 team-games of
nothing. The gradient the first residual test went looking for and could
not find does exist; it just needed the right variable.

**But sharpening the flag made the model worse, not better.** Six windows
on nogain+locked (`clinch_wide_windows.py`): four of six binary, three of
six graded, scales scattered across 50/80/115 and 115/155/200. Narrowing
multiplied the effect size by 2.3× and cut the flagged population by
3.2×, from 626 team-games to 193. The product shrank and the variance per
window grew.

## Clinch status, third pass: rank-frozen, adjust versus exclude

`locked` alone is 56 team-games in 55 games — 0.76% of the data, about
two team-games a season. `locked_test.py` runs both mechanisms on exactly
that population.

**Excluding them fails cleanly.** Predict and score the game but do not
learn from it (`skip_games` in run_elo, added for this):

| window    | theta* | −locked | diff    |
|-----------|--------|---------|---------|
| 2004–2009 | 0.6226 | 0.6244  | −0.0018 |
| 2010–2015 | 0.6182 | 0.6183  | −0.0001 |
| 2016–2018 | 0.6252 | 0.6254  | −0.0002 |
| 2019–2021 | 0.6334 | 0.6330  | +0.0005 |
| 2022–2025 | 0.6301 | 0.6309  | −0.0008 |
| 2019–2025 | 0.6315 | 0.6318  | −0.0003 |

One of six, whole sample 0.6267 → 0.6272, accuracy 0.6441 → 0.6429.

**This closes the late-season-exclusion question.** That earlier test
dropped all of REG weeks 17–18 — 508 games, 1,016 team-games, 7% of the
data — and came out slightly worse, leaving it open whether the idea was
wrong or merely too blunt. Dropping only the 55 games where a team's rank
is frozen is the sharpest possible version of it, and it fails the same
way. The idea is wrong: discarding a game costs more real signal than the
rested-starter noise it removes. Adjusting the prediction and refusing to
learn are not interchangeable.

**Adjusting on rank-frozen is the best clinch result in the project:**

| window    | theta* | +locked | diff    | tuned | late-only diff |
|-----------|--------|---------|---------|-------|----------------|
| 2004–2009 | 0.6226 | 0.6203  | +0.0024 |  130  | +0.0152 on 288 |
| 2010–2015 | 0.6182 | 0.6207  | −0.0025 |  260  | −0.0133 on 288 |
| 2016–2018 | 0.6252 | 0.6252  | −0.0000 |  130  | +0.0013 on 144 |
| 2019–2021 | 0.6334 | 0.6331  | +0.0004 |  130  | +0.0022 on 160 |
| 2022–2025 | 0.6301 | 0.6276  | +0.0025 |  130  | +0.0118 on 255 |
| 2019–2025 | 0.6315 | 0.6299  | +0.0016 |  130  | +0.0081 on 415 |

Still four of six — but **five of six windows independently selected
scale 130**, the first stable parameter any version of this feature has
produced, and the effect on late games is an order of magnitude larger
than the diluted flag managed. The single failure is the window that
picked 260 instead of 130: it doubled the adjustment and was punished for
it. With two flagged team-games a season, a training span offers about
ten games to estimate a scale from, which is not enough.

That is the whole problem in one line. The effect is confidently real and
too rare to calibrate.

## Clinch status: the decision, and what is wrong with the evidence

Not adopted. `CLINCH_SCALE = 130` in elo.py with the rank-frozen flag,
used by nothing.

The bar that admitted the QB adjustment and team EPA was *every window
positive*, with parameters clustering; the bar that rejected rolling H was
one favourable window out of six. Four of six with a stable parameter is
neither, and the whole-sample gain is bounded by the fact that the
adjustment is nonzero in 54 games out of 7,278.

**Two things are wrong with the evidence, and both matter more than the
window count.**

First, the pre-specified residual test came back flat and the analysis
continued anyway. The reason was sound — the variable pooled eliminated
and clinched teams, which move in opposite directions, and a variable like
that reads zero however strong either effect is. That was a design error
visible in advance, not a data-mining opportunity. But the error was only
noticed *because* the result was null. Had the pooled version come back
positive it would have shipped unexamined. Checking harder when the answer
is unwelcome is exactly the asymmetry the stopping rule exists to prevent.

Second, the six-window test has now been run on four different flags —
any-clinched, nogain+locked binary, nogain+locked graded, and locked. Each
additional variant makes a favourable result easier to reach by chance, so
four of six on the fourth attempt is weaker than four of six on the first
would have been. The scale clustering at 130 is the part worth trusting;
the window count is not worth what it looks like.

So everything here is hypothesis-generating rather than hypothesis-
testing, and this project does not adopt on that. What it does supply is a
fully specified hypothesis that no version of the feature has ever
touched: rank-frozen, adjust rather than exclude, scale 130. 2026 onward
is genuinely out of sample for it. At two team-games a season it will take
years to say anything — which is its own verdict on whether the feature is
worth carrying.

Remaining known weakness in the flag: tiebreakers are ignored, so a seed
settled on head-to-head rather than raw wins is not flagged. That
under-flags, so the true rank-frozen population is somewhat larger than 56
and the measured effect is if anything understated.
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
- ~~Clinch-status feature for late-season games~~ — built and tested
  across three passes; see the Clinch status sections. Held as a flag on
  the rank-frozen population at scale 130, not adopted
- ~~Widening the seed-locked flag beyond "clinched the #1 seed"~~ — done.
  Widening to 56 team-games gave the stable parameter; widening further to
  193 diluted it again
- Implementing the actual tiebreaker ladder, so a seed settled on
  head-to-head is flagged. The current flag under-counts rank-frozen teams
  by an unknown amount, and more of them is the only way this feature gets
  enough data to calibrate
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

### Two models forward, from Week 2

Team EPA was adopted in the backtest but never reached predict.py, so the
committed Week 1 forecast ran Elo + MOV + QB while every number above
describes Elo + MOV + QB + EPA. Fixed: predict.py now carries both sets of
ratings and writes both probabilities every week — `E` for the QB-only
model and `E_full` for theta* — and score.py reports each against the
other and against the closing line.

Week 1 is **not** back-filled. Two of its games had kicked off by the time
the fix landed, so a Week 1 `E_full` written now would be a forecast made
with knowledge of results, and even for the fourteen unplayed games it
would carry two games of information the committed `E` did not have. The
head-to-head therefore starts at Week 2, where both models are committed
from identical information. `E_full` is blank for Week 1 and those games
score the QB model alone.

What to expect: the two agree on the pick in all 16 Week 2 games and
differ by at most 0.055 in probability, which is what a +0.796 correlation
between the EPA differential and the model's own probability implies. The
backtest gap between them is 0.0026 in held-out L. A season of roughly 285
games cannot resolve a difference that size — the standard error on L over
285 games is around 0.03, an order of magnitude larger. Logging both
starts the clock; it does not settle anything this year, and a week where
one model looks better than the other is noise.

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
