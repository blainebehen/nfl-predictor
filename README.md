# NFL win predictor

An Elo rating model for NFL games, with margin-of-victory scaling and a
quarterback adjustment, benchmarked against the Vegas closing line.

Each team carries a rating; before every game the rating difference (plus a
home-field offset and a QB term) is converted to a win probability, and
afterward both ratings are updated in proportion to the prediction error.
This is stochastic gradient descent on log loss for a Bradley-Terry model —
see the derivation comments in elo.py.

Note this is not vanilla Elo. The margin-of-victory multiplier, offseason
reversion, and quarterback term are NFL-specific extensions; the QB term is
what separates these numbers from a textbook implementation.

## Results

7,278 games, 1999-2025. Every prediction uses only ratings built from prior
games, so the backtest is walk-forward and leakage-free.

| model                 | Acc    | log loss |
|-----------------------|--------|----------|
| always predict 0.5    | 0.5000 | 0.6931   |
| always pick home team | 0.5631 | —        |
| Elo + MOV             | 0.6406 | 0.6310   |
| + QB adjustment       | 0.6433 | 0.6289   |
| + team EPA            | 0.6441 | 0.6266   |
| Vegas closing line    | 0.6610 | 0.6140   |

Held out on 2019-2025, with all parameters tuned on earlier seasons, the
same three tiers give 0.6400, 0.6344 and 0.6318.

The model closes about 80% of the log-loss distance between an uninformed
prediction and the market. It is well calibrated: bucketed by predicted
probability, predicted and observed win rates agree within 0.012 across the
five buckets holding 87% of the games.

## Two features that worked

### The quarterback adjustment

Team Elo absorbs a franchise's average QB quality — a team that wins with an
elite starter accumulates rating for it. What Elo cannot see is who is
actually playing on a given Sunday. That blind spot is a large part of the
market's information advantage over a pure team rating.

Each quarterback carries a rating: an exponentially-weighted average of his
EPA per dropback, read before a game and updated after it. The adjustment
entering the prediction is the difference between the two starters' ratings.

Predicting an unplayed game means guessing who starts, which is inferred from
the current roster plus last season's attempts — see RESULTS.md, including
why weeks 17-18 have to be excluded from that inference.

Tested across six held-out windows, each tuned on data strictly preceding it:

| window    | no QB  | with QB | diff    |
|-----------|--------|---------|---------|
| 2004-2009 | 0.6249 | 0.6212  | +0.0037 |
| 2010-2015 | 0.6265 | 0.6183  | +0.0082 |
| 2016-2018 | 0.6271 | 0.6206  | +0.0064 |
| 2019-2021 | 0.6421 | 0.6300  | +0.0121 |
| 2022-2025 | 0.6339 | 0.6280  | +0.0059 |

Every window positive, with five of six independently selecting nearly the
same parameters.

### Team EPA

Elo updates on who won, scaled by margin of victory. EPA per play measures
how a team actually moved the ball, which is less noisy — a team that gains
6.5 yards a play and loses on a late turnover played better than the result
says. Offence and defence are tracked separately, defence being the
opponent's offensive EPA in the same game.

Positive in all six windows (+0.0011 to +0.0034 on top of the QB model), with
all six independently selecting the same scale. Smaller and steadier than the
QB gain, which is what a refinement looks like next to new information: EPA
correlates 0.80 with the model's own probability and improves the 20% where
they disagree.

## A feature in between: clinch status

Teams whose playoff seed can no longer move rest their starters in the
last few weeks. The market prices it; a rating system cannot see it.

Getting to that sentence took three passes, and the wrong turns are the
useful part. The first flag scored "nothing at stake", pooling eliminated
teams with seed-locked teams — opposite effects that cancelled, so the
residual test came back flat. The second flagged every team that had
clinched a berth, which buried the effect under 341 team-games of teams
still fighting for seeding. Only the third — rank frozen in both
directions — isolates it:

| flag                           | n   | shortfall | t    |
|--------------------------------|-----|-----------|------|
| clinched, seed still live      | 341 | +0.0146   | +0.6 |
| cannot improve, could fall     | 129 | +0.0748   | +1.9 |
| rank frozen both ways          |  54 | +0.2426   | +3.8 |

Two mechanisms were tested on that population. Docking rating points from
the frozen team wins four of six windows, and five of six independently
select 130 points — the only stable parameter any version produced.
*Excluding* those games from the update instead loses, one of six, which
settles an older open question: the earlier weeks-17–18 exclusion did not
fail for being too blunt. Discarding a game costs more signal than the
rested-starter noise it removes, however precisely it is aimed.

Not adopted. The adjustment is nonzero in 54 games out of 7,278 — about
two team-games a season — which is enough to measure the effect
confidently and not enough to calibrate it. `CLINCH_SCALE` in elo.py,
used by nothing. RESULTS.md has the full case, including what is wrong
with the evidence: the pre-specified test came back flat and the analysis
continued anyway, and four flag variants were tried before one looked
good.

## Features that did not survive the same test

Letting home-field advantage vary by season looked like a clear win: +0.0033
held out on 2019-2025. Run across the same six windows, the entire gain
turned out to sit in 2019-2021 — elsewhere the two schemes are within
+/-0.0017, and on 2016-2018 the rolling version is worse. Not adopted.

Rest and travel were rejected the same way. Travel's raw home-win rate climbs
cleanly with distance — exactly the pattern the feature predicts — but the
model residuals are flat. Long trips are made by the same few teams every
year, so the gradient was team quality, which the ratings already capture.

RESULTS.md has all three analyses, plus a correction: an earlier reading of
the smoothed data called the home-field dip a permanent collapse, which the
raw per-season rates do not support.

## Files

- data.py — loads nflverse schedules, merges relocated franchises
- elo.py — the model, hyperparameter search, held-out evaluation,
  calibration, and the Vegas benchmark
- qb_data.py — extracts each game's starting QB and his EPA per dropback
- qb.py — tunes the QB adjustment against a no-QB control
- qb_windows.py — the QB adjustment across six held-out windows
- holdout_windows.py — the same test applied to season-varying home field
- hfa_trend.py — raw home-win rate by season, unsmoothed
- rest_travel.py — residual analysis for rest and travel (both rejected)
- residual_scan.py — every schedule column tested the same way
- epa_test.py — residual analysis for rolling team EPA (adopted)
- epa.py / epa_windows.py — tuning and six-window validation for EPA
- clinch.py — playoff standings and clinch flags, leakage-free
- clinch_test.py — the residual test that came back flat
- clinch_probe.py — the same data split by flag, where the effect is
- clinch_windows.py / clinch_fixed.py — six-window validation of the
  clinched-a-berth flag, tuned and at a fixed scale
- clinch_wide.py — seed-range logic: in / cannot-improve / rank-frozen
- clinch_wide_windows.py — six windows on cannot-improve + rank-frozen
- locked_test.py — rank-frozen alone, adjusting versus excluding
- predict.py — forecasts upcoming games from the current ratings, running
  the QB-only and theta* models side by side, and inferring each team's
  starting QB from the roster and last season's attempts
- find_qb.py — looks up QB ids for manual starter overrides
- score.py — scores both saved models against results and the closing
  spread
- RESULTS.md — findings log, including rejected features and a correction

## Running it

    pip install nflreadpy pandas numpy scipy
    python elo.py

Takes several minutes — the grid searches replay all 7,276 games many times.
