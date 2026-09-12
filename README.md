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

Teams that have secured a playoff berth rest starters in the last few
weeks. The market prices it; a rating system cannot see it. The first
version of this feature failed the residual test flat — and failed it for
an instructive reason. It scored "nothing at stake" as one thing, pooling
eliminated teams with seed-locked teams. Those move in opposite
directions, and the pool was 98% eliminated teams, so they cancelled.

Split apart, clinched teams underperform their rating by 0.058 in win
probability (t +2.8, n=484), consistent across home and away and across
both halves of the sample, while eliminated teams do not.

The model term docks rating points from whichever side has clinched, and
is inert outside REG weeks 15+. Across the same six windows: four of six
positive under per-window tuning, five of six at a fixed scale, and a
smooth in-sample loss curve with a clean interior minimum at 50 rating
points. Whole-sample L 0.6267 → 0.6261, accuracy 0.6441 → 0.6454.

That is stronger than the rejected features and weaker than the adopted
ones, so it ships as a flag — `CLINCH_SCALE` in elo.py, used by nothing.
RESULTS.md has the full case both ways. 2026 provides a seventh window.

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
- clinch_windows.py / clinch_fixed.py — six-window validation, tuned and
  at a fixed scale
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
