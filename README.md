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

Held out on 2019-2025 (1,960 games), all parameters tuned on earlier seasons
only. Every prediction uses only ratings built from prior games, so the
backtest is walk-forward and leakage-free.

| model                 | Acc    | log loss |
|-----------------------|--------|----------|
| always predict 0.5    | 0.5000 | 0.6931   |
| always pick home team | 0.5631 | —        |
| Elo + MOV             | 0.6388 | 0.6374   |
| Elo + MOV + QB        | 0.6449 | 0.6302   |
| Vegas closing line    | 0.6611 | 0.6140   |

The model closes about 80% of the log-loss gap between an uninformed
prediction and the market, and the QB adjustment alone closes 30% of what
remained. It is well calibrated: bucketed by predicted probability, predicted
and observed win rates agree within 0.017 across the buckets holding most of
the games.

## The quarterback adjustment

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
- predict.py — forecasts upcoming games from the current ratings, inferring
  each team's starting QB from the roster and last season's attempts
- find_qb.py — looks up QB ids for manual starter overrides
- score.py — scores saved predictions against results and the closing spread
- RESULTS.md — findings log, including rejected features and a correction

## Running it

    pip install nflreadpy pandas numpy scipy
    python elo.py

Takes several minutes — the grid searches replay all 7,276 games many times.
