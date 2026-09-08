# NFL win predictor

An Elo rating model for NFL games, benchmarked against the Vegas closing line.

Each team carries a rating; before every game the rating difference (plus a
home-field offset) is converted to a win probability, and afterward both
ratings are updated in proportion to the prediction error. This is stochastic
gradient descent on log loss for a Bradley-Terry model — see the derivation
comments in elo.py.

## Results

7,276 games, 1999-2025. Every prediction uses only ratings built from prior
games, so the backtest is walk-forward and leakage-free.

| model                 | Acc    | log loss |
|-----------------------|--------|----------|
| always predict 0.5    | 0.5000 | 0.6931   |
| always pick home team | 0.5631 | —        |
| this model            | 0.6413 | 0.6309   |
| Vegas closing line    | 0.6611 | 0.6140   |

Elo closes about 79% of the log-loss gap between an uninformed prediction and
the market, using nothing but who played whom and who won. The model is well
calibrated: bucketed by predicted probability, predicted and observed win
rates agree within 0.010 across the buckets holding most of the games.

## A feature that only shows up out of sample

Letting home-field advantage vary by season — estimated from the prior five
seasons' home-win rate — is worth +0.0001 in-sample, which is nothing. Held
out on 2019-2025 it is worth +0.0033 and cuts the generalization gap by more
than a third.

The test window opens on 2019-2021, three seasons where home-field advantage
dipped well below its historical level (.512 against a .573 baseline). An
adaptive H tracks that; a constant fit on 1999-2018 cannot. In-sample
evaluation alone would have discarded the feature.

RESULTS.md has the full comparison, plus a correction: an earlier version of
this analysis read the dip as a permanent collapse, which the raw per-season
rates do not support — 2022-2025 recovered to .552.

## Files

- data.py — loads nflverse schedules via nflreadpy, merges relocated
  franchises, sorts chronologically
- elo.py — the model, hyperparameter search, held-out evaluation,
  calibration check, and Vegas benchmark
- hfa_trend.py — raw home-win rate by season, unsmoothed
- predict.py — forecasts upcoming games from the current ratings
- score.py — joins actual results onto saved predictions and scores them
  against the closing spread
- RESULTS.md — findings log, including negative results and corrections

## Running it

    pip install nflreadpy pandas numpy scipy
    python elo.py

Takes several minutes — the grid searches replay all 7,276 games about 1,650
times.
