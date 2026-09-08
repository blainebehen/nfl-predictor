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
| this model            | 0.6406 | 0.6310   |
| Vegas closing line    | 0.6611 | 0.6140   |

Elo closes about 78% of the log-loss gap between an uninformed prediction and
the market, using nothing but who played whom and who won. The model is well
calibrated: bucketed by predicted probability, predicted and observed win
rates agree within 0.017 across the buckets holding most of the games.

## Files

- data.py — loads nflverse schedules via nflreadpy, merges relocated
  franchises, sorts chronologically
- elo.py — the model, hyperparameter search, calibration check, and Vegas
  benchmark
- RESULTS.md — findings log, including negative results and known limitations

## Running it

    pip install nflreadpy pandas numpy scipy
    python elo.py

Takes a few minutes — the grid search replays all 7,276 games 750 times.
