"""
Score saved predictions against results and against the closing spread.

This is the only genuinely out-of-sample test in the project. Everything in
elo.py is a backtest -- honest, walk-forward, but still measured on games
that existed when the model was built. These predictions were committed
before kickoff.

Read the weekly numbers with the sample size in mind. Sixteen games has a
standard error of about 12 percentage points on accuracy, so a 10-6 week and
a 6-10 week are both entirely consistent with a 64% model. The record only
starts to mean anything around midseason.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.stats import norm

SPREAD_SD = 13.5    # SD of (actual margin - closing spread), empirical


def load_predictions(path='predictions_2026.csv'):
    return pd.read_csv(path)


def attach_results(preds):
    """Join actual outcomes onto predictions. Unplayed games get NaN."""
    sched = nfl.load_schedules().to_pandas()
    cols = ['season', 'week', 'away_team', 'home_team', 'result']

    merged = preds.merge(
        sched[cols],
        left_on=['season', 'week', 'away', 'home'],
        right_on=['season', 'week', 'away_team', 'home_team'],
        how='left',
    )
    return merged.drop(columns=['away_team', 'home_team'])


def outcomes(df):
    """S from result: 1 home win, 0.5 tie, 0 away win."""
    return np.where(df.result > 0, 1.0, np.where(df.result == 0, 0.5, 0.0))


def acc(p, S):
    return ((p - 0.5) * (S - 0.5) > 0).mean()


def ll(p, S):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(S * np.log(p) + (1 - S) * np.log(1 - p)).mean()


def score_block(df, label):
    """Model vs closing spread on the same games."""
    S = outcomes(df)
    E = df.E.values
    V = norm.cdf(df.spread.values / SPREAD_SD)

    n = len(df)
    se = np.sqrt(0.25 / n)
    print(f'\n{label} — {n} games (SE on accuracy ≈ {se:.3f})')
    print(f'  model   Acc {acc(E, S):.4f}  L {ll(E, S):.4f}')
    print(f'  spread  Acc {acc(V, S):.4f}  L {ll(V, S):.4f}')
    print(f'  diff in L: {ll(V, S) - ll(E, S):+.4f} '
          f'(positive means the model beat the line)')


if __name__ == '__main__':
    df = attach_results(load_predictions())
    played = df[df.result.notna()].copy()

    print(f'{len(played)} of {len(df)} predicted games played')

    if played.empty:
        print('\nNothing to score yet — check back after Sunday.')
        raise SystemExit

    S = outcomes(played)
    played['hit'] = np.where((played.E - 0.5) * (S - 0.5) > 0, 'Y', 'n')

    show = played[['week', 'away', 'home', 'E', 'spread', 'qb_adj',
                   'result', 'hit']]
    print()
    print(show.to_string(index=False))

    score_block(played, 'all games so far')

    if played.week.nunique() > 1:
        for wk in sorted(played.week.unique()):
            score_block(played[played.week == wk], f'week {wk}')

    # Games where the model disagrees most with the line. An edge, if one
    # exists anywhere, should be largest here -- and these are genuinely
    # out of sample, unlike the backtested version in ats.py.
    implied = SPREAD_SD * norm.ppf(np.clip(played.E.values, 1e-6, 1 - 1e-6))
    edge = implied - played.spread.values
    big = np.abs(edge) >= 3

    if big.sum():
        sub = played[big]
        Sb = outcomes(sub)
        picked_home = edge[big] > 0
        covered = sub.result.values > sub.spread.values
        ats = (picked_home == covered).mean()
        print(f'\nlarge disagreements (|edge| >= 3 pts): {big.sum()} games, '
              f'{ats:.3f} ATS')
        print('  Break-even at -110 is 0.5238. Far too few games to mean '
              'anything yet.')
