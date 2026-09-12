"""
Score saved predictions against results and against the closing spread.

This is the only genuinely out-of-sample test in the project. Everything
in elo.py is a backtest -- honest, walk-forward, but still measured on
games that existed when the model was built. These predictions were
committed before kickoff.

TWO MODELS ARE TRACKED.

  E       Elo + MOV + QB. The Week 1 2026 model.
  E_full  theta* -- the same plus team EPA.

The backtest says E_full is better by 0.0026 in held-out L. Scoring both
forward on the same games is how that claim gets tested against games
neither model was tuned on. E_full is blank for Week 1, which was
committed before team EPA reached the live path; those games score the
QB model alone rather than being back-filled, since a forecast written
after kickoff is not a forecast.

Read the weekly numbers with the sample size in mind. Sixteen games has a
standard error of about 12 percentage points on accuracy, so a 10-6 week
and a 6-10 week are both entirely consistent with a 64% model. The record
only starts to mean anything around midseason. Separating two models that
differ by 0.003 in L takes far longer than one season -- the honest
expectation is that a single year cannot resolve them, and the point of
logging both is to start the clock, not to settle it in November.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.stats import norm

SPREAD_SD = 13.5    # SD of (actual margin - closing spread), empirical

MODELS = [('E', 'qb only'), ('E_full', 'theta* (+epa)')]


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
    """Each model vs the closing spread, on the games each one covers."""
    n = len(df)
    print(f'\n{label} — {n} games (SE on accuracy ≈ {np.sqrt(0.25/n):.3f})')

    for col, name in MODELS:
        if col not in df.columns:
            continue
        sub = df[df[col].notna()]
        if sub.empty:
            print(f'  {name:<16} —')
            continue
        S = outcomes(sub)
        E = sub[col].values
        note = '' if len(sub) == n else f'  [{len(sub)} of {n} games]'
        print(f'  {name:<16} Acc {acc(E, S):.4f}  L {ll(E, S):.4f}{note}')

    V = norm.cdf(df.spread.values / SPREAD_SD)
    ok = ~np.isnan(V)
    if ok.any():
        S = outcomes(df)
        print(f'  {"closing spread":<16} Acc {acc(V[ok], S[ok]):.4f}  '
              f'L {ll(V[ok], S[ok]):.4f}'
              + ('' if ok.all() else f'  [{ok.sum()} of {n} with a line]'))

    # head to head, on games where both models have a forecast
    both = df[[c for c, _ in MODELS if c in df.columns]].notna().all(axis=1)
    if both.sum() and all(c in df.columns for c, _ in MODELS):
        sub = df[both]
        S = outcomes(sub)
        d = ll(sub.E.values, S) - ll(sub.E_full.values, S)
        print(f'  head to head on {int(both.sum())}: '
              f'{d:+.4f} in L to theta* (positive = theta* ahead)')


if __name__ == '__main__':
    df = attach_results(load_predictions())
    played = df[df.result.notna()].copy()

    print(f'{len(played)} of {len(df)} predicted games played')

    if played.empty:
        print('\nNothing to score yet — check back after Sunday.')
        raise SystemExit

    S = outcomes(played)
    played['hit'] = np.where((played.E - 0.5) * (S - 0.5) > 0, 'Y', 'n')
    if 'E_full' in played.columns:
        played['hit2'] = np.where(
            played.E_full.isna(), '-',
            np.where((played.E_full - 0.5) * (S - 0.5) > 0, 'Y', 'n'))

    cols = ['week', 'away', 'home', 'E']
    if 'E_full' in played.columns:
        cols += ['E_full']
    cols += ['spread', 'result', 'hit']
    if 'hit2' in played.columns:
        cols += ['hit2']
    print()
    print(played[cols].to_string(index=False))

    score_block(played, 'all games so far')

    if played.week.nunique() > 1:
        for wk in sorted(played.week.unique()):
            score_block(played[played.week == wk], f'week {wk}')

    # Games where the model disagrees most with the line. An edge, if one
    # exists anywhere, should be largest here -- and these are genuinely
    # out of sample, unlike the backtested version in ats.py.
    has_line = played.spread.notna()
    sub = played[has_line]
    if len(sub):
        implied = SPREAD_SD * norm.ppf(np.clip(sub.E.values, 1e-6, 1 - 1e-6))
        edge = implied - sub.spread.values
        big = np.abs(edge) >= 3

        if big.sum():
            b = sub[big]
            picked_home = edge[big] > 0
            covered = b.result.values > b.spread.values
            ats = (picked_home == covered).mean()
            print(f'\nlarge disagreements (|edge| >= 3 pts): {big.sum()} '
                  f'games, {ats:.3f} ATS')
            print('  Break-even at -110 is 0.5238. Far too few games to mean '
                  'anything yet.')
