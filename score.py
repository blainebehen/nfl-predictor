"""
Score my saved predictions against what actually happened, and against
the Vegas spread.

This is the real test. Everything in elo.py is a backtest on past games.
These predictions were saved and committed before kickoff.

Two models are tracked (see predict.py): E (Elo + QB) and E_full
(+ team EPA). E_full is blank for Week 1 because I saved Week 1 before
adding team EPA, and a prediction made after kickoff doesn't count.

Small samples: one week is 16 games, and accuracy over 16 games has a
standard error of about 12 percentage points. So a 10-6 week and a 6-10
week both fit a 64% model. It takes most of a season before the record
means much, and more than one season to tell two close models apart.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.stats import norm

SPREAD_SD = 13.5    # SD of (actual margin - closing spread), empirical

MODELS = [('E', 'Elo + QB'), ('E_full', '+ team EPA')]


def load_predictions(path='predictions_2026.csv'):
    return pd.read_csv(path)


def attach_results(preds):
    """
    Join actual outcomes onto predictions. Unplayed games get NaN.

    Also pulls spread_line fresh, as `close`. The `spread` saved in the
    CSV is the line that existed when the forecast was made, early in the
    week; by kickoff it has moved. Over 2026 weeks 1-3 the two differed
    on 31 of 48 games by an average of 0.87 points and up to 4.0, and
    scoring the market on the stale number understated it -- L 0.6704
    against 0.6500 on the real close.

    Both are kept because they answer different questions. The CLOSE is
    the benchmark: RESULTS.md measures the model against the closing
    line, so the live track record has to use the same yardstick. The
    STORED line is what was actually available to bet, so it is the right
    one for the ATS edge below.
    """
    sched = nfl.load_schedules().to_pandas()
    cols = ['season', 'week', 'away_team', 'home_team', 'result', 'spread_line']

    merged = preds.merge(
        sched[cols],
        left_on=['season', 'week', 'away', 'home'],
        right_on=['season', 'week', 'away_team', 'home_team'],
        how='left',
    )
    merged = merged.rename(columns={'spread_line': 'close'})
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

    V = norm.cdf(df.close.values / SPREAD_SD)
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
              f'{d:+.4f} log loss in favor of E_full (positive = E_full ahead)')


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

    # Games where my model disagrees with Vegas by 3+ points. If I have
    # any edge on the betting line, it should show up here.
    # the stored line, not the close: this is what could have been bet
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
