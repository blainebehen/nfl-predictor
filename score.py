#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:07:33 2026

@author: blainebehen
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


def score(df):
    """Accuracy and log loss for the model and the spread, on played games."""
    played = df[df.result.notna()].copy()
    if played.empty:
        return None, 0

    S = np.where(played.result > 0, 1.0,
                 np.where(played.result == 0, 0.5, 0.0))
    E = played.E.values
    V = norm.cdf(played.spread.values / SPREAD_SD)

    def acc(p):
        return ((p - 0.5) * (S - 0.5) > 0).mean()

    def ll(p):
        return -(S * np.log(p) + (1 - S) * np.log(1 - p)).mean()

    return pd.DataFrame({
        'Acc': [acc(E), acc(V)],
        'L': [ll(E), ll(V)],
    }, index=['model', 'spread']).round(4), len(played)


if __name__ == '__main__':
    df = attach_results(load_predictions())

    played = df[df.result.notna()]
    print(f'{len(played)} of {len(df)} games played\n')

    if len(played):
        show = played[['week', 'away', 'home', 'E', 'spread', 'result']].copy()
        show['hit'] = np.where(
            (show.E - 0.5) * (np.sign(show.result) / 2) > 0, 'Y', 'n')
        print(show.to_string(index=False))

        table, n = score(df)
        print(f'\nover {n} games:')
        print(table)
    else:
        print('nothing to score yet — check back after Sunday')