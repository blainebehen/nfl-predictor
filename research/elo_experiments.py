"""
The model from elo.py plus two options I tested and didn't keep:

    qb_beta   compare each starter to his team's usual QB (0 = off)
    hfa       home-field advantage that changes by season

The research scripts use this so they can switch those on and off.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from collections import defaultdict

import numpy as np

from elo import (K_GRID, H_GRID, RHO_GRID, TEST_START, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA, season_mask)
from elo import accuracy as Accuracy, log_loss as L

WINDOW_GRID = [3, 5, 8, 12]     # seasons of history behind a changing H


def rolling_hfa(games, window=5, prior=0.5631):
    """
    Home-field advantage for each season, from the home-win rate of the
    previous `window` seasons. If home teams win a share p of games,
    that's H = 400 * log10(p / (1 - p)) rating points.
    """
    by_season = games.assign(hw=games.result > 0).groupby('season').hw.mean()
    hfa = {}
    for season in by_season.index:
        past = by_season[by_season.index < season].tail(window)
        p = past.mean() if len(past) else prior
        hfa[season] = 400 * np.log10(p / (1 - p))
    return hfa


def run_elo(games, k=20, H=50, rho=0.5, mov=True, hfa=None,
            qb_map=None, qb_scale=0.0, qb_alpha=QB_ALPHA, qb_beta=0.0,
            epa_map=None, epa_scale=0.0, epa_alpha=EPA_ALPHA):
    """Same as elo.run_elo, plus qb_beta and hfa."""
    R = defaultdict(lambda: 1500.0)
    E_all, S_all = [], []
    season = None
    H_now = H

    use_qb = qb_map is not None and qb_scale != 0.0
    Q = defaultdict(float)      # each QB's rating
    T = defaultdict(float)      # each team's usual QB level (only if qb_beta > 0)

    use_epa = epa_map is not None and epa_scale != 0.0
    OFF, DEF = {}, {}

    for g in games.itertuples():
        if g.season != season:
            for t in R:
                R[t] = 1500 + (1 - rho) * (R[t] - 1500)
            season = g.season
            if hfa is not None:
                H_now = hfa[season]

        adj = 0.0
        if use_qb:
            h_qb = qb_map.get((g.game_id, g.home_team))
            a_qb = qb_map.get((g.game_id, g.away_team))
            dev_h = Q[h_qb[0]] - T[g.home_team] if h_qb else 0.0
            dev_a = Q[a_qb[0]] - T[g.away_team] if a_qb else 0.0
            adj = qb_scale * (dev_h - dev_a)

        if use_epa:
            net_h = OFF.get(g.home_team, 0.0) - DEF.get(g.home_team, 0.0)
            net_a = OFF.get(g.away_team, 0.0) - DEF.get(g.away_team, 0.0)
            adj += epa_scale * (net_h - net_a)

        d = R[g.home_team] + H_now - R[g.away_team] + adj
        E = 1 / (1 + 10 ** (-d / 400))
        S = 1.0 if g.result > 0 else 0.5 if g.result == 0 else 0.0
        E_all.append(E)
        S_all.append(S)

        if mov and g.result != 0:
            d_win = d if g.result > 0 else -d
            M = np.log(abs(g.result) + 1) * 2.2 / (0.001 * d_win + 2.2)
        else:
            M = 1.0
        R[g.home_team] += k * M * (S - E)
        R[g.away_team] -= k * M * (S - E)

        if use_qb:
            for qb, team in ((h_qb, g.home_team), (a_qb, g.away_team)):
                if qb is None:
                    continue
                pid, val = qb
                Q[pid] += qb_alpha * (val - Q[pid])
                T[team] += qb_beta * (val - T[team])

        if use_epa:
            for team in (g.home_team, g.away_team):
                entry = epa_map.get((g.game_id, team))
                if entry is None:
                    continue
                o, d_ = entry
                OFF[team] = OFF.get(team, o) + epa_alpha * (o - OFF.get(team, o))
                DEF[team] = DEF.get(team, d_) + epa_alpha * (d_ - DEF.get(team, d_))

    return np.array(E_all), np.array(S_all), dict(R)


def tune_fixed(games, min_season=None, max_season=None):
    """Try every (k, H, rho) with a fixed home-field advantage."""
    mask = season_mask(games, min_season, max_season)
    results = []
    for k in K_GRID:
        for H in H_GRID:
            for rho in RHO_GRID:
                E, S, _ = run_elo(games, k=k, H=H, rho=rho)
                results.append((L(E[mask], S[mask]), Accuracy(E[mask], S[mask]),
                                k, H, rho))
    results.sort()
    return results


def tune_rolling(games, min_season=None, max_season=None):
    """Try every (k, rho, window) with a changing home-field advantage."""
    mask = season_mask(games, min_season, max_season)
    hfas = {w: rolling_hfa(games, window=w) for w in WINDOW_GRID}
    results = []
    for k in K_GRID:
        for rho in RHO_GRID:
            for w in WINDOW_GRID:
                E, S, _ = run_elo(games, k=k, rho=rho, hfa=hfas[w])
                results.append((L(E[mask], S[mask]), Accuracy(E[mask], S[mask]),
                                k, w, rho))
    results.sort()
    return results
