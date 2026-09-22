"""
The model: Elo ratings for NFL teams, plus two adjustments.

How a game gets predicted:
  1. Every team has a rating (everyone starts at 1500).
  2. Take the home team's rating minus the away team's, and add a bonus
     for playing at home. Call that the edge, d.
  3. Add the two adjustments to d:
       - QB: is this team's starting QB better or worse than who they
         normally play?
       - EPA: how efficiently has each team been moving the ball, and
         stopping it, lately?
  4. Turn d into a win probability with the Elo formula.

After the game, both teams' ratings move toward what happened. A big upset
moves them a lot; an expected result barely moves them. The game is
always predicted BEFORE its result is used, so every prediction is a real
out-of-sample forecast.

Run this file to tune the model, test it on seasons it wasn't tuned on,
and compare it to Vegas. It takes a few minutes.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import load_games

# Hyperparameter values tried when tuning
K_GRID = [4, 6, 8, 10, 14, 20, 24, 28, 32, 36, 44, 48, 56, 64, 80]
H_GRID = [20, 30, 40, 50, 60, 70, 80]
RHO_GRID = [0.0, 0.15, 0.33, 0.5, 0.7]

TEST_START = 2019   # hold out 2019 onward to test on

# QB adjustment (tested in research/qb.py)
QB_SCALE = 600      # rating points per unit of EPA per dropback
QB_ALPHA = 0.02     # how fast a QB's rating updates
QB_BETA = 0.03      # how fast a team's "normal QB" level updates

# Team EPA adjustment (tested in research/epa.py)
EPA_SCALE = 200     # rating points per unit of net EPA per play
EPA_ALPHA = 0.15    # how fast a team's EPA rating updates


def run_elo(games, k=20, H=50, rho=0.5, mov=True,
            qb_map=None, qb_scale=0.0, qb_alpha=QB_ALPHA,
            epa_map=None, epa_scale=0.0, epa_alpha=EPA_ALPHA):
    """
    Go through every game in order: predict it, record it, then update.

    k      how much ratings move after each game
    H      home-field advantage, in rating points
    rho    how much ratings shrink back toward 1500 each offseason
    mov    if True, bigger wins move ratings more
    qb_map / epa_map   the features (from features.py). Leave them out,
           or set the scale to 0, to turn that adjustment off.

    Returns E (predicted chance the home team wins), S (what happened:
    1 home win, 0.5 tie, 0 away win) and the final ratings.
    """
    R = defaultdict(lambda: 1500.0)     # team ratings
    E_all, S_all = [], []
    season = None

    use_qb = qb_map is not None and qb_scale != 0.0
    Q = defaultdict(float)      # each QB's rating (EPA per dropback)
    T = defaultdict(float)      # each team's normal QB level

    use_epa = epa_map is not None and epa_scale != 0.0
    OFF, DEF = {}, {}           # each team's offensive / defensive EPA

    for g in games.itertuples():

        # New season: rosters change, so pull every rating partway
        # back toward 1500.
        if g.season != season:
            for t in R:
                R[t] = 1500 + (1 - rho) * (R[t] - 1500)
            season = g.season

        # --- predict ---------------------------------------------------
        adj = 0.0

        # QB: only the difference from the team's usual QB matters,
        # because the team rating already includes their usual QB.
        if use_qb:
            h_qb = qb_map.get((g.game_id, g.home_team))
            a_qb = qb_map.get((g.game_id, g.away_team))
            dev_h = Q[h_qb[0]] - T[g.home_team] if h_qb else 0.0
            dev_a = Q[a_qb[0]] - T[g.away_team] if a_qb else 0.0
            adj = qb_scale * (dev_h - dev_a)

        # EPA: net = offense EPA minus EPA allowed on defense
        if use_epa:
            net_h = OFF.get(g.home_team, 0.0) - DEF.get(g.home_team, 0.0)
            net_a = OFF.get(g.away_team, 0.0) - DEF.get(g.away_team, 0.0)
            adj += epa_scale * (net_h - net_a)

        d = R[g.home_team] + H - R[g.away_team] + adj
        E = 1 / (1 + 10 ** (-d / 400))      # the Elo formula

        # --- record ----------------------------------------------------
        S = 1.0 if g.result > 0 else 0.5 if g.result == 0 else 0.0
        E_all.append(E)
        S_all.append(S)

        # --- update ----------------------------------------------------
        # Margin of victory: log of the margin, so winning by 10 vs 3
        # matters more than 42 vs 35. Blowouts by a big favorite count
        # less, upsets count more.
        if mov and g.result != 0:
            d_win = d if g.result > 0 else -d
            M = np.log(abs(g.result) + 1) * 2.2 / (0.001 * d_win + 2.2)
        else:
            M = 1.0

        # Move both ratings by the surprise (S - E). What one team gains,
        # the other loses.
        R[g.home_team] += k * M * (S - E)
        R[g.away_team] -= k * M * (S - E)

        # Move each QB's rating (and his team's normal level) a little
        # toward what he did this game.
        if use_qb:
            for qb, team in ((h_qb, g.home_team), (a_qb, g.away_team)):
                if qb is None:
                    continue
                pid, val = qb
                Q[pid] += qb_alpha * (val - Q[pid])
                T[team] += QB_BETA * (val - T[team])

        # Same idea for each team's offensive and defensive EPA
        if use_epa:
            for team in (g.home_team, g.away_team):
                entry = epa_map.get((g.game_id, team))
                if entry is None:
                    continue
                o, d_ = entry
                OFF[team] = OFF.get(team, o) + epa_alpha * (o - OFF.get(team, o))
                DEF[team] = DEF.get(team, d_) + epa_alpha * (d_ - DEF.get(team, d_))

    return np.array(E_all), np.array(S_all), dict(R)


def accuracy(E, S):
    """Share of games where the favorite won."""
    return ((E - 0.5) * (S - 0.5) > 0).mean()


def log_loss(E, S):
    """
    Average of -log(probability given to what actually happened).
    Lower is better. Always guessing 50% scores 0.693.
    """
    return -(S * np.log(E) + (1 - S) * np.log(1 - E)).mean()


def season_mask(games, min_season=None, max_season=None):
    """True for games inside the season range."""
    m = np.ones(len(games), dtype=bool)
    if min_season is not None:
        m &= (games.season >= min_season).values
    if max_season is not None:
        m &= (games.season <= max_season).values
    return m


def tune(games, min_season=None, max_season=None):
    """Try every (k, H, rho) and sort by log loss, best first."""
    mask = season_mask(games, min_season, max_season)
    results = []
    for k in K_GRID:
        for H in H_GRID:
            for rho in RHO_GRID:
                E, S, _ = run_elo(games, k=k, H=H, rho=rho, mov=True)
                results.append((log_loss(E[mask], S[mask]),
                                accuracy(E[mask], S[mask]), k, H, rho))
    results.sort()
    return results


if __name__ == '__main__':
    from features import build_qb_map, team_game_epa

    games = load_games()

    # --- tune on everything ---------------------------------------------
    best = tune(games)
    _, _, k, H, rho = best[0]
    print(f'best on all games: k={k}, H={H}, rho={rho}')

    # --- honest test: tune before 2019, score 2019 onward ----------------
    test = season_mask(games, min_season=TEST_START)
    _, _, k_t, H_t, rho_t = tune(games, max_season=TEST_START - 1)[0]
    E, S, _ = run_elo(games, k=k_t, H=H_t, rho=rho_t, mov=True)
    print(f'tuned before {TEST_START}: k={k_t}, H={H_t}, rho={rho_t}  ->  '
          f'{TEST_START}+ log loss {log_loss(E[test], S[test]):.4f}, '
          f'accuracy {accuracy(E[test], S[test]):.4f}')

    # --- the three versions of the model ---------------------------------
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()
    qb = dict(qb_map=qb_map, qb_scale=QB_SCALE)
    epa = dict(epa_map=epa_map, epa_scale=EPA_SCALE)

    E0, S0, _ = run_elo(games, k=k, H=H, rho=rho)
    Eq, Sq, _ = run_elo(games, k=k, H=H, rho=rho, **qb)
    E, S, R = run_elo(games, k=k, H=H, rho=rho, **qb, **epa)

    # --- Vegas -------------------------------------------------------------
    # The spread is Vegas's predicted margin. Real margins land around it
    # with a standard deviation of about 13.5 points, so the chance the
    # home team wins is P(margin > 0) = Phi(spread / 13.5).
    spread = games.spread_line.values
    has_line = ~np.isnan(spread)
    vegas = norm.cdf(spread / 13.5)

    print(f'\n{"model":<22} {"accuracy":>8} {"log loss":>9}')
    print(f'{"always pick home":<22} {(S > 0.5).mean():8.4f}')
    print(f'{"Elo + margin":<22} {accuracy(E0, S0):8.4f} {log_loss(E0, S0):9.4f}')
    print(f'{"  + QB":<22} {accuracy(Eq, Sq):8.4f} {log_loss(Eq, Sq):9.4f}')
    print(f'{"  + team EPA":<22} {accuracy(E, S):8.4f} {log_loss(E, S):9.4f}')
    print(f'{"Vegas":<22} {accuracy(vegas[has_line], S[has_line]):8.4f} '
          f'{log_loss(vegas[has_line], S[has_line]):9.4f}')

    # --- calibration ------------------------------------------------------
    # When the model says 70%, does the home team win about 70% of the time?
    df = pd.DataFrame({'E': E, 'S': S})
    df['bucket'] = pd.cut(df.E, bins=[0, .3, .4, .5, .6, .7, .8, 1.0])
    print()
    print(df.groupby('bucket', observed=True).agg(
        games=('S', 'size'),
        predicted=('E', 'mean'),
        actual=('S', 'mean')).round(3))

    print('\ncurrent ratings:')
    for team, rating in sorted(R.items(), key=lambda x: -x[1]):
        print(f'  {team} {rating:.0f}')
