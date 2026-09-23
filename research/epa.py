"""
Does team EPA per play add anything on top of Elo + QB?

    python research/epa.py residuals   does EPA line up with what the model gets wrong?
    python research/epa.py tune        tune the EPA settings, then test on 2019+
    python research/epa.py windows     test on six different stretches of seasons
    python research/epa.py             all three

Idea: Elo only sees the final score. EPA per play measures how well a
team actually moved the ball (and stopped the other team), which should
be less noisy.

Result: with the old QB version, EPA helped a little in every stretch.
Once I switched to raw QB ratings, it basically stopped helping. The QB
ratings were already picking up most of it. See raw_qb_retune.py.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import load_games
from elo_experiments import (run_elo, season_mask, Accuracy, L, TEST_START,
                 QB_SCALE, QB_ALPHA)
from features import build_qb_map, team_game_epa

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50
ALPHA = 0.15          # how fast team EPA ratings update

SCALE_GRID = [0, 50, 100, 200, 350, 500, 750]
ALPHA_GRID = [0.05, 0.10, 0.15, 0.25, 0.40]

WINDOWS = [
    ('2004-2009', 2004, 2009),
    ('2010-2015', 2010, 2015),
    ('2016-2018', 2016, 2018),
    ('2019-2021', 2019, 2021),
    ('2022-2025', 2022, 2025),
    ('2019-2025', 2019, 2025),
]


def rolling_epa_diff(games, epa_map, alpha=ALPHA):
    """
    For each game: home team's net EPA rating minus the away team's,
    using only earlier games. Net = offense EPA minus defense EPA allowed.
    """
    off, deff, out = {}, {}, []
    for g in games.itertuples():
        def rating(team):
            return off.get(team, 0.0) - deff.get(team, 0.0)

        out.append(rating(g.home_team) - rating(g.away_team))

        for team in (g.home_team, g.away_team):
            entry = epa_map.get((g.game_id, team))
            if entry is None:
                continue
            o, d = entry
            off[team] = off.get(team, o) + alpha * (o - off.get(team, o))
            deff[team] = deff.get(team, d) + alpha * (d - deff.get(team, d))

    return np.array(out)


def tune_epa(games, qb_map, epa_map, min_season=None, max_season=None):
    """
    Try every EPA scale and alpha, with the QB adjustment on.
    scale = 0 is the QB-only baseline on the same games.
    """
    window = season_mask(games, min_season, max_season)
    results = []
    for scale in SCALE_GRID:
        for alpha in ALPHA_GRID:
            E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                              mov=True, qb_map=qb_map, qb_scale=QB_SCALE,
                              qb_alpha=QB_ALPHA, qb_beta=0.0, epa_map=epa_map,
                              epa_scale=scale, epa_alpha=alpha)
            Ew, Sw = E[window], S[window]
            results.append((L(Ew, Sw), Accuracy(Ew, Sw), scale, alpha))
            if scale == 0:
                break          # alpha is inert when scale is 0
    results.sort()
    return results


def residuals(games, qb_map, epa_map):
    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA, qb_beta=0.0)
    diff = rolling_epa_diff(games, epa_map)

    g = games.copy()
    g['resid'] = S - E
    g['epa_diff'] = diff
    g['home_win'] = (g.result > 0).astype(float)
    g = g[g.season >= 2000]      # drop the season where ratings are default

    print(f'EPA coverage: {len(epa_map)} team-games')
    print(f'epa_diff: mean {g.epa_diff.mean():+.4f}  sd {g.epa_diff.std():.4f}')
    print(f'correlation with model probability E: '
          f'{np.corrcoef(g.epa_diff, E[games.season >= 2000])[0, 1]:+.3f}')

    agg = g.groupby(pd.qcut(g.epa_diff, 8), observed=True).agg(
        n=('resid', 'size'), raw=('home_win', 'mean'),
        resid=('resid', 'mean'), sd=('resid', 'std'))
    agg['t'] = agg.resid / (agg.sd / np.sqrt(agg.n))

    print(f'\nmodel residual by rolling EPA differential (home - away)')
    print(f'{"bucket":>20} {"n":>6} {"raw":>7} {"resid":>8} {"t":>6}')
    for idx, r in agg.iterrows():
        print(f'{str(idx):>20} {int(r.n):>6} {r.raw:>7.3f} '
              f'{r.resid:>+8.4f} {r.t:>+6.1f}')

    print('\nIf EPA adds information, resid should go from negative to positive.')


def tune(games, qb_map, epa_map):
    res = tune_epa(games, qb_map, epa_map)
    control = [r for r in res if r[2] == 0][0]

    print(f'\n{"L":>7} {"Acc":>7} {"scale":>6} {"alpha":>6}')
    for loss, acc, scale, alpha in res[:8]:
        print(f'{loss:7.4f} {acc:7.4f} {scale:6d} {alpha:6.2f}')
    print(f'{control[0]:7.4f} {control[1]:7.4f} {0:6d} {"—":>6}'
          f'   <- control, QB but no EPA')
    print(f'\nin-sample improvement: {control[0] - res[0][0]:+.4f}')

    if res[0][2] in (SCALE_GRID[1], SCALE_GRID[-1]):
        print('  warning: scale at a grid edge — extend the range')
    if res[0][3] in (ALPHA_GRID[0], ALPHA_GRID[-1]):
        print('  warning: alpha at a grid edge — extend the range')

    test = season_mask(games, min_season=TEST_START)
    _, _, s_t, a_t = tune_epa(games, qb_map, epa_map,
                              max_season=TEST_START - 1)[0]

    Ee, Se, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA, qb_beta=0.0,
                        epa_map=epa_map, epa_scale=s_t, epa_alpha=a_t)
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA, qb_beta=0.0)

    print(f'\nheld out on {TEST_START}+ ({test.sum()} games), '
          f'tuned on earlier seasons only')
    print(f'  QB only     L {L(E0[test], S0[test]):.4f}  '
          f'Acc {Accuracy(E0[test], S0[test]):.4f}')
    print(f'  QB + EPA    L {L(Ee[test], Se[test]):.4f}  '
          f'Acc {Accuracy(Ee[test], Se[test]):.4f}  '
          f'(scale={s_t}, alpha={a_t})')
    print(f'  diff        {L(E0[test], S0[test]) - L(Ee[test], Se[test]):+.4f}')


def windows(games, qb_map, epa_map):
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA, qb_beta=0.0)

    print(f'\n{"window":>10} {"n":>5} {"QB only":>8} {"QB+EPA":>8} '
          f'{"diff":>8} {"tuned":>18}')
    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)
        _, _, scale, alpha = tune_epa(games, qb_map, epa_map,
                                      max_season=lo - 1)[0]
        Ee, Se, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                            mov=True, qb_map=qb_map, qb_scale=QB_SCALE,
                            qb_alpha=QB_ALPHA, qb_beta=0.0, epa_map=epa_map,
                            epa_scale=scale, epa_alpha=alpha)
        print(f'{label:>10} {test.sum():>5} {L(E0[test], S0[test]):>8.4f} '
              f'{L(Ee[test], Se[test]):>8.4f} '
              f'{L(E0[test], S0[test]) - L(Ee[test], Se[test]):>+8.4f} '
              f'{f"s={scale} a={alpha}":>18}')

    print('\ndiff > 0 means EPA helped in that stretch.')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    if what in ('residuals', 'all'):
        residuals(games, qb_map, epa_map)
    if what in ('tune', 'all'):
        tune(games, qb_map, epa_map)
    if what in ('windows', 'all'):
        windows(games, qb_map, epa_map)
