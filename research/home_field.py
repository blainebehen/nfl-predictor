"""
Should home-field advantage change over time?

Tested letting it move with the home-win rate of the last few seasons,
versus keeping it fixed. It looked better when tested on 2019-2025, but
that's because 2019-2021 had unusually few home wins. Checked across six
stretches of seasons, it only helped in that one. Kept fixed.

Slow: tunes both versions for every stretch (about 10 minutes).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from data import load_games
from elo_experiments import (run_elo, rolling_hfa, tune_fixed, tune_rolling,
                 season_mask, Accuracy, L)

# (label, first season of the test window, last season)
WINDOWS = [
    ('2004-2009', 2004, 2009),
    ('2010-2015', 2010, 2015),
    ('2016-2018', 2016, 2018),
    ('2019-2021', 2019, 2021),   # the anomalous stretch, isolated
    ('2022-2025', 2022, 2025),   # after it, isolated
    ('2019-2025', 2019, 2025),   # the original test window
]

if __name__ == '__main__':
    games = load_games()

    print(f'{"window":>10} {"n":>5} {"fixed L":>9} {"roll L":>9} '
          f'{"diff":>8} {"fixed th":>16} {"roll th":>16}')

    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)

        # tune both schemes on everything strictly before the window
        f = tune_fixed(games, max_season=lo - 1)
        r = tune_rolling(games, max_season=lo - 1)

        _, _, k_f, H_f, rho_f = f[0]
        Ef, Sf, _ = run_elo(games, k=k_f, H=H_f, rho=rho_f, mov=True)
        Lf = L(Ef[test], Sf[test])

        _, _, k_r, w_r, rho_r = r[0]
        Er, Sr, _ = run_elo(games, k=k_r, rho=rho_r, mov=True,
                            hfa=rolling_hfa(games, window=w_r))
        Lr = L(Er[test], Sr[test])

        print(f'{label:>10} {test.sum():>5} {Lf:>9.4f} {Lr:>9.4f} '
              f'{Lf - Lr:>+8.4f} '
              f'{f"k={k_f} H={H_f} p={rho_f}":>16} '
              f'{f"k={k_r} w={w_r} p={rho_r}":>16}')

    print('\ndiff > 0 means the changing home-field version won that stretch.')
