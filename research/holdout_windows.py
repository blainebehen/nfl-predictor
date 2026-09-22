"""
Does rolling H beat fixed H in general, or only on 2019-2021?

The +0.0033 held-out gain for rolling H was measured on a test window
(2019-2025) that opens with three anomalous seasons: home-win rates of
.521/.498/.516 against a .573 baseline for 1999-2018. An adaptive H can
track that; a constant H fit on earlier data cannot.

But the raw series shows no trend before 2019 -- just scatter around .573
with a per-season SE of ~.030. So there may be nothing for a rolling H to
track most of the time, and the gain may be an artifact of where the test
window happens to start.

This runs the same comparison over several test windows. For each, both
schemes are tuned on everything BEFORE the window and scored on it. If
rolling only wins on 2019+, it is an anomaly-handler, not an improvement.
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

    print('\ndiff > 0 means rolling H won that window.')
    print('If only the 2019-2021 rows are positive, the gain is an')
    print('anomaly-handler rather than a general improvement.')
