"""
Does the EPA adjustment hold up across test windows, or only on 2019+?

The test that rejected rolling H and confirmed the QB adjustment. A single
held-out window can favour a feature by accident; six cannot.

For each window: tune the EPA parameters on everything strictly before it,
then score both the EPA model and the QB-only control on the window itself.
Team and QB parameters stay fixed throughout.
"""
import numpy as np
from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from epa import tune_epa, K_STAR, H_STAR, RHO_STAR

WINDOWS = [
    ('2004-2009', 2004, 2009),
    ('2010-2015', 2010, 2015),
    ('2016-2018', 2016, 2018),
    ('2019-2021', 2019, 2021),
    ('2022-2025', 2022, 2025),
    ('2019-2025', 2019, 2025),
]

if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

    print(f'{"window":>10} {"n":>5} {"QB only":>8} {"QB+EPA":>8} {"diff":>8} '
          f'{"tuned":>18}')

    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)

        res = tune_epa(games, qb_map, epa_map, max_season=lo - 1)
        _, _, scale, alpha = res[0]

        Ee, Se, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                            qb_map=qb_map, qb_scale=QB_SCALE,
                            qb_alpha=QB_ALPHA, epa_map=epa_map,
                            epa_scale=scale, epa_alpha=alpha)

        L0 = L(E0[test], S0[test])
        Le = L(Ee[test], Se[test])

        print(f'{label:>10} {test.sum():>5} {L0:>8.4f} {Le:>8.4f} '
              f'{L0 - Le:>+8.4f} {f"s={scale} a={alpha}":>18}')

    print('\ndiff > 0 means EPA won that window, on top of the QB model.')
    print('Consistent positives make it a real improvement; concentration')
    print('in one window is the rolling-H pattern and means rejection.')
