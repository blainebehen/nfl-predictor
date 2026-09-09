"""
Does the QB adjustment hold up across test windows, or only on 2019+?

RESULT: every window positive, and the smallest gain exceeds rolling H's
largest. Five of six windows independently select alpha in {0.01, 0.02}
and scale in {600, 900}.

    2004-2009   1602   +0.0037
    2010-2015   1602   +0.0082
    2016-2018    801   +0.0064
    2019-2021    821   +0.0121
    2022-2025   1139   +0.0059
    2019-2025   1960   +0.0071

The gain is largest on 2019-2021, the window hardest for both models --
consistent with COVID protocols pulling starters out on short notice,
which is exactly what this feature catches and team Elo cannot.

The same check that killed the rolling-H feature. A single held-out window
can favour a feature by accident -- rolling H won on 2019-2025 and then
turned out to win nowhere else. It would be inconsistent to apply that test
to one feature and not another.

For each window: tune the QB parameters on everything strictly before it,
then score both the QB model and the no-QB control on the window itself.
Team parameters stay at theta*.
"""
import numpy as np
from data import load_games
from elo import run_elo, season_mask, Accuracy, L
from qb import tune_qb, K_STAR, H_STAR, RHO_STAR
from qb_data import build_qb_map

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

    # the control is the same for every window
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True)

    print(f'{"window":>10} {"n":>5} {"no QB":>8} {"with QB":>8} {"diff":>8} '
          f'{"tuned theta":>22}')

    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)

        res = tune_qb(games, qb_map, max_season=lo - 1)
        _, _, scale, alpha, beta = res[0]

        Eq, Sq, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                            qb_map=qb_map, qb_scale=scale,
                            qb_alpha=alpha, qb_beta=beta)

        L0 = L(E0[test], S0[test])
        Lq = L(Eq[test], Sq[test])

        print(f'{label:>10} {test.sum():>5} {L0:>8.4f} {Lq:>8.4f} '
              f'{L0 - Lq:>+8.4f} '
              f'{f"s={scale} a={alpha} b={beta}":>22}')

    print('\ndiff > 0 means the QB adjustment won that window.')
    print('Consistent positives across windows would make this a real')
    print('improvement rather than an artifact of one test period.')
