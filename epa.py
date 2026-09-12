"""
Tuning and evaluation for the team-EPA adjustment.

epa_test.py established there is signal: bucketing the model residual by a
rolling net-EPA differential gives a monotone gradient with both tails past
3 SE. This file grids the parameters and applies the same two tests the QB
adjustment had to pass -- an in-sample control at scale 0, and a held-out
window -- with the six-window replication in epa_windows.py.

The QB adjustment is held on throughout, since it is already adopted. What
is being measured is what EPA adds on top of it, not what EPA does alone.
"""
from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, TEST_START,
                 QB_SCALE, QB_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50

SCALE_GRID = [0, 50, 100, 200, 350, 500, 750]
ALPHA_GRID = [0.05, 0.10, 0.15, 0.25, 0.40]


def tune_epa(games, qb_map, epa_map, min_season=None, max_season=None):
    """
    Grid over (epa_scale, epa_alpha) with team and QB parameters fixed.

    epa_scale = 0 is the control: same code, same games, adjustment off.
    """
    window = season_mask(games, min_season, max_season)
    results = []
    for scale in SCALE_GRID:
        for alpha in ALPHA_GRID:
            E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                              mov=True, qb_map=qb_map, qb_scale=QB_SCALE,
                              qb_alpha=QB_ALPHA, epa_map=epa_map,
                              epa_scale=scale, epa_alpha=alpha)
            Ew, Sw = E[window], S[window]
            results.append((L(Ew, Sw), Accuracy(Ew, Sw), scale, alpha))
            if scale == 0:
                break          # alpha is inert when scale is 0
    results.sort()
    return results


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

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
    train = tune_epa(games, qb_map, epa_map, max_season=TEST_START - 1)
    _, _, s_t, a_t = train[0]

    Ee, Se, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                        epa_map=epa_map, epa_scale=s_t, epa_alpha=a_t)
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

    print(f'\nheld out on {TEST_START}+ ({test.sum()} games), '
          f'tuned on earlier seasons only')
    print(f'  QB only     L {L(E0[test], S0[test]):.4f}  '
          f'Acc {Accuracy(E0[test], S0[test]):.4f}')
    print(f'  QB + EPA    L {L(Ee[test], Se[test]):.4f}  '
          f'Acc {Accuracy(Ee[test], Se[test]):.4f}  '
          f'(scale={s_t}, alpha={a_t})')
    print(f'  diff        {L(E0[test], S0[test]) - L(Ee[test], Se[test]):+.4f}')
