"""
Tuning and evaluation for the quarterback adjustment.

Data extraction lives in qb_data.py. This file grids over the QB
parameters and measures whether the adjustment earns its place.

Design note: the original formulation compared each starter to a team
baseline T -- the idea being that team Elo already absorbs a franchise's
average QB, so only the deviation from it is new information. The data
rejected that. Tuning drove qb_beta (T's update rate) to zero at every
grid floor it was given, and at beta = 0 the baseline is a constant that
cancels out of the difference:

    (Q_home - T) - (Q_away - T) = Q_home - Q_away

Dropping T improved both in-sample and held-out loss and removed a
parameter. The reason the double-counting worry was unfounded: once the
adjustment enters d, team ratings update against a prediction that already
includes it, so they settle lower for teams with good quarterbacks. Elo
partitions the credit on its own without an explicit baseline.
"""
from data import load_games
from elo import run_elo, season_mask, Accuracy, L, TEST_START
from qb_data import build_qb_map

# theta* for the team model, held fixed while the QB parameters are tuned.
# A staged search, not a joint one: tuning all five at once is ~20x the
# runtime over a surface already known to be flat in k and rho.
K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50

SCALE_GRID = [0, 100, 200, 400, 600, 900, 1200]
ALPHA_GRID = [0.003, 0.005, 0.01, 0.02, 0.05, 0.10]
BETA_GRID = [0.0]        # settled: the team baseline earns nothing


def tune_qb(games, qb_map, min_season=None, max_season=None):
    """
    Grid over (qb_scale, qb_alpha) with the team parameters fixed.

    qb_scale = 0 is the control: it disables the adjustment entirely, so
    its row is the no-QB baseline measured on exactly the same games.
    """
    window = season_mask(games, min_season, max_season)
    results = []
    for scale in SCALE_GRID:
        for alpha in ALPHA_GRID:
            for beta in BETA_GRID:
                E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                                  mov=True, qb_map=qb_map, qb_scale=scale,
                                  qb_alpha=alpha, qb_beta=beta)
                Ew, Sw = E[window], S[window]
                results.append((L(Ew, Sw), Accuracy(Ew, Sw), scale, alpha, beta))
                if scale == 0:
                    break          # alpha/beta are inert when scale is 0
            if scale == 0:
                break
    results.sort()
    return results


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games)

    res = tune_qb(games, qb_map)
    control = [r for r in res if r[2] == 0][0]

    print(f'\n{"L":>7} {"Acc":>7} {"scale":>6} {"alpha":>6}')
    for loss, acc, scale, alpha, beta in res[:8]:
        print(f'{loss:7.4f} {acc:7.4f} {scale:6d} {alpha:6.3f}')
    print(f'{control[0]:7.4f} {control[1]:7.4f} {0:6d} {"—":>6}'
          f'   <- control, no QB adjustment')
    print(f'\nin-sample improvement: {control[0] - res[0][0]:+.4f}')

    test = season_mask(games, min_season=TEST_START)
    train_res = tune_qb(games, qb_map, max_season=TEST_START - 1)
    _, _, s_t, a_t, b_t = train_res[0]

    Eq, Sq, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=s_t, qb_alpha=a_t, qb_beta=b_t)
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True)

    print(f'\nheld out on {TEST_START}+ ({test.sum()} games), '
          f'tuned on earlier seasons only')
    print(f'  no QB    L {L(E0[test], S0[test]):.4f}  '
          f'Acc {Accuracy(E0[test], S0[test]):.4f}')
    print(f'  with QB  L {L(Eq[test], Sq[test]):.4f}  '
          f'Acc {Accuracy(Eq[test], Sq[test]):.4f}  '
          f'(scale={s_t}, alpha={a_t})')
    print(f'  diff     {L(E0[test], S0[test]) - L(Eq[test], Sq[test]):+.4f}')
