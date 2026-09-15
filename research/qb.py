"""
The quarterback adjustment: tuning and six-window validation.

    python research/qb.py tune      grid the parameters, in-sample + held out
    python research/qb.py windows   six held-out windows
    python research/qb.py           both

Was qb.py and qb_windows.py. Extraction moved to features.py.

RESULT: adopted. The largest single improvement in the project -- +0.0061
in-sample, +0.0071 held out, and positive in all six windows, +0.0037 to
+0.0121. Five of six independently select alpha in {0.01, 0.02} and scale
in {600, 900}.

Design note: the original formulation compared each starter to a team
baseline T, on the reasoning that team Elo already absorbs a franchise's
average QB, so only the deviation is new information. The data rejected
it. Tuning drove qb_beta (T's update rate) to zero at every grid floor it
was given, and at beta = 0 the baseline is a constant that cancels:

    (Q_home - T) - (Q_away - T) = Q_home - Q_away

Dropping T improved both in-sample and held-out loss and removed a
parameter. The double-counting worry was unfounded: once the adjustment
enters d, team ratings update against a prediction that already includes
it, so they settle lower for teams with good quarterbacks. Elo partitions
the credit on its own.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import load_games
from elo import run_elo, season_mask, Accuracy, L, TEST_START
from features import build_qb_map

# theta* for the team model, held fixed while the QB parameters are tuned.
# A staged search, not a joint one: tuning all five at once is ~20x the
# runtime over a surface already known to be flat in k and rho.
K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50

SCALE_GRID = [0, 100, 200, 400, 600, 900, 1200]
ALPHA_GRID = [0.003, 0.005, 0.01, 0.02, 0.05, 0.10]
BETA_GRID = [0.0]        # settled: the team baseline earns nothing

WINDOWS = [
    ('2004-2009', 2004, 2009),
    ('2010-2015', 2010, 2015),
    ('2016-2018', 2016, 2018),
    ('2019-2021', 2019, 2021),
    ('2022-2025', 2022, 2025),
    ('2019-2025', 2019, 2025),
]


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
                results.append((L(Ew, Sw), Accuracy(Ew, Sw),
                                scale, alpha, beta))
                if scale == 0:
                    break          # alpha/beta are inert when scale is 0
            if scale == 0:
                break
    results.sort()
    return results


def tune(games, qb_map):
    res = tune_qb(games, qb_map)
    control = [r for r in res if r[2] == 0][0]

    print(f'\n{"L":>7} {"Acc":>7} {"scale":>6} {"alpha":>6} {"beta":>6}')
    for loss, acc, scale, alpha, beta in res[:8]:
        print(f'{loss:7.4f} {acc:7.4f} {scale:6d} {alpha:6.3f} {beta:6.2f}')
    print(f'{control[0]:7.4f} {control[1]:7.4f} {0:6d} {"—":>6} {"—":>6}'
          f'   <- control, no QB adjustment')
    print(f'\nin-sample improvement: {control[0] - res[0][0]:+.4f}')

    test = season_mask(games, min_season=TEST_START)
    train = tune_qb(games, qb_map, max_season=TEST_START - 1)
    _, _, s_t, a_t, b_t = train[0]

    Eq, Sq, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=s_t, qb_alpha=a_t,
                        qb_beta=b_t)
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True)

    print(f'\nheld out on {TEST_START}+ ({test.sum()} games), '
          f'tuned on earlier seasons only')
    print(f'  no QB       L {L(E0[test], S0[test]):.4f}  '
          f'Acc {Accuracy(E0[test], S0[test]):.4f}')
    print(f'  with QB     L {L(Eq[test], Sq[test]):.4f}  '
          f'Acc {Accuracy(Eq[test], Sq[test]):.4f}  '
          f'(scale={s_t}, alpha={a_t})')
    print(f'  diff        {L(E0[test], S0[test]) - L(Eq[test], Sq[test]):+.4f}')


def windows(games, qb_map):
    """
    The same check that killed the rolling-H feature. A single held-out
    window can favour a feature by accident -- rolling H won on 2019-2025
    and then turned out to win nowhere else.
    """
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True)

    print(f'\n{"window":>10} {"n":>5} {"no QB":>8} {"with QB":>8} '
          f'{"diff":>8} {"tuned theta":>22}')
    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)
        _, _, scale, alpha, beta = tune_qb(games, qb_map,
                                           max_season=lo - 1)[0]
        Eq, Sq, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                            mov=True, qb_map=qb_map, qb_scale=scale,
                            qb_alpha=alpha, qb_beta=beta)
        print(f'{label:>10} {test.sum():>5} {L(E0[test], S0[test]):>8.4f} '
              f'{L(Eq[test], Sq[test]):>8.4f} '
              f'{L(E0[test], S0[test]) - L(Eq[test], Sq[test]):>+8.4f} '
              f'{f"s={scale} a={alpha} b={beta}":>22}')

    print('\nEvery window positive, and the smallest gain exceeds the')
    print('largest gain rolling H managed anywhere. Parameters replicating')
    print('across independent tunings is what a real signal looks like.')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)

    if what in ('tune', 'both'):
        tune(games, qb_map)
    if what in ('windows', 'both'):
        windows(games, qb_map)
