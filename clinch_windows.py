"""
Does the clinch adjustment hold up across test windows?

The test that adopted the QB adjustment and team EPA and rejected rolling
H. For each window: tune the clinch scale on everything strictly before
it, then score both the clinch model and the theta* control on the window
itself. Team, QB and EPA parameters stay fixed throughout.

This feature needs the test more than the others did. It was not found by
the residual scan -- the scan's pooled version came back flat. It appeared
only after splitting a bucket, which is the forking-paths risk the project
warns about everywhere else. Six independent windows are the check on
that: a split that found noise will not replicate.

Read it the way RESULTS.md reads the others. Every window positive, with
parameters clustering, is adoption. One big window and five flat ones is
the rolling-H pattern and means rejection, however good the mechanism
sounds.

One structural difference from QB and EPA: this feature touches only REG
weeks 15+, about 11% of games. Even a real effect can only move whole-
sample L by a small amount, so the per-window numbers here will be an
order of magnitude smaller than the QB adjustment's. The right comparison
is against zero and against each other, not against the QB table.
"""
import numpy as np

from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from clinch import build_clinch_map

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50

SCALE_GRID = [0, 15, 30, 50, 75, 110, 160]

WINDOWS = [
    ('2004-2009', 2004, 2009),
    ('2010-2015', 2010, 2015),
    ('2016-2018', 2016, 2018),
    ('2019-2021', 2019, 2021),
    ('2022-2025', 2022, 2025),
    ('2019-2025', 2019, 2025),
]


def tune_clinch(games, qb_map, epa_map, clinch_map,
                min_season=None, max_season=None, late_only=False):
    """
    Grid over clinch_scale with everything else at theta*.

    late_only scores on REG weeks 15+ alone. The feature is inert
    elsewhere, so whole-sample loss dilutes it by roughly 9x -- useful for
    reading the effect, but the adoption decision is made on whole-sample
    loss like every other feature in this project.
    """
    window = season_mask(games, min_season, max_season)
    if late_only:
        window = window & ((games.game_type == 'REG')
                           & (games.week >= 15)).values

    results = []
    for scale in SCALE_GRID:
        E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                          qb_map=qb_map, qb_scale=QB_SCALE,
                          qb_alpha=QB_ALPHA, epa_map=epa_map,
                          epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA,
                          clinch_map=clinch_map, clinch_scale=scale)
        Ew, Sw = E[window], S[window]
        results.append((L(Ew, Sw), Accuracy(Ew, Sw), scale))
    results.sort()
    return results


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()
    clinch_map = build_clinch_map(games)
    print(f'clinch flags: {len(clinch_map)} team-games\n')

    # control: theta*, clinch off
    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                        epa_map=epa_map, epa_scale=EPA_SCALE,
                        epa_alpha=EPA_ALPHA)

    late = ((games.game_type == 'REG') & (games.week >= 15)).values

    print(f'{"window":>10} {"n":>5} {"theta*":>8} {"+clinch":>8} {"diff":>8} '
          f'{"tuned":>8}   {"late-only diff":>14}')

    diffs = []
    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)

        res = tune_clinch(games, qb_map, epa_map, clinch_map,
                          max_season=lo - 1)
        _, _, scale = res[0]

        Ec, Sc, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                            mov=True, qb_map=qb_map, qb_scale=QB_SCALE,
                            qb_alpha=QB_ALPHA, epa_map=epa_map,
                            epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA,
                            clinch_map=clinch_map, clinch_scale=scale)

        L0, Lc = L(E0[test], S0[test]), L(Ec[test], Sc[test])

        tl = test & late
        L0l, Lcl = L(E0[tl], S0[tl]), L(Ec[tl], Sc[tl])

        diffs.append((label, L0 - Lc, scale))
        print(f'{label:>10} {test.sum():>5} {L0:>8.4f} {Lc:>8.4f} '
              f'{L0 - Lc:>+8.4f} {scale:>8}   '
              f'{L0l - Lcl:>+8.4f} on {tl.sum():>4}')

    print('\ndiff > 0 means clinch won that window, on top of theta*.')
    pos = sum(1 for _, d, _ in diffs if d > 0)
    print(f'{pos} of {len(diffs)} windows positive; '
          f'scales selected: {sorted({s for _, _, s in diffs})}')
