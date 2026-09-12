"""
Six windows on the widened flag -- teams that cannot improve their seed.

clinch_windows.py ran this protocol on 'any team that has clinched a
berth' and got 4 of 6. clinch_wide.py showed why that was the wrong
population: the shortfall runs +0.0146 (t +0.6) for clinched teams whose
seed is still live, +0.0748 for teams that cannot improve, and +0.2426
for teams whose rank cannot change at all. The broad flag was the real
effect diluted by 341 team-games of nothing.

This is the same test on the corrected flag. Same protocol, same
control, same windows -- tune the scale on everything strictly before
the window, score on the window.
"""
import numpy as np

from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from clinch_wide import build_nogain_map
from clinch_windows import WINDOWS, K_STAR, H_STAR, RHO_STAR

SCALE_GRID = [0, 25, 50, 80, 115, 155, 200, 260]


def tune(games, qb_map, epa_map, cmap, max_season):
    window = season_mask(games, None, max_season)
    res = []
    for s in SCALE_GRID:
        E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                          qb_map=qb_map, qb_scale=QB_SCALE,
                          qb_alpha=QB_ALPHA, epa_map=epa_map,
                          epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA,
                          clinch_map=cmap, clinch_scale=s)
        res.append((L(E[window], S[window]), s))
    res.sort()
    return res[0][1]


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()
    late = ((games.game_type == 'REG') & (games.week >= 15)).values

    E0, S0, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                        epa_map=epa_map, epa_scale=EPA_SCALE,
                        epa_alpha=EPA_ALPHA)

    for graded in (False, True):
        cmap = build_nogain_map(games, graded=graded)
        name = 'GRADED (locked 1.0 / nogain 0.5)' if graded else 'BINARY (nogain+locked = 1.0)'
        print(f'\n{"="*74}\n{name} — {len(cmap)} flagged team-games')
        print(f'{"window":>10} {"n":>5} {"theta*":>8} {"+clinch":>8} '
              f'{"diff":>8} {"tuned":>6}   {"late-only":>18}')
        diffs, scales = [], []
        for label, lo, hi in WINDOWS:
            test = season_mask(games, min_season=lo, max_season=hi)
            s = tune(games, qb_map, epa_map, cmap, lo - 1)
            Ec, Sc, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR,
                                mov=True, qb_map=qb_map, qb_scale=QB_SCALE,
                                qb_alpha=QB_ALPHA, epa_map=epa_map,
                                epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA,
                                clinch_map=cmap, clinch_scale=s)
            L0, Lc = L(E0[test], S0[test]), L(Ec[test], Sc[test])
            tl = test & late
            d = L0 - Lc
            diffs.append(d); scales.append(s)
            print(f'{label:>10} {test.sum():>5} {L0:>8.4f} {Lc:>8.4f} '
                  f'{d:>+8.4f} {s:>6}   '
                  f'{L(E0[tl],S0[tl])-L(Ec[tl],Sc[tl]):>+8.4f} on {tl.sum():>4}')
        print(f'  {sum(1 for d in diffs if d>0)} of {len(diffs)} positive; '
              f'scales {sorted(set(scales))}')
