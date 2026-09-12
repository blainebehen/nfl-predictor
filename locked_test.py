"""
The locked flag on its own: rank frozen in both directions.

56 team-games across 27 seasons, about two a season. Both mechanisms are
tested on exactly that population:

  ADJUST  dock rating points from the locked team before the game. Fixes
          the prediction; the game still trains the model.
  EXCLUDE predict and score the game as normal, but do not update any
          rating from it. Protects the ratings from learning that a team
          resting its starters is worse than it is.

The targeted version of the late-season exclusion in RESULTS.md, which
dropped all of REG weeks 17-18 -- 1,016 team-games -- and came out
slightly worse. This drops 56.

Same six windows, same control, same protocol as QB and EPA.
"""
import numpy as np

from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from clinch_wide import build_wide_map
from clinch_windows import WINDOWS, K_STAR, H_STAR, RHO_STAR

SCALE_GRID = [0, 40, 80, 130, 190, 260, 340]

if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()
    lab = build_wide_map(games)

    locked = {k: 1.0 for k, v in lab.items() if v == 'locked'}
    skip = {gid for (gid, t) in locked}
    print(f'locked team-games {len(locked)}, in {len(skip)} games\n')

    base = dict(k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True, qb_map=qb_map,
                qb_scale=QB_SCALE, qb_alpha=QB_ALPHA, epa_map=epa_map,
                epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA)

    E0, S0, _ = run_elo(games, **base)
    late = ((games.game_type == 'REG') & (games.week >= 15)).values

    # ---------------- EXCLUDE ----------------
    Ex, Sx, _ = run_elo(games, skip_games=skip, **base)
    print('EXCLUDE — no tuning, nothing to tune')
    print(f'{"window":>10} {"n":>5} {"theta*":>8} {"-locked":>8} {"diff":>8}')
    npos = 0
    for label, lo, hi in WINDOWS:
        t = season_mask(games, min_season=lo, max_season=hi)
        d = L(E0[t], S0[t]) - L(Ex[t], Sx[t])
        npos += d > 0
        print(f'{label:>10} {t.sum():>5} {L(E0[t],S0[t]):>8.4f} '
              f'{L(Ex[t],Sx[t]):>8.4f} {d:>+8.4f}')
    print(f'  {npos} of {len(WINDOWS)} positive')
    print(f'  whole sample: {L(E0,S0):.4f} -> {L(Ex,Sx):.4f} '
          f'({L(E0,S0)-L(Ex,Sx):+.4f})   '
          f'Acc {Accuracy(E0,S0):.4f} -> {Accuracy(Ex,Sx):.4f}')

    # ---------------- ADJUST ----------------
    print('\nADJUST — scale tuned inside each training span')
    print(f'{"window":>10} {"n":>5} {"theta*":>8} {"+locked":>8} {"diff":>8} '
          f'{"tuned":>6}   {"late-only":>18}')
    npos, scales = 0, []
    for label, lo, hi in WINDOWS:
        tr = season_mask(games, None, lo - 1)
        res = []
        for s in SCALE_GRID:
            E, S, _ = run_elo(games, clinch_map=locked, clinch_scale=s, **base)
            res.append((L(E[tr], S[tr]), s))
        res.sort()
        s = res[0][1]
        Ec, Sc, _ = run_elo(games, clinch_map=locked, clinch_scale=s, **base)
        t = season_mask(games, min_season=lo, max_season=hi)
        tl = t & late
        d = L(E0[t], S0[t]) - L(Ec[t], Sc[t])
        npos += d > 0; scales.append(s)
        print(f'{label:>10} {t.sum():>5} {L(E0[t],S0[t]):>8.4f} '
              f'{L(Ec[t],Sc[t]):>8.4f} {d:>+8.4f} {s:>6}   '
              f'{L(E0[tl],S0[tl])-L(Ec[tl],Sc[tl]):>+8.4f} on {tl.sum():>4}')
    print(f'  {npos} of {len(WINDOWS)} positive; scales {sorted(set(scales))}')

    # ---------------- BOTH ----------------
    Eb, Sb, _ = run_elo(games, clinch_map=locked, clinch_scale=130,
                        skip_games=skip, **base)
    print(f'\nBOTH (adjust at 130 + exclude), whole sample: '
          f'{L(E0,S0):.4f} -> {L(Eb,Sb):.4f} ({L(E0,S0)-L(Eb,Sb):+.4f})')
