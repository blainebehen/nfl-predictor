"""
The clinch adjustment at a fixed scale, and its whole-sample effect.

clinch_windows.py tunes the scale inside each window's training span and
gets 4 of 6 positive, with 2004-2009 selecting scale 0. That window trains
on 1999-2003 alone -- five seasons, a few hundred flagged team-games -- so
selecting 0 there may be a power problem rather than evidence against the
feature. This file separates the two questions:

  1. Held at one scale for every window, does the effect show up?
     (Not a validation -- the scale is informed by the whole sample. It
     asks whether the effect exists, not whether it can be tuned.)
  2. What does it do to the headline scoreboard?

Both matter for the writeup, and neither substitutes for the tuned test.
"""
import numpy as np

from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from clinch import build_clinch_map
from clinch_windows import WINDOWS, K_STAR, H_STAR, RHO_STAR, SCALE_GRID

FIXED = 50      # modal selection across the tuned windows


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()
    clinch_map = build_clinch_map(games)

    late = ((games.game_type == 'REG') & (games.week >= 15)).values

    base = dict(k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True, qb_map=qb_map,
                qb_scale=QB_SCALE, qb_alpha=QB_ALPHA, epa_map=epa_map,
                epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA)

    E0, S0, _ = run_elo(games, **base)
    Ec, Sc, Rc = run_elo(games, clinch_map=clinch_map,
                         clinch_scale=FIXED, **base)

    # --- 1. every window at the same fixed scale ------------------------
    print(f'clinch_scale held at {FIXED} for every window')
    print(f'{"window":>10} {"n":>5} {"theta*":>8} {"+clinch":>8} {"diff":>8}'
          f'   {"late games only":>22}')
    for label, lo, hi in WINDOWS:
        test = season_mask(games, min_season=lo, max_season=hi)
        tl = test & late
        print(f'{label:>10} {test.sum():>5} {L(E0[test], S0[test]):>8.4f} '
              f'{L(Ec[test], Sc[test]):>8.4f} '
              f'{L(E0[test], S0[test]) - L(Ec[test], Sc[test]):>+8.4f}   '
              f'{L(E0[tl], S0[tl]) - L(Ec[tl], Sc[tl]):>+8.4f} on {tl.sum():>4}')

    # --- 2. the whole-sample scale curve --------------------------------
    # A real effect should give a smooth curve with an interior minimum,
    # not a monotone slide (which would mean the grid is truncated) and
    # not a jagged one (noise).
    print(f'\nwhole-sample L by scale (in-sample, {len(games)} games):')
    print(f'{"scale":>7} {"L all":>9} {"L late":>9} {"Acc late":>9}')
    for scale in SCALE_GRID:
        E, S, _ = run_elo(games, clinch_map=clinch_map,
                          clinch_scale=scale, **base)
        print(f'{scale:>7} {L(E, S):>9.4f} {L(E[late], S[late]):>9.4f} '
              f'{Accuracy(E[late], S[late]):>9.4f}')

    # --- 3. what it costs elsewhere -------------------------------------
    # The adjustment is inert outside late REG weeks, so any change there
    # comes from ratings that updated against a different prediction.
    early = ~late
    print(f'\nspillover onto the {early.sum()} games it does not touch:')
    print(f'  theta*   L {L(E0[early], S0[early]):.4f}')
    print(f'  +clinch  L {L(Ec[early], Sc[early]):.4f}  '
          f'({L(E0[early], S0[early]) - L(Ec[early], Sc[early]):+.4f})')

    print(f'\nheadline, all {len(games)} games:')
    print(f'  theta*   L {L(E0, S0):.4f}  Acc {Accuracy(E0, S0):.4f}')
    print(f'  +clinch  L {L(Ec, Sc):.4f}  Acc {Accuracy(Ec, Sc):.4f}')
