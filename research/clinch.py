"""
Clinch status: the whole investigation, three passes.

    python research/clinch.py residuals   does the flag predict the error?
    python research/clinch.py windows     does it survive six held-out windows?
    python research/clinch.py             both

Was clinch_test.py, clinch_probe.py, clinch_windows.py, clinch_fixed.py,
clinch_wide_windows.py and locked_test.py. The flag builders moved to
features.py, since elo.py and predict.py need them and this file is the
homework, not the model.

THE SHORT VERSION. Teams that have nothing left to play for rest their
starters; the market prices it and a rating system cannot see it. The
first flag scored "nothing at stake", pooling eliminated teams with
seed-locked teams -- opposite effects, and the pool was 98% eliminated
teams, so the residual test came back flat. Split apart, the effect is
there. Narrowing to the teams whose rank is genuinely frozen gives the
largest effect in the project (+0.2426, t +3.8) on 56 team-games.

Not adopted. 4 of 6 windows, and the adjustment is nonzero in 54 games
out of 7,278 -- enough to measure the effect confidently, not enough to
calibrate it. Full writeup, including what is wrong with the evidence,
in RESULTS.md.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import load_games
from elo import (run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from features import (build_qb_map, team_game_epa, build_status_map,
                      build_stake_map, build_clinch_map)

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50

WINDOWS = [
    ('2004-2009', 2004, 2009),
    ('2010-2015', 2010, 2015),
    ('2016-2018', 2016, 2018),
    ('2019-2021', 2019, 2021),
    ('2022-2025', 2022, 2025),
    ('2019-2025', 2019, 2025),
]

# scale grids: the first pass used a narrow one, the rank-frozen flag
# needs a wider one because the per-team adjustment is much larger
SCALE_GRID = [0, 15, 30, 50, 75, 110, 160]
LOCKED_GRID = [0, 40, 80, 130, 190, 260, 340]


def theta_star(games, qb_map, epa_map, **kw):
    """The adopted model, which is the control for everything here."""
    return run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                   qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                   epa_map=epa_map, epa_scale=EPA_SCALE,
                   epa_alpha=EPA_ALPHA, **kw)


def signed(late, pred):
    """
    Residual of the flagged team, signed so positive means it did WORSE
    than the model predicted.

    Only games where exactly one side carries the flag count -- with both
    flagged there is no contrast to measure, which is why a combined row
    is smaller than the sum of its parts.
    """
    v = []
    for r in late.itertuples():
        h, a = pred(r.h), pred(r.a)
        if h and not a:
            v.append(-r.resid)
        elif a and not h:
            v.append(r.resid)
    return np.array(v)


def cell(v, label):
    if len(v) < 15:
        return f'{label:>32} {len(v):>6}  too few'
    m, sd, n = v.mean(), v.std(), len(v)
    return f'{label:>32} {n:>6} {m:>+11.4f} {m / (sd / np.sqrt(n)):>+6.1f}'


# ------------------------------------------------------------ residuals

def residuals(games, qb_map, epa_map):
    E, S, _ = theta_star(games, qb_map, epa_map)

    g = games.copy()
    g['resid'] = S - E
    g['home_win'] = (g.result > 0).astype(float)

    # --- pass 1: the stake differential, which came back flat ----------
    stake = build_stake_map(games)
    diff = []
    for r in games.itertuples():
        h = stake.get((r.game_id, r.home_team))
        a = stake.get((r.game_id, r.away_team))
        diff.append(0.0 if h is None or a is None else h - a)
    g['stake_diff'] = diff
    g['has_stake'] = [(r.game_id, r.home_team) in stake
                      for r in games.itertuples()]
    late = g[g.has_stake]

    agg = late.groupby('stake_diff', observed=True).agg(
        n=('resid', 'size'), raw=('home_win', 'mean'),
        resid=('resid', 'mean'), sd=('resid', 'std'))
    agg['t'] = agg.resid / (agg.sd / np.sqrt(agg.n))

    print(f'PASS 1 -- residual by stake differential, {len(late)} late games')
    print(f'{"bucket":>10} {"n":>6} {"raw":>7} {"resid":>9} {"t":>6}')
    for idx, r in agg.iterrows():
        print(f'{idx:>10.1f} {int(r.n):>6} {r.raw:>7.3f} '
              f'{r.resid:>+9.4f} {r.t:>+6.1f}')
    print('\nA perfectly monotone RAW gradient and residuals that are')
    print('scatter. On the reading that killed rest and travel, stop here.')

    # --- pass 2: why it was flat ---------------------------------------
    lab = build_status_map(games)
    g['h'] = [lab.get((r.game_id, r.home_team)) for r in games.itertuples()]
    g['a'] = [lab.get((r.game_id, r.away_team)) for r in games.itertuples()]
    late = g[g.h.notna()].copy()

    print(f'\nPASS 2 -- the same games, split by what the flag actually is')
    print('The first flag gave eliminated and seed-locked teams the same')
    print('score. They move in OPPOSITE directions and the pool was 98%')
    print('eliminated, so they cancelled.\n')
    print(f'{"":>32} {"n":>6} {"shortfall":>11} {"t":>6}')
    for label, pred in (
            ('eliminated', lambda x: x == 'elim'),
            ('clinched, seed still live', lambda x: x == 'in'),
            ('cannot improve (nogain)', lambda x: x == 'nogain'),
            ('rank frozen (locked)', lambda x: x == 'locked'),
            ('nogain + locked', lambda x: x in ('nogain', 'locked')),
            ('any clinched', lambda x: x in ('in', 'nogain', 'locked'))):
        print(cell(signed(late, pred), label))

    print('\nshortfall > 0 means the flagged team did worse than predicted,')
    print('pooled across home and away -- two independent samples.')

    print('\nby era, nogain + locked:')
    for lo, hi in ((1999, 2011), (2012, 2025)):
        v = signed(late[(late.season >= lo) & (late.season <= hi)],
                   lambda x: x in ('nogain', 'locked'))
        m, sd, n = v.mean(), v.std(), len(v)
        print(f'  {lo}-{hi}  n={n:>4}  shortfall {m:>+.4f}  '
              f't {m / (sd / np.sqrt(n)):>+.1f}')

    counts = pd.concat([late.h, late.a]).value_counts()
    print(f'\nteam-games by flag:\n{counts.to_string()}')


# -------------------------------------------------------------- windows

def tune(games, qb_map, epa_map, cmap, grid, max_season):
    """Pick the scale that minimises L on everything before the window."""
    tr = season_mask(games, None, max_season)
    res = []
    for s in grid:
        E, S, _ = theta_star(games, qb_map, epa_map,
                             clinch_map=cmap, clinch_scale=s)
        res.append((L(E[tr], S[tr]), s))
    res.sort()
    return res[0][1]


def window_table(games, qb_map, epa_map, cmap, grid, E0, S0, late, title):
    print(f'\n{title}')
    print(f'{"window":>10} {"n":>5} {"theta*":>8} {"+clinch":>8} '
          f'{"diff":>8} {"tuned":>6}   {"late-only":>18}')
    npos, scales = 0, []
    for label, lo, hi in WINDOWS:
        s = tune(games, qb_map, epa_map, cmap, grid, lo - 1)
        Ec, Sc, _ = theta_star(games, qb_map, epa_map,
                               clinch_map=cmap, clinch_scale=s)
        t = season_mask(games, min_season=lo, max_season=hi)
        tl = t & late
        d = L(E0[t], S0[t]) - L(Ec[t], Sc[t])
        npos += d > 0
        scales.append(s)
        print(f'{label:>10} {t.sum():>5} {L(E0[t], S0[t]):>8.4f} '
              f'{L(Ec[t], Sc[t]):>8.4f} {d:>+8.4f} {s:>6}   '
              f'{L(E0[tl], S0[tl]) - L(Ec[tl], Sc[tl]):>+8.4f} '
              f'on {tl.sum():>4}')
    print(f'  {npos} of {len(WINDOWS)} positive; scales {sorted(set(scales))}')
    return npos


def windows(games, qb_map, epa_map):
    E0, S0, _ = theta_star(games, qb_map, epa_map)
    late = ((games.game_type == 'REG') & (games.week >= 15)).values

    # pass 1 flag: any team that has clinched a berth, from the ORIGINAL
    # labels, kept so the RESULTS.md table stays reproducible
    stake_lab = build_stake_map(games, detail=True)
    broad = {k: 1.0 for k, v in stake_lab.items() if v in ('in', 'top')}
    window_table(games, qb_map, epa_map, broad, SCALE_GRID, E0, S0, late,
                 f'PASS 1 -- "has clinched a berth", {len(broad)} team-games')

    # fixed scale: not a validation (50 was chosen knowing the whole
    # sample), but it separates "does the effect exist" from "can five
    # seasons estimate its size"
    print('\nsame flag, scale held at 50 for every window:')
    Ef, Sf, _ = theta_star(games, qb_map, epa_map,
                           clinch_map=broad, clinch_scale=50)
    for label, lo, hi in WINDOWS:
        t = season_mask(games, min_season=lo, max_season=hi)
        print(f'{label:>10} {L(E0[t], S0[t]) - L(Ef[t], Sf[t]):>+9.4f}')
    print('\nin-sample L on late games by scale (the shape is the evidence):')
    for s in SCALE_GRID:
        E, S, _ = theta_star(games, qb_map, epa_map,
                             clinch_map=broad, clinch_scale=s)
        print(f'  scale {s:>4}  L {L(E[late], S[late]):.4f}')
    print('  smooth with a clean interior minimum; noise gives a jagged')
    print('  curve or a monotone slide into the grid edge')

    # pass 2 flag: cannot improve its seed
    nogain = build_clinch_map(games, flags=('nogain', 'locked'))
    window_table(games, qb_map, epa_map, nogain, LOCKED_GRID, E0, S0, late,
                 f'PASS 2 -- "cannot improve", {len(nogain)} team-games')

    # pass 3 flag: rank frozen both ways
    locked = build_clinch_map(games, flags=('locked',))
    window_table(games, qb_map, epa_map, locked, LOCKED_GRID, E0, S0, late,
                 f'PASS 3 -- "rank frozen", {len(locked)} team-games')
    print('  five of six select 130 -- the only stable parameter any')
    print('  version of this produced')

    # the other mechanism: refuse to learn from those games at all
    skip = {gid for (gid, _) in locked}
    Ex, Sx, _ = theta_star(games, qb_map, epa_map, skip_games=skip)
    print(f'\nEXCLUDE instead of adjust -- same {len(skip)} games, '
          f'predicted and scored but not learned from:')
    print(f'{"window":>10} {"theta*":>8} {"-locked":>8} {"diff":>8}')
    npos = 0
    for label, lo, hi in WINDOWS:
        t = season_mask(games, min_season=lo, max_season=hi)
        d = L(E0[t], S0[t]) - L(Ex[t], Sx[t])
        npos += d > 0
        print(f'{label:>10} {L(E0[t], S0[t]):>8.4f} '
              f'{L(Ex[t], Sx[t]):>8.4f} {d:>+8.4f}')
    print(f'  {npos} of {len(WINDOWS)} positive; whole sample '
          f'{L(E0, S0):.4f} -> {L(Ex, Sx):.4f}, '
          f'Acc {Accuracy(E0, S0):.4f} -> {Accuracy(Ex, Sx):.4f}')
    print('\nThis settles the late-season-exclusion question in RESULTS.md.')
    print('That test dropped all of REG weeks 17-18 -- 1,016 team-games --')
    print('and came out slightly worse. It did not fail for being blunt.')
    print('Discarding a game costs more real signal than the rested-starter')
    print('noise it removes, however precisely it is aimed.')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'

    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    if what in ('residuals', 'both'):
        residuals(games, qb_map, epa_map)
    if what in ('windows', 'both'):
        windows(games, qb_map, epa_map)
