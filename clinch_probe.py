"""
Second pass on clinch status, after the headline test came back flat.

The first cut (clinch_test.py) gave a perfectly monotone RAW gradient --
.426 .463 .573 .656 .732 across the stake differential -- and residuals
that are scatter, max |t| 1.5. That is the travel pattern exactly: teams
still alive in week 15 are good teams, teams eliminated are bad ones, and
the ratings already carry it.

But the first cut had a design flaw. It gave stake 0.0 to two situations
that are not alike:

    elim   eliminated from the playoffs
    top    clinched the #1 seed, nothing left to gain

Only the second rests starters. An eliminated team is just a bad team
playing its normal lineup, and Elo already knows it is bad. Pooling them
dilutes whatever signal 'top' carries -- and 'top' is rare, so the pooled
bucket is mostly 'elim'.

This file splits the four labels apart and tests each. It also reports
what share of the bucket each label holds, since a flat pooled cell made
of two opposing effects and a flat cell look the same until you split it.
"""
import numpy as np
import pandas as pd

from data import load_games
from elo import (run_elo, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from clinch import build_stake_map

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50
ORDER = ['elim', 'in', 'top', 'live']


def cell(sub, label):
    n = len(sub)
    if n < 15:
        return f'{label:>28} {n:>6}  too few'
    m, sd = sub.resid.mean(), sub.resid.std()
    t = m / (sd / np.sqrt(n))
    return (f'{label:>28} {n:>6} {sub.home_win.mean():>7.3f} '
            f'{m:>+8.4f} {t:>+6.1f}')


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                      epa_map=epa_map, epa_scale=EPA_SCALE,
                      epa_alpha=EPA_ALPHA)

    lab = build_stake_map(games, detail=True)

    g = games.copy()
    g['resid'] = S - E
    g['home_win'] = (g.result > 0).astype(float)
    g['h'] = [lab.get((r.game_id, r.home_team)) for r in games.itertuples()]
    g['a'] = [lab.get((r.game_id, r.away_team)) for r in games.itertuples()]
    late = g[g.h.notna()].copy()

    print(f'late REG games: {len(late)}')
    print('\nlabel counts (team-games):')
    counts = pd.concat([late.h, late.a]).value_counts()
    print(counts.to_string())

    # --- how much of the pooled 0.0 bucket was 'elim'? ------------------
    print(f"\nthe pooled stake=0.0 bucket was "
          f"{counts.get('elim',0)} elim vs {counts.get('top',0)} top "
          f"-- {counts.get('elim',0)/(counts.get('elim',0)+counts.get('top',1)):.0%} "
          f"eliminated teams")

    # --- residual by each label, home side and away side ----------------
    print(f'\n{"":>28} {"n":>6} {"raw":>7} {"resid":>8} {"t":>6}')
    print('HOME team label:')
    for k in ORDER:
        print(cell(late[late.h == k], k))
    print('AWAY team label  (resid > 0 = home beat model, i.e. away '
          'underperformed):')
    for k in ORDER:
        print(cell(late[late.a == k], k))

    # --- the direct test of the resting story ---------------------------
    # A team with nothing to gain against a team with everything to play
    # for. If resting starters costs games, these are where it shows.
    print('\nthe resting hypothesis, stated directly:')
    print(f'{"":>28} {"n":>6} {"raw":>7} {"resid":>8} {"t":>6}')
    print(cell(late[(late.h == 'top') & (late.a == 'live')],
               'home locked vs live away'))
    print(cell(late[(late.a == 'top') & (late.h == 'live')],
               'away locked vs live home'))
    print(cell(late[(late.h == 'elim') & (late.a == 'live')],
               'home elim vs live away'))
    print(cell(late[(late.a == 'elim') & (late.h == 'live')],
               'away elim vs live home'))

    # --- pooling 'top' on both sides into one signed sample -------------
    # Sign-flip the away cases so every row reads "the locked team's
    # residual". More power than either cell alone.
    locked = []
    for r in late.itertuples():
        if r.h == 'top' and r.a != 'top':
            locked.append(-r.resid)        # home is locked; its shortfall
        elif r.a == 'top' and r.h != 'top':
            locked.append(r.resid)
    locked = np.array(locked)
    if len(locked) >= 15:
        m, sd, n = locked.mean(), locked.std(), len(locked)
        print(f'\nlocked team, both sides pooled and signed: n={n}  '
              f'mean shortfall {m:+.4f}  t {m/(sd/np.sqrt(n)):+.1f}')
        print('(positive = the locked team did WORSE than the model said)')

    # same for eliminated, as a comparison
    elim = []
    for r in late.itertuples():
        if r.h == 'elim' and r.a != 'elim':
            elim.append(-r.resid)
        elif r.a == 'elim' and r.h != 'elim':
            elim.append(r.resid)
    elim = np.array(elim)
    m, sd, n = elim.mean(), elim.std(), len(elim)
    print(f'eliminated team, same construction:        n={n}  '
          f'mean shortfall {m:+.4f}  t {m/(sd/np.sqrt(n)):+.1f}')

    # --- the usable population -------------------------------------------
    # 'top' has the effect but only 19 team-games -- far too few to build
    # on, and exactly the single-cell excursion that killed travel. 'in'
    # is the same situation in weaker form (berth secured, seed still
    # live) and has 600. Test it the same way; the two sides of the
    # comparison are independent samples, so agreement between them is
    # worth more than either t alone.
    def signed(pred):
        out = []
        for r in late.itertuples():
            if pred(r.h) and not pred(r.a):
                out.append(-r.resid)
            elif pred(r.a) and not pred(r.h):
                out.append(r.resid)
        return np.array(out)

    print('\nshortfall of the flagged team, pooled and signed:')
    print(f'{"":>28} {"n":>6} {"shortfall":>10} {"t":>6}')
    for label, pred in (
            ('clinched berth (in)', lambda x: x == 'in'),
            ('clinched, any (in+top)', lambda x: x in ('in', 'top')),
            ('locked top seed (top)', lambda x: x == 'top'),
            ('eliminated (elim)', lambda x: x == 'elim')):
        v = signed(pred)
        if len(v) < 15:
            print(f'{label:>28} {len(v):>6}  too few')
            continue
        m, sd, n = v.mean(), v.std(), len(v)
        print(f'{label:>28} {n:>6} {m:>+10.4f} {m/(sd/np.sqrt(n)):>+6.1f}')

    # --- does it hold across eras? ---------------------------------------
    # One pooled t can come from one unusual stretch. Split by era; the
    # feature has to show up in both halves or it is a period artifact.
    print('\nclinched (in+top) shortfall by era:')
    for lo, hi in ((1999, 2011), (2012, 2025)):
        sub = late[(late.season >= lo) & (late.season <= hi)]
        out = []
        for r in sub.itertuples():
            hin, ain = r.h in ('in', 'top'), r.a in ('in', 'top')
            if hin and not ain:
                out.append(-r.resid)
            elif ain and not hin:
                out.append(r.resid)
        v = np.array(out)
        m, sd, n = v.mean(), v.std(), len(v)
        print(f'  {lo}-{hi}  n={n:>4}  shortfall {m:>+.4f}  '
              f't {m/(sd/np.sqrt(n)):>+.1f}')

    # --- week 18 only ---------------------------------------------------
    # Resting is concentrated in the final week. Smaller sample, cleaner
    # exposure.
    w18 = late[late.week == 18]
    l18 = []
    for r in w18.itertuples():
        if r.h == 'top' and r.a != 'top':
            l18.append(-r.resid)
        elif r.a == 'top' and r.h != 'top':
            l18.append(r.resid)
    l18 = np.array(l18)
    if len(l18) >= 10:
        m, sd, n = l18.mean(), l18.std(), len(l18)
        print(f'\nweek 18 only, locked team: n={n}  shortfall {m:+.4f}  '
              f't {m/(sd/np.sqrt(n)):+.1f}')
    else:
        print(f'\nweek 18 only, locked team: n={len(l18)}, too few')
