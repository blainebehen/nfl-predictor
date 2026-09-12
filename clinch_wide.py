"""
Widening the seed-locked flag, and re-running the six windows on it.

The open question left by clinch_probe.py. 'top' -- clinched the #1 seed
outright -- carries by far the largest effect in the project (+0.4234,
t +4.1) on 19 team-games. Nineteen is the single-cell excursion this
project rejects on sight, so the model term fell back to the broad
'clinched' flag, which is weaker (+0.0575) and only managed 4 of 6
windows.

But #1-seed-clinched is not the population that rests starters. The
population is any team that is in the field and cannot improve its
position by winning, whatever seed that is. A team locked into the 5
seed has exactly as little to play for as a team locked into the 1 seed.
That population should be several times larger, and if the effect holds
on it, the evidence changes character.

Three flags are built here, each a strict subset of the one before:

    in       clinched a berth
    nogain   clinched, and cannot improve its seed by winning
    locked   clinched, and its conference rank cannot change at all

APPROXIMATIONS, all documented because they decide the answer:

Seeding is by division winner first (seeds 1-4) then wildcard (5-7),
resolved on head-to-head, division and conference records. This uses win
totals and division-leader status only. Concretely:

  - t has clinched its division when no rival's maximum reaches t's
    current wins; t is out of the division race when a rival's current
    wins exceed t's maximum.
  - "cannot improve" means: t's division outcome is settled either way,
    AND no team currently ahead of t inside its own seed group (division
    winners, or wildcards) is reachable -- t's maximum falls short of
    their current wins.
  - "rank cannot change" additionally requires that no team currently
    behind t can reach it.

Ties count half a win. Tiebreakers are ignored, so a seed settled only
on a tiebreaker is not flagged. Every approximation here UNDER-flags,
which shrinks the measured effect rather than inventing one.
"""
import numpy as np
import pandas as pd

from data import load_games
from clinch import (team_meta, conference_of, division_of, berths,
                    FIRST_WEEK)

META = None


def build_wide_map(games, first_week=FIRST_WEEK):
    """dict: (game_id, team) -> label in {elim, live, in, nogain, locked}."""
    global META
    if META is None:
        META = team_meta()
    meta = META

    reg = games[games.game_type == 'REG']
    out = {}

    for season, sgames in reg.groupby('season'):
        n_berths = berths(season)
        scheduled = pd.concat([sgames.home_team,
                               sgames.away_team]).value_counts()

        for week in sorted(sgames.week.unique()):
            if week < first_week:
                continue
            prior = sgames[sgames.week < week]
            if prior.empty:
                continue

            wins, played = {}, {}
            for g in prior.itertuples():
                for t in (g.home_team, g.away_team):
                    played[t] = played.get(t, 0) + 1
                    wins.setdefault(t, 0.0)
                if g.result > 0:
                    wins[g.home_team] += 1.0
                elif g.result < 0:
                    wins[g.away_team] += 1.0
                else:
                    wins[g.home_team] += 0.5
                    wins[g.away_team] += 0.5

            teams = list(wins)
            maxw = {t: wins[t] + (scheduled.get(t, 0) - played[t])
                    for t in teams}
            conf = {t: conference_of(t, season, meta) for t in teams}
            div = {t: division_of(t, season, meta) for t in teams}

            # who currently leads each division -- the seed-group split
            leaders = {}
            for t in teams:
                d = div[t]
                if d is None:
                    continue
                if d not in leaders or wins[t] > wins[leaders[d]]:
                    leaders[d] = t

            status = {}
            for t in teams:
                c = conf[t]
                others = [j for j in teams if j != t and conf[j] == c]
                rivals = [j for j in others
                          if div[j] is not None and div[j] == div[t]]

                div_won = (bool(rivals)
                           and all(maxw[j] < wins[t] for j in rivals))
                div_dead = (any(wins[j] > maxw[t] for j in rivals)
                            if rivals else True)

                # --- eliminated ---------------------------------------
                ahead = sum(1 for j in others if wins[j] > maxw[t])
                if div_dead and ahead >= n_berths:
                    status[t] = 'elim'
                    continue

                # --- clinched a berth ---------------------------------
                could_pass = sum(1 for j in others if maxw[j] > wins[t])
                if could_pass >= n_berths:
                    status[t] = 'live'
                    continue

                # t is in the field. Can winning improve its position?
                # The division race must be settled either way first --
                # an unsettled division is always something to play for.
                if not (div_won or div_dead):
                    status[t] = 'in'
                    continue

                # seed group: division winners seed above wildcards, so
                # a team can only climb past teams in its own group
                if div_won:
                    group = [j for j in others if j in leaders.values()]
                else:
                    group = [j for j in others if j not in leaders.values()]

                above = [j for j in group if wins[j] > wins[t]]
                below = [j for j in group if wins[j] < wins[t]]

                can_climb = any(maxw[t] >= wins[j] for j in above)
                can_fall = any(maxw[j] >= wins[t] for j in below)

                if can_climb:
                    status[t] = 'in'
                elif can_fall:
                    status[t] = 'nogain'
                else:
                    status[t] = 'locked'

            for g in sgames[sgames.week == week].itertuples():
                for t in (g.home_team, g.away_team):
                    if t in status:
                        out[(g.game_id, t)] = status[t]

    return out


def signed_shortfall(late, pred):
    """Residual of the flagged team, signed so positive = did worse."""
    v = []
    for r in late.itertuples():
        h, a = pred(r.h), pred(r.a)
        if h and not a:
            v.append(-r.resid)
        elif a and not h:
            v.append(r.resid)
    return np.array(v)


if __name__ == '__main__':
    from elo import (run_elo, QB_SCALE, QB_ALPHA, EPA_SCALE, EPA_ALPHA)
    from qb_data import build_qb_map
    from epa_test import team_game_epa

    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    E, S, _ = run_elo(games, k=20, H=50, rho=0.50, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                      epa_map=epa_map, epa_scale=EPA_SCALE,
                      epa_alpha=EPA_ALPHA)

    lab = build_wide_map(games)
    g = games.copy()
    g['resid'] = S - E
    g['h'] = [lab.get((r.game_id, r.home_team)) for r in games.itertuples()]
    g['a'] = [lab.get((r.game_id, r.away_team)) for r in games.itertuples()]
    late = g[g.h.notna()].copy()

    counts = pd.concat([late.h, late.a]).value_counts()
    print('team-game counts by flag:')
    print(counts.to_string())

    print(f'\n{"flag":>34} {"n":>6} {"shortfall":>10} {"t":>6}')
    tests = [
        ('locked (rank cannot change)', lambda x: x == 'locked'),
        ('nogain (cannot improve)', lambda x: x == 'nogain'),
        ('nogain + locked', lambda x: x in ('nogain', 'locked')),
        ('in (clinched, can still move)', lambda x: x == 'in'),
        ('any clinched', lambda x: x in ('in', 'nogain', 'locked')),
        ('eliminated', lambda x: x == 'elim'),
    ]
    for label, pred in tests:
        v = signed_shortfall(late, pred)
        if len(v) < 15:
            print(f'{label:>34} {len(v):>6}  too few')
            continue
        m, sd, n = v.mean(), v.std(), len(v)
        print(f'{label:>34} {n:>6} {m:>+10.4f} {m/(sd/np.sqrt(n)):>+6.1f}')

    print('\nby era, nogain + locked:')
    for lo, hi in ((1999, 2011), (2012, 2025)):
        sub = late[(late.season >= lo) & (late.season <= hi)]
        v = signed_shortfall(sub, lambda x: x in ('nogain', 'locked'))
        m, sd, n = v.mean(), v.std(), len(v)
        print(f'  {lo}-{hi}  n={n:>4}  shortfall {m:>+.4f}  '
              f't {m/(sd/np.sqrt(n)):>+.1f}')


def build_nogain_map(games, first_week=FIRST_WEEK, graded=False):
    """
    dict: (game_id, team) -> weight, for teams in the field that cannot
    improve their seed by winning.

    graded=False flags 'nogain' and 'locked' alike at 1.0 -- one
    parameter, and the simplest statement of the hypothesis.
    graded=True splits them 0.5 / 1.0, on the reading that a team which
    can still FALL has more reason to try than one whose rank is fixed.

    'in' -- clinched a berth but the seed is still live -- is not
    flagged. Its shortfall is +0.0146 on 341 team-games, t +0.6: nothing.
    That cell is what diluted the original 'any clinched' flag.
    """
    lab = build_wide_map(games, first_week=first_week)
    w = {'locked': 1.0, 'nogain': 0.5 if graded else 1.0}
    return {k: w[v] for k, v in lab.items() if v in w}
