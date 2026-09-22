"""
Playoff-status flags for the clinch experiment (tested, not adopted).

Moved out of features.py because the model doesn't use them.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import nflreadpy as nfl

from data import RELOCATED

FIRST_CLINCH_WEEK = 15    # REG weeks from here on can have settled teams
REALIGNMENT = 2002        # divisions before this season are not today's


# ------------------------------------------------------------- CLINCH

_META = None


def team_meta():
    """team -> (conference, division), with relocations merged."""
    global _META
    if _META is None:
        t = nfl.load_teams().to_pandas()
        _META = {RELOCATED.get(r.team_abbr, r.team_abbr):
                 (r.team_conf, r.team_division) for r in t.itertuples()}
    return _META


def conference_of(team, season, meta):
    # Seattle played in the AFC West until the 2002 realignment
    if team == 'SEA' and season < REALIGNMENT:
        return 'AFC'
    return meta.get(team, (None, None))[0]


def division_of(team, season, meta):
    if season < REALIGNMENT:
        return None
    return meta.get(team, (None, None))[1]


def berths(season):
    return 7 if season >= 2020 else 6


def _weekly_standings(games, first_week):
    """
    Yield (season, week, week_games, wins, maxw, conf, div, leaders) for
    every REG week from `first_week` on.

    Standings entering week w use REG games of weeks < w only, so no game
    informs its own status. Shared by both status maps below, which
    previously carried near-identical copies of this loop.

    wins counts a tie as half. maxw is the most wins a team can still
    finish with. leaders maps each division to its current wins leader,
    which is what separates the two seeding groups: division winners take
    seeds 1-4, wildcards 5-7.
    """
    meta = team_meta()
    reg = games[games.game_type == 'REG']

    for season, sgames in reg.groupby('season'):
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

            leaders = {}
            for t in teams:
                d = div[t]
                if d is None:
                    continue
                if d not in leaders or wins[t] > wins[leaders[d]]:
                    leaders[d] = t

            yield (season, week, sgames[sgames.week == week],
                   wins, maxw, conf, div, leaders)


def build_status_map(games, first_week=FIRST_CLINCH_WEEK):
    """
    dict: (game_id, team) -> one of

        elim    eliminated from the playoffs
        live    still competing for a berth
        in      clinched a berth, seed still live
        nogain  clinched, cannot improve its seed, could still fall
        locked  clinched, conference rank cannot move either way

    Only REG games from `first_week` on get an entry.

    APPROXIMATIONS, documented because they decide the answer. Real
    seeding resolves on head-to-head, division and conference records and
    strength of victory; this uses win totals and division-leader status
    only. A team has clinched its division when no rival's maximum
    reaches its current wins, and is out of the race when a rival's
    current wins exceed its maximum. "Cannot improve" means the division
    outcome is settled either way AND no team currently ahead of it
    inside its own seed group is reachable. "Rank cannot move" adds that
    no team currently behind it can reach it.

    Ties count half a win and tiebreakers are ignored, so a seed settled
    only on a tiebreaker is not flagged. Every approximation here
    UNDER-flags, which shrinks a measured effect rather than inventing
    one.
    """
    out = {}
    for (season, week, wk_games, wins, maxw, conf, div, leaders) \
            in _weekly_standings(games, first_week):
        n_berths = berths(season)
        status = {}

        for t in wins:
            others = [j for j in wins if j != t and conf[j] == conf[t]]
            rivals = [j for j in others
                      if div[j] is not None and div[j] == div[t]]

            div_won = bool(rivals) and all(maxw[j] < wins[t] for j in rivals)
            div_dead = (any(wins[j] > maxw[t] for j in rivals)
                        if rivals else True)

            if div_dead and sum(1 for j in others
                                if wins[j] > maxw[t]) >= n_berths:
                status[t] = 'elim'
                continue

            if sum(1 for j in others if maxw[j] > wins[t]) >= n_berths:
                status[t] = 'live'
                continue

            # In the field. An unsettled division is always something to
            # play for, whatever the seeding looks like.
            if not (div_won or div_dead):
                status[t] = 'in'
                continue

            if div_won:
                group = [j for j in others if j in leaders.values()]
            else:
                group = [j for j in others if j not in leaders.values()]

            can_climb = any(maxw[t] >= wins[j] for j in group
                            if wins[j] > wins[t])
            can_fall = any(maxw[j] >= wins[t] for j in group
                           if wins[j] < wins[t])

            status[t] = ('in' if can_climb
                         else 'nogain' if can_fall else 'locked')

        for g in wk_games.itertuples():
            for t in (g.home_team, g.away_team):
                if t in status:
                    out[(g.game_id, t)] = status[t]

    return out


def build_stake_map(games, first_week=FIRST_CLINCH_WEEK, detail=False):
    """
    The FIRST version of the clinch flag, kept so RESULTS.md stays
    reproducible. Superseded by build_status_map.

    Labels are elim / top / in / live, where 'top' is only "clinched the
    #1 seed outright". stake values are 0.0 for elim and top, 0.5 for in,
    1.0 for live -- which is the bug the writeup describes: eliminated
    and locked teams move in opposite directions and share a value, so
    pooling them cancels the effect. Do not build anything new on this.
    """
    out = {}
    for (season, week, wk_games, wins, maxw, conf, div, leaders) \
            in _weekly_standings(games, first_week):
        n_berths = berths(season)
        status = {}

        for t in wins:
            others = [j for j in wins if j != t and conf[j] == conf[t]]
            rivals = [j for j in others
                      if div[j] is not None and div[j] == div[t]]
            div_dead = (any(wins[j] > maxw[t] for j in rivals)
                        if rivals else True)

            if div_dead and sum(1 for j in others
                                if wins[j] > maxw[t]) >= n_berths:
                status[t] = 'elim' if detail else 0.0
            elif sum(1 for j in others if maxw[j] > wins[t]) < n_berths:
                top = all(maxw[j] < wins[t] for j in others)
                if detail:
                    status[t] = 'top' if top else 'in'
                else:
                    status[t] = 0.0 if top else 0.5
            else:
                status[t] = 'live' if detail else 1.0

        for g in wk_games.itertuples():
            for t in (g.home_team, g.away_team):
                if t in status:
                    out[(g.game_id, t)] = status[t]

    return out


def build_clinch_map(games, flags=('locked',), first_week=FIRST_CLINCH_WEEK):
    """
    dict: (game_id, team) -> 1.0 for teams carrying any of `flags`.

    The form run_elo takes. The default is the rank-frozen population,
    which is the one that carries the effect -- 56 team-games, shortfall
    +0.2426, t +3.8. Widening to ('nogain', 'locked') or to ('in',
    'nogain', 'locked') dilutes it; see RESULTS.md.

    Eliminated teams are deliberately never flagged. They show the
    opposite sign, and pooling them with clinched teams is what made the
    first residual test come back flat.
    """
    lab = build_status_map(games, first_week=first_week)
    return {k: 1.0 for k, v in lab.items() if v in flags}
