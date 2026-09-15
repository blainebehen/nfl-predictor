"""
Everything the model reads about a game, in one place.

Three feature families, each a map keyed by (game_id, team), each built
only from games played strictly before the game in question:

    build_qb_map      the starting quarterback and his EPA per dropback
    team_game_epa     a team's offensive and defensive EPA per play
    build_status_map  playoff standing: eliminated, live, clinched, locked

This file is imported by elo.py and predict.py. Nothing in research/ is.
That separation is the point of the file: the EPA extractor used to live
in epa_test.py and the clinch flags in clinch.py, so the model imported
its inputs from the scripts that were investigating them. Any of those
scripts could break the live forecast.

Nothing here decides anything. Tuning, residual tests and window
validation all live in research/; the numbers they produced are in
RESULTS.md.
"""
from datetime import date

import numpy as np
import pandas as pd
import nflreadpy as nfl

from data import RELOCATED

FIRST_CLINCH_WEEK = 15    # REG weeks from here on can have settled teams
REALIGNMENT = 2002        # divisions before this season are not today's


def current_season(today=None):
    """
    The NFL season currently in progress, or the most recent one.

    Seasons are named for the calendar year they start in and start in
    September, so anything before September belongs to the previous year.
    Hardcoding an end year meant the model silently stopped learning as
    soon as a new season began -- games appeared in the schedule but
    carried no QB or EPA value.
    """
    today = today or date.today()
    return today.year if today.month >= 9 else today.year - 1


SEASONS = list(range(1999, current_season() + 1))


# ---------------------------------------------------------------- QB

def qb_game_values(seasons=SEASONS):
    """
    One row per (game_id, team): the starting QB and his EPA per dropback.

    Starter = most pass attempts. Checked on 2024: in team-games with two
    QBs the second threw a median 4% of attempts, so this picks the real
    starter almost always.

    Value combines passing and rushing EPA -- ignoring the run would
    systematically underrate mobile quarterbacks. Dropbacks include sacks,
    since a sack is a failed dropback and belongs in the denominator.

    Sanity check on the output: career leaders are Mahomes, Jackson,
    Manning, Allen, Rodgers, Brees, Brady; trailers are Harrington,
    D.Carr, Sanchez. The measure agrees with what any football fan would
    say, which is the point of checking.
    """
    ps = nfl.load_player_stats(seasons).to_pandas()

    qb = ps[(ps.position == 'QB') & (ps.attempts > 0)].copy()
    qb = (qb.sort_values('attempts', ascending=False)
            .groupby(['game_id', 'team'], as_index=False)
            .first())

    dropbacks = qb.attempts + qb.sacks_suffered.fillna(0)
    epa = qb.passing_epa.fillna(0) + qb.rushing_epa.fillna(0)
    qb['value'] = np.where(dropbacks > 0,
                           epa / dropbacks.replace(0, np.nan), 0.0)
    qb['team'] = qb.team.replace(RELOCATED)

    return qb[['game_id', 'team', 'player_id', 'player_name', 'value']]


def build_qb_map(games, seasons=SEASONS, verbose=True):
    """
    dict: (game_id, team) -> (player_id, value)

    Coverage is 99.7% of team-games; the rest are games where no QB
    recorded an attempt. Those contribute no adjustment rather than being
    dropped. A much lower rate would mean team codes diverge between
    player_stats and schedules -- worth seeing rather than silently
    losing games.
    """
    qb = qb_game_values(seasons)
    qb_map = {(r.game_id, r.team): (r.player_id, r.value)
              for r in qb.itertuples()}

    if verbose:
        want = [(g.game_id, t) for g in games.itertuples()
                for t in (g.home_team, g.away_team)]
        hit = sum(1 for key in want if key in qb_map)
        print(f'QB coverage: {hit}/{len(want)} team-games '
              f'({hit / len(want):.1%})')

    return qb_map


# ---------------------------------------------------------------- EPA

def team_game_epa(seasons=SEASONS):
    """
    dict: (game_id, team) -> (offence EPA/play, defence EPA/play).

    Offence is passing + rushing EPA per play; receiving would count the
    same plays from the other side. Defence is not shipped by nflverse
    and is recovered as the opponent's offensive EPA in the same game.
    EPA is a total, so it is divided by plays (attempts + carries +
    sacks) -- a team running 75 plays accumulates more than one running
    55.
    """
    ts = nfl.load_team_stats(seasons=seasons).to_pandas()
    ts = ts[ts.season_type == 'REG'].copy()
    ts['team'] = ts.team.replace(RELOCATED)

    plays = (ts.attempts.fillna(0) + ts.carries.fillna(0)
             + ts.sacks_suffered.fillna(0))
    epa = ts.passing_epa.fillna(0) + ts.rushing_epa.fillna(0)
    ts['off_epa'] = np.where(plays > 0, epa / plays.replace(0, np.nan), 0.0)

    opp = ts[['game_id', 'team', 'off_epa']].rename(
        columns={'team': 'opponent_team', 'off_epa': 'def_epa'})
    ts = ts.merge(opp, on=['game_id', 'opponent_team'], how='left')

    return {(r.game_id, r.team): (r.off_epa, r.def_epa)
            for r in ts.itertuples() if pd.notna(r.def_epa)}


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


if __name__ == '__main__':
    from data import load_games

    games = load_games()

    qb_map = build_qb_map(games)
    print(f'EPA coverage: {len(team_game_epa())} team-games')

    lab = build_status_map(games)
    counts = pd.Series(list(lab.values())).value_counts()
    print(f'\nclinch status, {len(lab)} team-games '
          f'(REG weeks {FIRST_CLINCH_WEEK}+):')
    print(counts.to_string())

    print('\nsanity: 2024 week 18')
    wk = games[(games.season == 2024) & (games.week == 18)]
    by = {}
    for g in wk.itertuples():
        for t in (g.home_team, g.away_team):
            by.setdefault(lab.get((g.game_id, t), '?'), []).append(t)
    for k in ('elim', 'in', 'nogain', 'locked', 'live'):
        if k in by:
            print(f'  {k:<7} {" ".join(sorted(by[k]))}')
