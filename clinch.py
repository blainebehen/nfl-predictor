"""
Clinch status for late-season games -- the targeted fix named in RESULTS.md.

Two findings in this project point at the same gap. 15 of the 20 largest
disagreements with the closing line are Week 16-18 games. And the naive
starter rule in predict.py picked third-stringers for Kansas City and
Denver because those teams had their seeds locked in Week 18. Both are the
same thing: a team with nothing to play for rests its starters, the market
prices it, and a rating system cannot see it.

Blanket exclusion of late games was tried and rejected (L 0.6314 vs 0.6310)
-- most late games still involve teams competing, so it discards signal to
remove noise. This file builds the targeted version: identify WHICH teams
have nothing at stake, and adjust only those games.

STAKE
-----
Per team, entering a game, on games completed strictly before it:

    0.0  eliminated from the playoffs, or has clinched the #1 seed
    0.5  has clinched a berth but not the top seed
    1.0  still competing for a berth

Both ends of the 0.0 case rest starters for the same reason -- the result
cannot change the outcome -- so they share a value. 0.5 is the middle case:
in the field, but seeding is still live.

The adjustment entering d is clinch_scale * (stake_home - stake_away), and
it is zero outside REG weeks >= FIRST_WEEK, where nothing is settled.

APPROXIMATIONS
--------------
Real clinching runs on head-to-head, division and conference records, and
strength of victory. This uses win totals only:

  - a team is eliminated if a division rival's current wins exceed its own
    maximum possible AND `berths` conference teams' current wins do too;
  - a team has clinched if fewer than `berths` other conference teams could
    still finish above its current wins;
  - #1 seed is clinched when no conference team's maximum reaches it.

Ties count half a win throughout. Tiebreakers are ignored, which makes both
flags CONSERVATIVE: a team clinching on a tiebreaker is not flagged until
the arithmetic alone settles it. Under-flagging weakens the measured effect
rather than inventing one, which is the safe direction for a feature test.

Divisions were realigned in 2002, so division logic only runs from 2002 on;
before that only the conference test applies. Playoff berths per conference
are 6 through 2019 and 7 from 2020.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl

from data import RELOCATED

FIRST_WEEK = 15       # REG weeks from here on can have settled teams
REALIGNMENT = 2002    # divisions before this season are not today's


def team_meta():
    """team -> (conference, division), with relocations merged."""
    t = nfl.load_teams().to_pandas()
    meta = {}
    for r in t.itertuples():
        team = RELOCATED.get(r.team_abbr, r.team_abbr)
        meta[team] = (r.team_conf, r.team_division)
    # Seattle played in the AFC West until the 2002 realignment; handled
    # by season in conference_of() below.
    return meta


META = None


def conference_of(team, season, meta):
    if team == 'SEA' and season < REALIGNMENT:
        return 'AFC'
    return meta.get(team, (None, None))[0]


def division_of(team, season, meta):
    if season < REALIGNMENT:
        return None
    return meta.get(team, (None, None))[1]


def berths(season):
    return 7 if season >= 2020 else 6


def build_stake_map(games, first_week=FIRST_WEEK, detail=False):
    """
    dict: (game_id, team) -> stake in {0.0, 0.5, 1.0}

    With detail=True the value is a label instead -- 'elim', 'top', 'in',
    'live'. The two labels sharing stake 0.0 are different situations and
    the residual test needs to see them apart: a locked team rests its
    starters, an eliminated team is simply bad, and the ratings already
    know about bad.

    Only REG games from `first_week` on get an entry. Everything else is
    absent and contributes no adjustment.

    Standings entering week w use REG games of weeks < w only, so no game
    informs its own stake.
    """
    global META
    if META is None:
        META = team_meta()
    meta = META

    reg = games[games.game_type == 'REG']
    stake = {}

    for season, sgames in reg.groupby('season'):
        n_berths = berths(season)

        # how many REG games each team is scheduled to play this season
        scheduled = pd.concat([sgames.home_team, sgames.away_team]).value_counts()

        for week in sorted(sgames.week.unique()):
            if week < first_week:
                continue

            prior = sgames[sgames.week < week]
            if prior.empty:
                continue

            # --- standings entering this week -------------------------
            wins, played = {}, {}
            for g in prior.itertuples():
                h, a = g.home_team, g.away_team
                for t in (h, a):
                    played[t] = played.get(t, 0) + 1
                    wins.setdefault(t, 0.0)
                if g.result > 0:
                    wins[h] += 1.0
                elif g.result < 0:
                    wins[a] += 1.0
                else:
                    wins[h] += 0.5
                    wins[a] += 0.5

            teams = list(wins)
            maxw = {t: wins[t] + (scheduled.get(t, 0) - played[t]) for t in teams}

            conf = {t: conference_of(t, season, meta) for t in teams}
            div = {t: division_of(t, season, meta) for t in teams}

            status = {}
            for t in teams:
                c = conf[t]
                others = [j for j in teams if j != t and conf[j] == c]

                # --- eliminated ---------------------------------------
                # cannot win the division: a rival already has more wins
                # than t can finish with
                rivals = [j for j in others if div[j] is not None
                          and div[j] == div[t]]
                div_dead = any(wins[j] > maxw[t] for j in rivals) if rivals else True

                # cannot reach the field: `berths` conference teams
                # already have more wins than t can finish with
                ahead = sum(1 for j in others if wins[j] > maxw[t])
                wc_dead = ahead >= n_berths

                if div_dead and wc_dead:
                    status[t] = 'elim' if detail else 0.0
                    continue

                # --- clinched -----------------------------------------
                # fewer than `berths` conference teams could still finish
                # above t's current wins
                could_pass = sum(1 for j in others if maxw[j] > wins[t])
                if could_pass < n_berths:
                    # top seed: nobody in the conference can reach t
                    top = all(maxw[j] < wins[t] for j in others)
                    if detail:
                        status[t] = 'top' if top else 'in'
                    else:
                        status[t] = 0.0 if top else 0.5
                else:
                    status[t] = 'live' if detail else 1.0

            for g in sgames[sgames.week == week].itertuples():
                for t in (g.home_team, g.away_team):
                    if t in status:
                        stake[(g.game_id, t)] = status[t]

    return stake


def build_clinch_map(games, first_week=FIRST_WEEK):
    """
    dict: (game_id, team) -> 1.0 for a team that has clinched a playoff
    berth entering that game. Everything else is absent.

    This is the form the model uses. Note what it does NOT flag:
    eliminated teams. The first design pooled them with clinched teams as
    "nothing at stake", which is how the headline residual test came back
    flat -- the two groups move in opposite directions and cancel.
    Eliminated teams slightly OVERperform their rating (-0.029, t -1.6);
    clinched teams underperform (+0.058, t +2.8). Only the second is a
    feature. See clinch_probe.py.
    """
    labels = build_stake_map(games, first_week=first_week, detail=True)
    return {key: 1.0 for key, lab in labels.items() if lab in ('in', 'top')}


def stake_diff(games, stake_map):
    """Per game: stake_home - stake_away, 0.0 where either side is absent."""
    out = []
    for g in games.itertuples():
        h = stake_map.get((g.game_id, g.home_team))
        a = stake_map.get((g.game_id, g.away_team))
        out.append(0.0 if h is None or a is None else h - a)
    return np.array(out)


if __name__ == '__main__':
    from data import load_games

    games = load_games()
    stake = build_stake_map(games)

    print(f'stake assigned to {len(stake)} team-games '
          f'(REG weeks {FIRST_WEEK}+)\n')

    df = pd.DataFrame(
        [(gid, t, v) for (gid, t), v in stake.items()],
        columns=['game_id', 'team', 'stake'])
    wk = games.set_index('game_id').week
    sn = games.set_index('game_id').season
    df['week'] = df.game_id.map(wk)
    df['season'] = df.game_id.map(sn)

    print('share of team-games at each stake level, by week:')
    tab = (df.groupby(['week', 'stake']).size()
             .unstack(fill_value=0))
    tab = tab.div(tab.sum(axis=1), axis=0).round(3)
    print(tab.to_string())

    print('\nteams with nothing at stake (0.0), count per season, week 17-18:')
    late = df[(df.week >= 17) & (df.stake == 0.0)]
    print(late.groupby('season').size().to_string())

    print('\nsanity: 2024 week 18 status by team')
    s24 = df[(df.season == 2024) & (df.week == 18)].sort_values('stake')
    print('  dead :', ' '.join(sorted(s24[s24.stake == 0.0].team)))
    print('  in   :', ' '.join(sorted(s24[s24.stake == 0.5].team)))
    print('  live :', ' '.join(sorted(s24[s24.stake == 1.0].team)))
