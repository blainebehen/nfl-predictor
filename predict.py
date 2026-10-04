"""
Predict this week's games.

    python predict.py 4        (predicts week 4)

I run two versions side by side and save both, so I can see which one
actually does better on real games:

  E       Elo + QB. This is the model I started the 2026 season with.
  E_full  Elo + QB + team EPA. Almost identical in the backtest, so the
          live season is the tiebreaker.

Starting Week 3 of 2026, both use raw QB ratings everywhere (the theta
column says qb=raw). Earlier weeks were saved with the older version.

Picking the starting QB: for games that haven't happened yet, I don't know
who will start. So for each team I take the QBs on the current roster and
pick the one who threw the most passes last season (weeks 1-16 only, since
teams that have already locked in a playoff spot rest starters in weeks
17-18). That misses rookies and injuries, so STARTER_OVERRIDES below lets
me set a starter by hand. Always check the starters column in the output.
"""
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import nflreadpy as nfl

from data import load_games, RELOCATED
from elo import run_elo, QB_SCALE, QB_ALPHA, EPA_SCALE, EPA_ALPHA
from features import build_qb_map, team_game_epa, current_season

K, H, RHO = 20, 50, 0.50

SEASON = current_season()

# Manual starter picks: team -> player_id. Use find_qb.py to look up ids.
STARTER_OVERRIDES = {
    # 'SF': '00-0036972',                 # Mac Jones (289 att)
    'SF': '00-0037834',                 # Brock Purdy (224 att)
    # 'SF': '00-0040589',                 # Kurtis Rourke (0 att)
    # 'CIN': '00-0026158',                # Joe Flacco (411 att)
    'CIN': '00-0036442',                # Joe Burrow (189 att)
    # 'CLE': '00-0040668',                # Shedeur Sanders (167 att)
    'CLE': '00-0033537',                # Deshaun Watson (0 att)
    # 'CLE': '00-0041092',                # Taylen Green (0 att)
    # 'MIA': '00-0040398',                # Brady Cook (98 att)
    'MIA': '00-0038128',                # Malik Willis (14 att)
    # 'MIA': '00-0040014',                # Kyle McCord (0 att)
    # 'LV': '00-0029604',                 # Kirk Cousins (217 att)
    # 'LV': '00-0038579',                 # Aidan O'Connell (0 att)
    # 'LV': '00-0041562',                 # Fernando Mendoza (0 att)
    # 'MIN': '00-0039923',                # J.J. McCarthy (220 att)
    # 'MIN': '00-0032950',                # Carson Wentz (169 att)
    'MIN': '00-0035228',                # Kyler Murray (161 att)
    # 'WAS': '00-0032268',                # Marcus Mariota (227 att)
    'WAS': '00-0039910',                # Jayden Daniels (188 att)
    # 'WAS': '00-0041117',                # Athan Kaliakmanis (0 att)
    'ATL': '00-0039917',                # Michael Penix Jr. (named Week 3 starter)
    # 'ATL': '00-0036212',                # Tua Tagovailoa
    # 'ATL': '00-0033662',                # Cooper Rush
    'NYG': '00-0031503',                # Jameis Winston (Dart knee injury)
    # 'NYG': '00-0040691',                # Jaxson Dart
    # 'SEA': '00-0035704',                # Drew Lock (if Darnold is out)
}


def infer_starters(season, last_season, week=1):
    """
    Each team's likely starter, ranked on the most relevant attempts
    available.

    In week 1 the only signal is last season, weeks 1-16 (17-18 excluded
    because a team with its seed locked rests its starter, which once put
    C.Oladokun under centre for Kansas City).

    From week 2 on the signal is RECENCY, not the season total: whoever
    threw most in his team's most recent game. Cumulative attempts look
    like the sturdier statistic and are wrong mid-season, because a
    quarterback who took over in week 3 has not yet outthrown the man he
    replaced. Checked against 2026 week 4, the cumulative rule picked
    Rush over Penix, Williams over Keenum, Wentz over Murray and Lock
    over Darnold -- wrong on all four, and each one needed a manual
    override. Recency gets all four right.

    The inverse error is a one-week fill-in becoming the pick after the
    starter returns, so a team whose recent starter trails the season
    leader by a wide margin is flagged rather than silently trusted.

    Falls back to last season per team, not globally -- a team whose QBs
    have no attempts yet this season still gets a sensible guess.
    """
    # QBs on each roster right now
    r = nfl.load_rosters([season]).to_pandas()
    if 'week' in r.columns and r.week.notna().any():
        r = r[r.week == r.week.max()]           # latest weekly snapshot
    qbs = r[(r.position == 'QB') & (r.status == 'ACT')].copy()
    qbs['team'] = qbs.team.replace(RELOCATED)

    def attempts(seasons, max_week=None):
        ps = nfl.load_player_stats(seasons).to_pandas()
        m = ((ps.position == 'QB') & (ps.attempts > 0)
             & (ps.season_type == 'REG'))
        if max_week is not None:
            m &= ps.week <= max_week
        return ps[m]

    prior = attempts([last_season], max_week=16)
    last_att = prior.groupby('player_id').attempts.sum()

    cur = attempts([season], max_week=week - 1) if week > 1 else None
    cur_att = (cur.groupby('player_id').attempts.sum()
               if cur is not None and len(cur) else None)

    # most recent week each QB threw, and how much he threw in it
    if cur is not None and len(cur):
        recent = (cur.sort_values(['week', 'attempts'])
                     .groupby('player_id').last()[['week', 'attempts']]
                     .rename(columns={'week': 'last_wk',
                                      'attempts': 'last_wk_att'}))
    else:
        recent = None

    qbs['last_att'] = qbs.gsis_id.map(last_att).fillna(0)
    qbs['cur_att'] = (qbs.gsis_id.map(cur_att).fillna(0)
                      if cur_att is not None else 0.0)
    qbs['last_wk'] = (qbs.gsis_id.map(recent.last_wk).fillna(0)
                      if recent is not None else 0.0)
    qbs['last_wk_att'] = (qbs.gsis_id.map(recent.last_wk_att).fillna(0)
                          if recent is not None else 0.0)

    starters, notes = {}, {}
    names = dict(zip(qbs.gsis_id, qbs.full_name))

    for team, grp in qbs.groupby('team'):
        if grp.cur_att.sum() > 0:
            # most recent game first, then who threw most in it
            ranked = grp.sort_values(['last_wk', 'last_wk_att'],
                                     ascending=False)
            pick = ranked.iloc[0]
            starters[team] = pick.gsis_id

            leader = grp.sort_values('cur_att', ascending=False).iloc[0]
            if (leader.gsis_id != pick.gsis_id
                    and leader.cur_att > 2 * pick.cur_att):
                notes[team] = (
                    f'started wk {int(pick.last_wk)} but has only '
                    f'{int(pick.cur_att)} attempts this season against '
                    f'{names.get(leader.gsis_id, "?")}\'s '
                    f'{int(leader.cur_att)} — possible fill-in')
        else:
            ranked = grp.sort_values('last_att', ascending=False)
            pick = ranked.iloc[0]
            starters[team] = pick.gsis_id
            if pick.last_att == 0:
                notes[team] = 'no attempts anywhere — guess'

    if notes:
        print('Eyeball these — override in STARTER_OVERRIDES if wrong:')
        for t, why in sorted(notes.items()):
            print(f'  {t}: {names.get(starters[t], starters[t])} — {why}')

    return starters, names


def current_state(last_season=None, week=1):
    """
    Replay every completed game to get where things stand now: team
    ratings for both models, QB ratings, team EPA, likely starters,
    and player names.
    """
    last_season = last_season or (SEASON - 1)

    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    # model 1: Elo + QB
    _, _, R_qb = run_elo(games, k=K, H=H, rho=RHO, mov=True,
                         qb_map=qb_map, qb_scale=QB_SCALE,
                         qb_alpha=QB_ALPHA)

    # model 2: Elo + QB + team EPA
    _, _, R_full = run_elo(games, k=K, H=H, rho=RHO, mov=True,
                           qb_map=qb_map, qb_scale=QB_SCALE,
                           qb_alpha=QB_ALPHA, epa_map=epa_map,
                           epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA)

    # current QB ratings and team EPA, same updates as in run_elo
    Q = defaultdict(float)
    OFF, DEF = {}, {}
    for g in games.itertuples():
        for team in (g.home_team, g.away_team):
            entry = qb_map.get((g.game_id, team))
            if entry is not None:
                pid, val = entry
                Q[pid] += QB_ALPHA * (val - Q[pid])

            entry = epa_map.get((g.game_id, team))
            if entry is not None:
                o, d = entry
                OFF[team] = OFF.get(team, o) + EPA_ALPHA * (o - OFF.get(team, o))
                DEF[team] = DEF.get(team, d) + EPA_ALPHA * (d - DEF.get(team, d))

    starters, names = infer_starters(SEASON, last_season, week=week)

    # names for anyone not on a current roster
    stat_names = (nfl.load_player_stats([last_season]).to_pandas()
                    .drop_duplicates('player_id')
                    .set_index('player_id').player_name.to_dict())
    names = {**stat_names, **names}

    net_epa = {t: OFF.get(t, 0.0) - DEF.get(t, 0.0)
               for t in set(OFF) | set(DEF)}

    return R_qb, R_full, dict(Q), net_epa, starters, names


ET = ZoneInfo('America/New_York')


def _kickoff(sched):
    """
    Scheduled kickoff as a timezone-aware instant, one per row.

    nflverse gives gameday as a date and gametime as HH:MM Eastern.
    A missing gametime becomes NaT, which compares false against the
    clock, so such a game is only caught by its result -- the old
    behaviour, and the safe direction is to keep forecasting it rather
    than silently drop it.
    """
    stamp = sched.gameday.astype(str) + ' ' + sched.gametime.astype(str)
    return pd.to_datetime(stamp, errors='coerce').dt.tz_localize(
        ET, ambiguous='NaT', nonexistent='NaT')


def predict_week(R_qb, R_full, Q, net_epa, starters, names, season, week,
                 revert=True):
    """
    Chance the home team wins, under both models, for each game this week.

    revert: shrink ratings toward 1500 first. Only for a preseason
    forecast, before any game of the new season has been played.
    """
    sched = nfl.load_schedules().to_pandas()
    upcoming = sched[(sched.season == season) & (sched.week == week)]

    # Never forecast a game that has already kicked off. A week picked up
    # late -- after its Thursday game, say -- gets forecasts for the games
    # still ahead and simply has no row for the one that is gone. Writing
    # one now would be a prediction made after the fact, which is the one
    # thing this file exists to avoid.
    #
    # The gate is SCHEDULED KICKOFF against the clock, not whether a
    # result has appeared. Results lag: on 4 Oct 2026 the London game
    # kicked off 06:30 PT and had finished, but nflverse still showed no
    # result, so a result-based gate let it through and it had to be
    # removed from the record afterwards. Kickoff time does not lag.
    upcoming = upcoming.assign(kick=_kickoff(upcoming))
    now = datetime.now(ET)

    started = (upcoming.kick <= now) | upcoming.result.notna()
    if started.any():
        for g in upcoming[started].itertuples():
            when = ('result already in' if pd.notna(g.result)
                    else f'kicked off {g.kick.strftime("%a %H:%M %Z")}')
            print(f'  skipping {g.away_team} at {g.home_team} — {when}')
        print(f'{int(started.sum())} of {len(upcoming)} week-{week} games '
              f'already under way — forecasting the '
              f'{int((~started).sum())} still ahead.')
        upcoming = upcoming[~started]

    # A forecast committed minutes before kickoff is technically clean and
    # proves nothing to anyone reading the git log later. Say so now,
    # while there is still time to care.
    soon = upcoming[upcoming.kick <= now + timedelta(hours=1)]
    if len(soon):
        first = soon.kick.min()
        print(f'\nWARNING: {len(soon)} of these kick off within the hour '
              f'(first at {first.strftime("%a %H:%M %Z")}).')
        print('Commit now and the timestamp barely precedes kickoff. Run')
        print('this Tuesday or Wednesday instead.\n')

    if revert:
        R_qb = {t: 1500 + (1 - RHO) * (r - 1500) for t, r in R_qb.items()}
        R_full = {t: 1500 + (1 - RHO) * (r - 1500) for t, r in R_full.items()}

    rows = []
    for g in upcoming.itertuples():
        qb_h = STARTER_OVERRIDES.get(g.home_team, starters.get(g.home_team))
        qb_a = STARTER_OVERRIDES.get(g.away_team, starters.get(g.away_team))

        qb_adj = QB_SCALE * (Q.get(qb_h, 0.0) - Q.get(qb_a, 0.0))
        epa_adj = EPA_SCALE * (net_epa.get(g.home_team, 0.0)
                               - net_epa.get(g.away_team, 0.0))

        d_qb = (R_qb.get(g.home_team, 1500) + H
                - R_qb.get(g.away_team, 1500) + qb_adj)
        d_full = (R_full.get(g.home_team, 1500) + H
                  - R_full.get(g.away_team, 1500) + qb_adj + epa_adj)

        rows.append((g.away_team, g.home_team,
                     1 / (1 + 10 ** (-d_qb / 400)),
                     1 / (1 + 10 ** (-d_full / 400)),
                     g.spread_line, qb_adj, epa_adj,
                     names.get(qb_a, '?'), names.get(qb_h, '?')))
    return rows


COLUMNS = ['away', 'home', 'E', 'E_full', 'spread', 'qb_adj', 'epa_adj',
           'away_qb', 'home_qb']


if __name__ == '__main__':
    week = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    R_qb, R_full, Q, net_epa, starters, names = current_state(week=week)

    # shrink toward 1500 only if no game this season has been played yet
    played = load_games()
    revert = not (played.season == SEASON).any()

    rows = predict_week(R_qb, R_full, Q, net_epa, starters, names,
                        SEASON, week, revert=revert)

    print(f'\n{SEASON} week {week}'
          f'{"  (preseason reversion applied)" if revert else ""}\n')
    print(f'{"away":>4} {"home":>4} {"E_qb":>7} {"E_full":>7} {"diff":>6} '
          f'{"spread":>7} {"qb":>6} {"epa":>6}  '
          f'{"assumed starters (away / home)":<40}')
    for away, home, e_qb, e_full, spread, qadj, eadj, qa, qh in rows:
        sp = f'{spread:7.1f}' if pd.notna(spread) else f'{"—":>7}'
        print(f'{away:>4} {home:>4} {e_qb:7.3f} {e_full:7.3f} '
              f'{e_full - e_qb:+6.3f} {sp} {qadj:+6.0f} {eadj:+6.0f}  '
              f'{qa} / {qh}')

    disagree = [r for r in rows if (r[2] - 0.5) * (r[3] - 0.5) < 0]
    print(f'\n{len(disagree)} of {len(rows)} games where the two models pick '
          f'different sides'
          + (':' if disagree else ''))
    for away, home, e_qb, e_full, *_ in disagree:
        print(f'  {away} at {home}: qb {e_qb:.3f} -> full {e_full:.3f}')

    print('\nCheck the starters above. Fix any wrong ones in '
          'STARTER_OVERRIDES and rerun.')

    # Save before kickoff, and commit, so the timestamp proves the
    # prediction came first.
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.insert(0, 'week', week)
    df.insert(0, 'season', SEASON)
    df['theta'] = (f'k={K},H={H},rho={RHO},qb=raw,'
                   f'qb_scale={QB_SCALE},qb_alpha={QB_ALPHA},'
                   f'epa_scale={EPA_SCALE},epa_alpha={EPA_ALPHA}')

    path = f'predictions_{SEASON}.csv'
    try:
        old = pd.read_csv(path)
        clash = ((old.season == SEASON) & (old.week == week)).sum()
        if clash:
            print(f'\nWARNING: replacing {clash} existing rows for week '
                  f'{week}. Only OK before kickoff.')
        old = old[~((old.season == SEASON) & (old.week == week))]
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(path, index=False)
    print(f'\nwrote week {week} to {path} ({len(df)} rows total)')
