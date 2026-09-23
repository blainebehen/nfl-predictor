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


def infer_starters(season, last_season):
    """Each team's likely starter: rostered QB with the most attempts last season."""
    # QBs on each roster right now
    r = nfl.load_rosters([season]).to_pandas()
    if 'week' in r.columns and r.week.notna().any():
        r = r[r.week == r.week.max()]           # latest weekly snapshot
    qbs = r[(r.position == 'QB') & (r.status == 'ACT')].copy()
    qbs['team'] = qbs.team.replace(RELOCATED)

    # their pass attempts last season (for any team), weeks 1-16
    ps = nfl.load_player_stats([last_season]).to_pandas()
    prior = ps[(ps.position == 'QB') & (ps.attempts > 0)
               & (ps.season_type == 'REG') & (ps.week <= 16)]
    att = prior.groupby('player_id').attempts.sum()

    qbs['prior_att'] = qbs.gsis_id.map(att).fillna(0)

    top = (qbs.sort_values('prior_att', ascending=False)
              .groupby('team', as_index=False).first())

    starters = dict(zip(top.team, top.gsis_id))
    names = dict(zip(qbs.gsis_id, qbs.full_name))
    unproven = {t: names.get(p, p) for t, p in starters.items()
                if top.set_index('team').prior_att.get(t, 0) == 0}

    if unproven:
        print('No prior attempts, so these are guesses — override if wrong:')
        for t, n in sorted(unproven.items()):
            print(f'  {t}: {n}')

    return starters, names


def current_state(last_season=None):
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

    starters, names = infer_starters(SEASON, last_season)

    # names for anyone not on a current roster
    stat_names = (nfl.load_player_stats([last_season]).to_pandas()
                    .drop_duplicates('player_id')
                    .set_index('player_id').player_name.to_dict())
    names = {**stat_names, **names}

    net_epa = {t: OFF.get(t, 0.0) - DEF.get(t, 0.0)
               for t in set(OFF) | set(DEF)}

    return R_qb, R_full, dict(Q), net_epa, starters, names


def predict_week(R_qb, R_full, Q, net_epa, starters, names, season, week,
                 revert=True):
    """
    Chance the home team wins, under both models, for each game this week.

    revert: shrink ratings toward 1500 first. Only for a preseason
    forecast, before any game of the new season has been played.
    """
    sched = nfl.load_schedules().to_pandas()
    upcoming = sched[(sched.season == season) & (sched.week == week)]

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

    R_qb, R_full, Q, net_epa, starters, names = current_state()

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
