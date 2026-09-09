"""
Forecast upcoming games from the current model state.

The QB adjustment needs to know who is STARTING, which for an unplayed game
is unknown -- qb_data only records who actually played. Starters are inferred
by intersecting two sources:

  1. the current season's roster (who is on the team NOW), which catches
     trades and free-agent signings that last season's stats cannot; and
  2. pass attempts in the previous season's REG weeks 1-16, to pick which
     of a team's rostered QBs is actually the starter.

Attempts are counted wherever the player threw them, not just for his
current team, so a quarterback who changed teams keeps his history.

Excluding weeks 17-18 from step 2 matters. The naive rule -- "whoever started
most recently" -- picks whoever took snaps in Week 18, which for any team
with its playoff seed locked is a third-stringer resting the starter. That
produced Kansas City fielding C.Oladokun and Denver fielding J.Stidham. Same
late-season contamination documented in RESULTS.md, in a different place.

Remaining failure mode: a rookie or an unproven QB winning a camp battle has
no attempt history and will lose to a rostered veteran backup. Those go in
STARTER_OVERRIDES. Always eyeball the printed starter column.
"""
from collections import defaultdict

import pandas as pd
import nflreadpy as nfl

from data import load_games, RELOCATED
from elo import run_elo, QB_SCALE, QB_ALPHA
from qb_data import build_qb_map

K, H, RHO = 20, 50, 0.50

SEASON, WEEK = 2026, 1

# Manual starter overrides: (team) -> player_id. Use when the assumed
# starter is known to be wrong -- a trade, a retirement, an injury.
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
}


def infer_starters(season, last_season):
    """
    Each team's presumed starter for the upcoming season.

    Roster tells us who is on the team; last season's attempts tell us
    which of them starts. See the module docstring for why weeks 17-18
    are excluded and why attempts are counted across all teams.
    """
    # who is on each roster now
    r = nfl.load_rosters([season]).to_pandas()
    if 'week' in r.columns and r.week.notna().any():
        r = r[r.week == r.week.max()]           # latest weekly snapshot
    qbs = r[(r.position == 'QB') & (r.status == 'ACT')].copy()
    qbs['team'] = qbs.team.replace(RELOCATED)

    # how much each of them threw last season, wherever he threw it
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


def current_state(last_season=2025):
    """
    Team ratings, QB ratings, presumed starters, and a player-id to name
    lookup -- all after every completed game.

    QB ratings are replayed here rather than pulled out of run_elo: the
    EWMA depends only on the QB values and the game order, not on the Elo
    state, so it can be computed independently.
    """
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)

    _, _, R = run_elo(games, k=K, H=H, rho=RHO, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

    Q = defaultdict(float)
    for g in games.itertuples():
        for team in (g.home_team, g.away_team):
            entry = qb_map.get((g.game_id, team))
            if entry is None:
                continue
            pid, val = entry
            Q[pid] += QB_ALPHA * (val - Q[pid])

    starters, names = infer_starters(SEASON, last_season)

    # fall back to stats names for anyone not on a current roster
    stat_names = (nfl.load_player_stats([last_season]).to_pandas()
                    .drop_duplicates('player_id')
                    .set_index('player_id').player_name.to_dict())
    names = {**stat_names, **names}

    return R, dict(Q), starters, names


def predict_week(R, Q, starters, names, season, week):
    """P(home wins) for each scheduled game, with the new season's reversion."""
    sched = nfl.load_schedules().to_pandas()
    upcoming = sched[(sched.season == season) & (sched.week == week)]

    # new season: shrink every rating toward 1500 before predicting
    R = {t: 1500 + (1 - RHO) * (r - 1500) for t, r in R.items()}

    rows = []
    for g in upcoming.itertuples():
        qb_h = STARTER_OVERRIDES.get(g.home_team, starters.get(g.home_team))
        qb_a = STARTER_OVERRIDES.get(g.away_team, starters.get(g.away_team))

        adj = QB_SCALE * (Q.get(qb_h, 0.0) - Q.get(qb_a, 0.0))
        d = R.get(g.home_team, 1500) + H - R.get(g.away_team, 1500) + adj
        E = 1 / (1 + 10 ** (-d / 400))

        rows.append((g.away_team, g.home_team, E, g.spread_line, adj,
                     names.get(qb_a, '?'), names.get(qb_h, '?')))
    return rows


if __name__ == '__main__':
    R, Q, starters, names = current_state()
    rows = predict_week(R, Q, starters, names, SEASON, WEEK)

    print(f'\n{SEASON} week {WEEK}\n')
    print(f'{"away":>4} {"home":>4} {"P(home)":>8} {"spread":>7} {"qb adj":>7}  '
          f'{"assumed starters (away / home)":<40}')
    for away, home, E, spread, adj, qa, qh in rows:
        print(f'{away:>4} {home:>4} {E:8.3f} {spread:7.1f} {adj:+7.0f}  '
              f'{qa} / {qh}')

    print('\nStarters come from the 2026 roster, ranked by 2025 REG wk1-16 '
          'attempts.\nRookies and in-week injuries will still be wrong -- '
          'fix those in\nSTARTER_OVERRIDES before committing.')

    # Save before kickoff. The commit timestamp is what makes a track
    # record credible; regenerating after the fact would not.
    df = pd.DataFrame(rows, columns=['away', 'home', 'E', 'spread',
                                     'qb_adj', 'away_qb', 'home_qb'])
    df.insert(0, 'week', WEEK)
    df.insert(0, 'season', SEASON)
    df['theta'] = (f'k={K},H={H},rho={RHO},'
                   f'qb_scale={QB_SCALE},qb_alpha={QB_ALPHA}')

    path = f'predictions_{SEASON}.csv'
    try:
        old = pd.read_csv(path)
        old = old[~((old.season == SEASON) & (old.week == WEEK))]
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(path, index=False)
    print(f'\nwrote week {WEEK} to {path} ({len(df)} rows total)')