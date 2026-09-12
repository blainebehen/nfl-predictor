"""
Forecast upcoming games from the current model state.

TWO MODELS ARE FORECAST SIDE BY SIDE.

  E_qb    Elo + MOV + the QB adjustment. This is what produced the Week 1
          2026 forecast, before team EPA was wired into the live path.
  E_full  theta* -- the same thing plus the team-EPA adjustment, which is
          adopted in RESULTS.md and validated across six windows.

Backtests say E_full is the better model by 0.0026 in held-out L. That is
a backtest talking. Running both forward and scoring both on the same
games is the only way to watch the claim survive contact with games that
did not exist when either was tuned, so both are written to the CSV every
week and score.py reports them against each other and against the line.

Expect the two to agree on most games -- the EPA differential correlates
+0.80 with the model's own probability. The weeks worth watching are the
ones where they part.

STARTER INFERENCE
-----------------
The QB adjustment needs to know who is STARTING, which for an unplayed
game is unknown -- qb_data only records who actually played. Starters are
inferred by intersecting two sources:

  1. the current season's roster (who is on the team NOW), which catches
     trades and free-agent signings that last season's stats cannot; and
  2. pass attempts in the previous season's REG weeks 1-16, to pick which
     of a team's rostered QBs is actually the starter.

Attempts are counted wherever the player threw them, not just for his
current team, so a quarterback who changed teams keeps his history.

Excluding weeks 17-18 from step 2 matters. The naive rule -- "whoever
started most recently" -- picks whoever took snaps in Week 18, which for
any team with its playoff seed locked is a third-stringer resting the
starter. That produced Kansas City fielding C.Oladokun and Denver
fielding J.Stidham. Same late-season contamination documented in
RESULTS.md, in a different place.

Remaining failure mode: a rookie or an unproven QB winning a camp battle
has no attempt history and will lose to a rostered veteran backup. Those
go in STARTER_OVERRIDES. Always eyeball the printed starter column.
"""
import sys
from collections import defaultdict

import pandas as pd
import nflreadpy as nfl

from data import load_games, RELOCATED
from elo import run_elo, QB_SCALE, QB_ALPHA, EPA_SCALE, EPA_ALPHA
from qb_data import build_qb_map, current_season
from epa_test import team_game_epa

K, H, RHO = 20, 50, 0.50

SEASON = current_season()

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


def current_state(last_season=None):
    """
    Everything needed to forecast: two sets of team ratings (one per
    model), QB ratings, team EPA ratings, presumed starters, and a
    player-id to name lookup -- all after every completed game.

    The QB and EPA ratings are replayed here rather than pulled out of
    run_elo: each is an EWMA over its own observations and the game
    order, independent of the Elo state, so it can be computed
    separately. Elo ratings cannot -- they depend on the adjustments in
    force -- which is why run_elo is called twice.
    """
    last_season = last_season or (SEASON - 1)

    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    # model 1: QB only -- the Week 1 2026 model
    _, _, R_qb = run_elo(games, k=K, H=H, rho=RHO, mov=True,
                         qb_map=qb_map, qb_scale=QB_SCALE,
                         qb_alpha=QB_ALPHA)

    # model 2: theta* -- QB plus team EPA
    _, _, R_full = run_elo(games, k=K, H=H, rho=RHO, mov=True,
                           qb_map=qb_map, qb_scale=QB_SCALE,
                           qb_alpha=QB_ALPHA, epa_map=epa_map,
                           epa_scale=EPA_SCALE, epa_alpha=EPA_ALPHA)

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

    # fall back to stats names for anyone not on a current roster
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
    P(home wins) under both models for each scheduled game.

    revert applies the offseason shrink toward 1500. It belongs on a
    forecast made before the season's first game and nowhere else -- the
    ratings handed back by run_elo have already been reverted at the
    start of any season whose games are in the data.
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

    # Revert only when no game of this season has been played yet.
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

    print(f'\nStarters come from the {SEASON} roster, ranked by {SEASON - 1} '
          f'REG wk1-16 attempts.\nRookies and in-week injuries will still be '
          'wrong -- fix those in\nSTARTER_OVERRIDES before committing.')

    # Save before kickoff. The commit timestamp is what makes a track
    # record credible; regenerating after the fact would not.
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.insert(0, 'week', week)
    df.insert(0, 'season', SEASON)
    df['theta'] = (f'k={K},H={H},rho={RHO},'
                   f'qb_scale={QB_SCALE},qb_alpha={QB_ALPHA},'
                   f'epa_scale={EPA_SCALE},epa_alpha={EPA_ALPHA}')

    path = f'predictions_{SEASON}.csv'
    try:
        old = pd.read_csv(path)
        clash = ((old.season == SEASON) & (old.week == week)).sum()
        if clash:
            print(f'\nWARNING: replacing {clash} existing rows for week '
                  f'{week}. Only legitimate before kickoff.')
        old = old[~((old.season == SEASON) & (old.week == week))]
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(path, index=False)
    print(f'\nwrote week {week} to {path} ({len(df)} rows total)')
