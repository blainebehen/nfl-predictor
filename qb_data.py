"""
Quarterback game values from nflverse player stats.

Extraction only -- no model code, so elo.py can import this without a
circular dependency on qb.py (which imports elo.py for tuning).

Team Elo already absorbs a franchise's average QB quality. What it cannot
see is who is actually under center on a given Sunday: the starter is hurt,
benched, or a rookie is in. That blind spot is a large part of the model's
remaining gap to the closing line.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl

from data import RELOCATED

def current_season(today=None):
    """
    The NFL season currently in progress, or the most recent one.

    Seasons are named for the calendar year they start in, and start in
    September, so anything before September belongs to the previous year.
    Hardcoding an end year meant the model silently stopped learning as
    soon as a new season began -- games appeared in the schedule but
    carried no QB or EPA value.
    """
    from datetime import date
    today = today or date.today()
    return today.year if today.month >= 9 else today.year - 1


SEASONS = list(range(1999, current_season() + 1))


def qb_game_values(seasons=SEASONS):
    """
    One row per (game_id, team): the starting QB and his EPA per dropback.

    Starter = most pass attempts. Checked on 2024: in team-games with two
    QBs the second threw a median 4% of attempts, so this picks the real
    starter almost always.

    Value combines passing and rushing EPA -- ignoring the run would
    systematically underrate mobile quarterbacks. Dropbacks include sacks,
    since a sack is a failed dropback and belongs in the denominator.

    Sanity check on the output: the career leaders are Mahomes, Jackson,
    Manning, Allen, Rodgers, Brees, Brady; the trailers are Harrington,
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
    qb['value'] = np.where(dropbacks > 0, epa / dropbacks.replace(0, np.nan), 0.0)

    qb['team'] = qb.team.replace(RELOCATED)

    return qb[['game_id', 'team', 'player_id', 'player_name', 'value']]


def build_qb_map(games, seasons=SEASONS, verbose=True):
    """
    dict: (game_id, team) -> (player_id, value)

    Coverage is 99.7% of team-games; the rest are games where no QB
    recorded an attempt. Those contribute no adjustment rather than
    being dropped. A much lower rate here would mean team codes diverge
    between player_stats and schedules -- worth seeing rather than
    silently losing games.
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


if __name__ == '__main__':
    from data import load_games

    games = load_games()
    qb_map = build_qb_map(games)

    qb = qb_game_values()
    print(f'\n{len(qb)} QB starts, {qb.player_id.nunique()} distinct QBs')
    print(f'value (EPA/dropback): mean {qb.value.mean():.3f}  '
          f'sd {qb.value.std():.3f}  '
          f'p10 {qb.value.quantile(.10):.3f}  '
          f'p90 {qb.value.quantile(.90):.3f}')

    names = qb.groupby('player_id').player_name.first()
    best = (qb.groupby('player_id').value.agg(['size', 'mean'])
              .query('size >= 60').sort_values('mean', ascending=False))
    print('\nbest 15 by career EPA/dropback (min 60 starts):')
    for pid, row in best.head(15).iterrows():
        print(f'  {names[pid]:<22} {int(row["size"]):>4} starts  {row["mean"]:+.3f}')
    print('\nworst 5:')
    for pid, row in best.tail(5).iterrows():
        print(f'  {names[pid]:<22} {int(row["size"]):>4} starts  {row["mean"]:+.3f}')
