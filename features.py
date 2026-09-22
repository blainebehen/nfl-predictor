"""
The two features the model uses, both keyed by (game_id, team):

    build_qb_map    who started at QB, and his EPA per dropback that game
    team_game_epa   the team's offensive and defensive EPA per play

These are what happened IN each game. elo.py only reads them after the
game is predicted, so nothing leaks into its own prediction.
"""
from datetime import date

import numpy as np
import pandas as pd
import nflreadpy as nfl

from data import RELOCATED


def current_season(today=None):
    """The season in progress. Seasons start in September."""
    today = today or date.today()
    return today.year if today.month >= 9 else today.year - 1


SEASONS = list(range(1999, current_season() + 1))


# ---------------------------------------------------------------- QB

def qb_game_values(seasons=SEASONS):
    """
    One row per (game, team): the starting QB and his EPA per dropback.

    Starter = the QB with the most pass attempts that game.
    EPA includes passing AND rushing, so running QBs get credit.
    Dropbacks include sacks, since a sack is a failed dropback.
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
    dict: (game_id, team) -> (player_id, EPA per dropback)

    Covers about 99.7% of team-games. The rest (no QB threw a pass) just
    get no QB adjustment.
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
    dict: (game_id, team) -> (offense EPA per play, defense EPA per play)

    Offense = passing + rushing EPA divided by plays.
    Defense = the opponent's offensive EPA in the same game.
    Regular season only.
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


if __name__ == '__main__':
    from data import load_games

    games = load_games()
    build_qb_map(games)
    print(f'EPA coverage: {len(team_game_epa())} team-games')
