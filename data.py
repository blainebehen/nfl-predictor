#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:49:52 2026
@author: blainebehen
"""
import nflreadpy as nfl

# franchises that relocated — nflverse uses different codes before/after,
# which would otherwise split each team's history into two ratings
RELOCATED = {'STL': 'LA', 'SD': 'LAC', 'OAK': 'LV'}


def load_games():
    """Completed games, in chronological order, with relocations merged."""
    sched = nfl.load_schedules().to_pandas()
    games = sched[sched.result.notna()].copy()

    games['home_team'] = games.home_team.replace(RELOCATED)
    games['away_team'] = games.away_team.replace(RELOCATED)

    return games.sort_values(['season', 'week'])


if __name__ == '__main__':
    games = load_games()
    print(len(games), round((games.result > 0).mean(), 4))

    teams = sorted(set(games.home_team) | set(games.away_team))
    print(len(teams), 'teams:', ' '.join(teams))