"""
Look up QB ids for STARTER_OVERRIDES.

predict.py picks each team's starter as the rostered QB with the most pass
attempts in the previous season's REG weeks 1-16. That handles trades and
signings, but not the case where a backup outthrew the starter because the
starter was hurt -- San Francisco with M.Jones over B.Purdy, Cincinnati with
J.Flacco over J.Burrow.

Enter the teams you need to correct, run, and paste the printed block into
predict.py. Each team's QBs are listed with attempt counts so it is obvious
who is who.
"""
import nflreadpy as nfl
from data import RELOCATED

# ---- edit this -------------------------------------------------------
TEAMS = ['SF', 'CIN','CLE','MIA','LV','MIN','WAS']

SEASON = 2026          # roster year
LAST_SEASON = 2025     # where attempts come from
# ----------------------------------------------------------------------


def roster_qbs(season=SEASON, last_season=LAST_SEASON):
    """Rostered QBs per team, with last season's attempts attached."""
    r = nfl.load_rosters([season]).to_pandas()
    if 'week' in r.columns and r.week.notna().any():
        r = r[r.week == r.week.max()]
    qbs = r[(r.position == 'QB') & (r.status == 'ACT')].copy()
    qbs['team'] = qbs.team.replace(RELOCATED)

    ps = nfl.load_player_stats([last_season]).to_pandas()
    prior = ps[(ps.position == 'QB') & (ps.attempts > 0)
               & (ps.season_type == 'REG') & (ps.week <= 16)]
    att = prior.groupby('player_id').attempts.sum()
    gms = prior.groupby('player_id').size()

    qbs['att'] = qbs.gsis_id.map(att).fillna(0).astype(int)
    qbs['gp'] = qbs.gsis_id.map(gms).fillna(0).astype(int)

    return qbs.sort_values(['team', 'att'], ascending=[True, False])


if __name__ == '__main__':
    qbs = roster_qbs()
    teams = TEAMS if TEAMS else sorted(qbs.team.unique())

    for team in teams:
        sub = qbs[qbs.team == team]
        if sub.empty:
            print(f'\n{team}: no active QBs found — check the team code')
            continue
        print(f'\n{team}')
        for i, r in enumerate(sub.itertuples()):
            mark = '  <- currently picked' if i == 0 else ''
            print(f'  {r.full_name:<22} {r.gsis_id:<14} '
                  f'{r.att:>4} att  {r.gp:>2} g{mark}')

    print('\n' + '=' * 62)
    print('Paste into predict.py; uncomment one line per team:\n')
    print('STARTER_OVERRIDES = {')
    for team in teams:
        for r in qbs[qbs.team == team].itertuples():
            print(f"    # '{team}': '{r.gsis_id}',".ljust(42)
                  + f'# {r.full_name} ({r.att} att)')
    print('}')