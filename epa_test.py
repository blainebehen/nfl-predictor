"""
Does rolling team EPA predict what the model gets wrong?

Elo updates on who won, scaled by margin of victory. EPA per play measures
how a team actually moved the ball, which is less noisy than the scoreboard
-- a team that gains 6.5 yards a play and loses on a late turnover played
better than the result says.

The open question is whether that is NEW information. Scoring margin and
EPA are highly correlated, and Elo already sees margin through the MOV
multiplier, so Elo may have extracted most of what EPA contains through a
noisier channel. If so the residual gradient will be flat and there is
nothing to build.

Construction notes:
  - offense EPA = passing_epa + rushing_epa. receiving_epa is the same
    plays counted from the receiver side and would double-count.
  - defence EPA is not shipped; it is the opponent's offensive EPA in the
    same game, recovered by a self-join on game_id.
  - EPA is a total, not a rate, so it is divided by plays (attempts +
    carries + sacks) -- a team running 75 plays accumulates more than one
    running 55.
  - ratings are EWMAs read BEFORE each game and updated after, so no game
    informs its own prediction.

RESULT: EPA passes. Residual by octile of the home-away differential --

    -0.046  -0.062  -0.009  -0.006  -0.014  -0.001  +0.055  +0.053
     t=-2.9  t=-3.9                                   t=+3.7  t=+4.0

Negative at the bottom, flat in the middle, positive at the top, with both
tails significant and pointing the right way: when a team has been better
by EPA than Elo credits, it beats the model's prediction.

Contrast with travel, where a single cell cleared 2.8 SE surrounded by
noise and the extreme bucket reversed. Here both extremes are strong and
the sign flips where it should.

epa_diff correlates +0.796 with the model probability E, so EPA and the Elo
rating largely agree. The residual gradient says the 20% where they
disagree is where Elo is wrong -- which is what new information looks like.

Caveat carried forward: the middle four buckets are all slightly negative
(-0.001 to -0.014), so the effect may be concentrated in mismatches rather
than spread evenly. Worth watching whether a linear term captures it.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl

from data import load_games, RELOCATED
from elo import run_elo, QB_SCALE, QB_ALPHA
from qb_data import build_qb_map, current_season

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50
SEASONS = list(range(1999, current_season() + 1))
ALPHA = 0.15          # EWMA rate for team EPA


def team_game_epa(seasons=SEASONS):
    """(game_id, team) -> (offence EPA/play, defence EPA/play)."""
    ts = nfl.load_team_stats(seasons=seasons).to_pandas()
    ts = ts[ts.season_type == 'REG'].copy()
    ts['team'] = ts.team.replace(RELOCATED)

    plays = (ts.attempts.fillna(0) + ts.carries.fillna(0)
             + ts.sacks_suffered.fillna(0))
    epa = ts.passing_epa.fillna(0) + ts.rushing_epa.fillna(0)
    ts['off_epa'] = np.where(plays > 0, epa / plays.replace(0, np.nan), 0.0)

    # defence = what the opponent's offence did in the same game
    opp = ts[['game_id', 'team', 'off_epa']].rename(
        columns={'team': 'opponent_team', 'off_epa': 'def_epa'})
    ts = ts.merge(opp, on=['game_id', 'opponent_team'], how='left')

    return {(r.game_id, r.team): (r.off_epa, r.def_epa)
            for r in ts.itertuples() if pd.notna(r.def_epa)}


def rolling_epa_diff(games, epa_map, alpha=ALPHA):
    """
    Per game: the home team's net-EPA rating minus the away team's, using
    only games played before it.

    net = offence EPA/play - defence EPA/play allowed. Higher is better.
    """
    off = {}
    deff = {}
    out = []

    for g in games.itertuples():
        def rating(team):
            return off.get(team, 0.0) - deff.get(team, 0.0)

        out.append(rating(g.home_team) - rating(g.away_team))

        for team in (g.home_team, g.away_team):
            entry = epa_map.get((g.game_id, team))
            if entry is None:
                continue
            o, d = entry
            off[team] = off.get(team, o) + alpha * (o - off.get(team, o))
            deff[team] = deff.get(team, d) + alpha * (d - deff.get(team, d))

    return np.array(out)


if __name__ == '__main__':
    games = load_games()

    qb_map = build_qb_map(games, verbose=False)
    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

    epa_map = team_game_epa()
    print(f'EPA coverage: {len(epa_map)} team-games')

    diff = rolling_epa_diff(games, epa_map)

    g = games.copy()
    g['resid'] = S - E
    g['epa_diff'] = diff
    g['home_win'] = (g.result > 0).astype(float)

    # drop the first season, where every rating is still near its default
    g = g[g.season >= 2000]

    print(f'epa_diff: mean {g.epa_diff.mean():+.4f}  sd {g.epa_diff.std():.4f}')
    print(f'correlation with model probability E: '
          f'{np.corrcoef(g.epa_diff, E[games.season >= 2000])[0,1]:+.3f}')

    q = pd.qcut(g.epa_diff, 8)
    agg = g.groupby(q, observed=True).agg(
        n=('resid', 'size'), raw=('home_win', 'mean'),
        resid=('resid', 'mean'), sd=('resid', 'std'))
    agg['se'] = agg.sd / np.sqrt(agg.n)
    agg['t'] = agg.resid / agg.se

    print(f'\nmodel residual by rolling EPA differential (home - away)')
    print(f'{"bucket":>20} {"n":>6} {"raw":>7} {"resid":>8} {"t":>6}')
    for idx, r in agg.iterrows():
        print(f'{str(idx):>20} {int(r.n):>6} {r.raw:>7.3f} '
              f'{r.resid:>+8.4f} {r.t:>+6.1f}')

    print('\nA monotone gradient here means EPA holds information Elo lacks.')
    print('A flat column means margin of victory already carried it, and')
    print('this stops here.')
