"""
Does clinch status predict what the model gets wrong?

Same test that adopted EPA and rejected rest and travel: take the model
residual S - E, bucket by the candidate feature, and look for a monotone
gradient with significant tails. Team quality is already removed, because
that is what produced E.

The control here is the full adopted model -- Elo + MOV + QB + EPA -- since
clinch status would be added on top of it, not instead of it.

stake_diff = stake_home - stake_away takes five values on late games. A
positive value means the home team has more to play for. If resting
starters costs a team, the residual should climb with stake_diff.

The prior is that this works: 15 of the 20 largest disagreements with the
closing line are Week 16-18 games, and the market clearly prices rested
starters. But rest and travel also had a clear mechanism and a clean raw
gradient, and both were team quality in disguise. So the raw home-win rate
is reported alongside the residual, and only the residual counts.
"""
import numpy as np
import pandas as pd

from data import load_games
from elo import (run_elo, Accuracy, L, QB_SCALE, QB_ALPHA,
                 EPA_SCALE, EPA_ALPHA)
from qb_data import build_qb_map
from epa_test import team_game_epa
from clinch import build_stake_map, stake_diff, FIRST_WEEK

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50


def residual_table(g, key, label, min_n=30):
    agg = g.groupby(key, observed=True).agg(
        n=('resid', 'size'),
        raw=('home_win', 'mean'),
        resid=('resid', 'mean'),
        sd=('resid', 'std'))
    agg = agg[agg.n >= min_n]
    agg['se'] = agg.sd / np.sqrt(agg.n)
    agg['t'] = agg.resid / agg.se

    print(f'\n{label}')
    print(f'{"bucket":>14} {"n":>6} {"raw":>7} {"resid":>8} {"t":>6}')
    for idx, r in agg.iterrows():
        print(f'{str(idx):>14} {int(r.n):>6} {r.raw:>7.3f} '
              f'{r.resid:>+8.4f} {r.t:>+6.1f}')
    return agg


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()

    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                      epa_map=epa_map, epa_scale=EPA_SCALE,
                      epa_alpha=EPA_ALPHA)

    stake = build_stake_map(games)
    diff = stake_diff(games, stake)

    g = games.copy()
    g['resid'] = S - E
    g['home_win'] = (g.result > 0).astype(float)
    g['stake_diff'] = diff
    g['home_stake'] = [stake.get((r.game_id, r.home_team), np.nan)
                       for r in games.itertuples()]
    g['away_stake'] = [stake.get((r.game_id, r.away_team), np.nan)
                       for r in games.itertuples()]

    print(f'n = {len(g)}, mean residual {g.resid.mean():+.4f} (should be ~0)')

    late = g[g.home_stake.notna()]
    print(f'late REG games (week {FIRST_WEEK}+): {len(late)}')
    print(f'  of which stake_diff != 0: {(late.stake_diff != 0).sum()}')

    # --- the main test --------------------------------------------------
    residual_table(late, 'stake_diff',
                   'model residual by stake differential (home - away)')

    # --- one-sided views ------------------------------------------------
    # Does a dead team underperform regardless of who it plays?
    residual_table(late, 'home_stake',
                   'residual by HOME stake (higher resid = home beat model)')
    residual_table(late, 'away_stake',
                   'residual by AWAY stake (higher resid = home beat model)')

    # --- the sharpest cut ------------------------------------------------
    # A live team hosting or visiting a team with nothing to play for.
    sharp = late[(late.home_stake == 1.0) & (late.away_stake == 0.0)]
    sharp2 = late[(late.home_stake == 0.0) & (late.away_stake == 1.0)]
    for name, sub in (('live host vs dead visitor', sharp),
                      ('dead host vs live visitor', sharp2)):
        if len(sub) < 10:
            print(f'\n{name}: n={len(sub)}, too few')
            continue
        m, sd, n = sub.resid.mean(), sub.resid.std(), len(sub)
        print(f'\n{name}: n={n}  resid {m:+.4f}  t {m/(sd/np.sqrt(n)):+.1f}')

    # --- loss on the affected games only ---------------------------------
    lm = g.home_stake.notna().values & (g.stake_diff != 0)
    print(f'\non the {lm.sum()} games the feature would touch:')
    print(f'  model L {L(E[lm], S[lm]):.4f}  Acc {Accuracy(E[lm], S[lm]):.4f}')
    rest = ~lm
    print(f'  everywhere else ({rest.sum()}): L {L(E[rest], S[rest]):.4f}  '
          f'Acc {Accuracy(E[rest], S[rest]):.4f}')

    print('\nA monotone gradient with significant tails means build it.')
    print('Scatter, or a gradient in raw but not resid, means stop -- that')
    print('was travel, where the raw climb turned out to be team quality.')
