"""
Against-the-spread performance: could this model actually beat the market?

Win probability and spread betting are different markets, and being good at
one says little about the other. The model picks winners at ~64% straight
up, but favourites win most games -- the moneyline price already reflects
that, so picking them correctly earns nothing.

The spread is set so each side is close to a coin flip. That makes it the
market where a real edge shows up as a pick rate above chance. At standard
-110 juice you risk 110 to win 100, so break-even is

    110 / 210 = 52.38%

Not 50%. The 2.38 points between them is the house's cut, and it is the
entire obstacle.

Method: invert the model's win probability into an implied spread, compare
to the posted line, and bet the side the model prefers. A game is a push
when the margin lands exactly on the line -- pushes are refunded, so they
are excluded rather than scored as half-wins.
"""
import numpy as np
from scipy.stats import norm

from data import load_games
from elo import run_elo, season_mask, Accuracy, L, QB_SCALE, QB_ALPHA
from qb import K_STAR, H_STAR, RHO_STAR
from qb_data import build_qb_map

SPREAD_SD = 13.5      # SD of (actual margin - closing spread)
BREAK_EVEN = 110 / 210


def ats_record(E, games, edge_min=0.0):
    """
    Returns (n_bets, win_rate, roi) for betting every game where the
    model's implied spread differs from the posted line by at least
    `edge_min` points.

    edge_min filters to games where the model disagrees most. If an edge
    exists anywhere it should be strongest there -- and if the win rate
    does NOT rise as the filter tightens, that is evidence there is no
    edge, only noise.
    """
    spread = games.spread_line.values
    result = games.result.values

    implied = SPREAD_SD * norm.ppf(np.clip(E, 1e-6, 1 - 1e-6))
    edge = implied - spread

    live = (~np.isnan(spread) & ~np.isnan(result)
            & (result != spread)            # drop pushes
            & (np.abs(edge) >= edge_min))

    if live.sum() == 0:
        return 0, np.nan, np.nan

    pick_home = edge[live] > 0
    home_covered = result[live] > spread[live]
    wins = (pick_home == home_covered)

    n = len(wins)
    rate = wins.mean()
    # -110: win 100, lose 110
    roi = (wins.sum() * 100 - (n - wins.sum()) * 110) / (n * 110)
    return n, rate, roi


if __name__ == '__main__':
    games = load_games()
    qb_map = build_qb_map(games, verbose=False)

    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

    print(f'break-even at -110: {BREAK_EVEN:.2%}\n')

    print(f'{"min edge":>9} {"bets":>6} {"win rate":>9} {"vs BE":>8} '
          f'{"ROI":>8} {"SE":>7}')
    for edge_min in [0, 1, 2, 3, 4, 6, 8]:
        n, rate, roi = ats_record(E, games, edge_min)
        if n == 0:
            continue
        se = np.sqrt(0.25 / n)          # SE of a proportion near 0.5
        print(f'{edge_min:>9} {n:>6} {rate:>9.4f} {rate - BREAK_EVEN:>+8.4f} '
              f'{roi:>+8.2%} {se:>7.4f}')

    print('\nSE is the standard error of the win rate. A result inside')
    print('2 SE of break-even is indistinguishable from no edge.\n')

    # Same thing on recent seasons only, where the model is tuned closest
    # to current conditions and the lines are sharpest.
    for lo in [2015, 2019, 2022]:
        m = season_mask(games, min_season=lo)
        n, rate, roi = ats_record(E[m], games[m])
        se = np.sqrt(0.25 / n)
        print(f'{lo}+  {n:>5} bets  {rate:.4f}  '
              f'({(rate - BREAK_EVEN) / se:+.1f} SE from break-even)')
