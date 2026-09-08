import numpy as np
import pandas as pd
import nflreadpy as nfl
from elo import run_elo
from data import load_games

K, H, RHO = 20, 50, 0.50


def current_ratings():
    """Ratings after every completed game, at theta*."""
    _, _, R = run_elo(load_games(), k=K, H=H, rho=RHO, mov=True)
    return R


def predict_week(R, season, week):
    """P(home wins) for each scheduled game, ratings reverted for the new season."""
    sched = nfl.load_schedules().to_pandas()
    games = sched[(sched.season == season) & (sched.week == week)]

    # new season: apply offseason reversion before predicting
    R = {t: 1500 + (1 - RHO) * (r - 1500) for t, r in R.items()}

    rows = []
    for g in games.itertuples():
        d = R.get(g.home_team, 1500) + H - R.get(g.away_team, 1500)
        E = 1 / (1 + 10 ** (-d / 400))
        rows.append((g.away_team, g.home_team, E, g.spread_line))
    return rows


if __name__ == '__main__':
    SEASON, WEEK = 2026, 1

    R = current_ratings()
    rows = predict_week(R, SEASON, WEEK)

    print(f'{"away":>5} {"home":>5} {"P(home)":>8} {"spread":>7}')
    for away, home, E, spread in rows:
        print(f'{away:>5} {home:>5} {E:8.3f} {spread:7.1f}')

    # save predictions
    df = pd.DataFrame(rows, columns=['away', 'home', 'E', 'spread'])
    df.insert(0, 'week', WEEK)
    df.insert(0, 'season', SEASON)
    df['theta'] = f'k={K},H={H},rho={RHO}'
    df.to_csv('predictions_2026.csv', index=False)
    print(f'\nwrote {len(df)} predictions to predictions_2026.csv')