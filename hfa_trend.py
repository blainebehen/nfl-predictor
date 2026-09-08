"""
Raw home-win rate by season -- no smoothing, no conversion to rating points.

Written to check a claim that came out of the smoothed rolling-H table:
that home-field advantage collapsed after 2020 and stayed low. The raw
rates do not support it. 2019-2021 dipped (.521/.498/.516) and then
recovered to .563/.565/.547/.533. The apparent collapse was the 5-season
rolling window still carrying those three depressed seasons well after
they had passed.

Per-season standard error is about 0.030 (n ~ 270), so single-season
swings of a few points are noise.
"""
from data import load_games

if __name__ == '__main__':
    games = load_games()

    rates = games.assign(hw=games.result > 0).groupby('season').hw.mean()
    counts = games.groupby('season').size()

    print(f'{"season":>6} {"n":>5} {"home win rate":>14}')
    for s in rates.index:
        print(f'{s:>6} {counts[s]:>5} {rates[s]:>14.3f}')

    print(f'\nall seasons      {rates.mean():.3f}')
    print(f'1999-2018        {rates[rates.index <= 2018].mean():.3f}')
    print(f'2019-2021        {rates[(rates.index >= 2019) & (rates.index <= 2021)].mean():.3f}')
    print(f'2022-2025        {rates[rates.index >= 2022].mean():.3f}')
