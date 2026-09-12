"""
Rest and travel: do they show up in the raw data before we model them?

Same discipline as hfa_trend.py. The rolling-H episode came from theorising
about a smoothed series without checking the raw one, so: compute the
candidate features, bucket them, and look at actual home-win rates before
writing any model code.

A team off a bye playing a team on a short week should have an edge. 
Travel is plausible too, though the effect is usually reported as small 
and tangled up with home field, which the model already has.

Two passes. The first looks at raw home-win rate by bucket. The second
looks at MODEL RESIDUALS by bucket, which is the test that actually
matters.

Why residuals: travel distance is not randomly assigned. Long trips are
made by the same handful of teams every year -- Seattle, the LA clubs, San
Francisco, Las Vegas, Miami -- so the long-travel bucket is really a sample
of those franchises, not a sample of tired teams. If they were good over
this period, and several were, the raw rate reflects facing good opponents
rather than jet lag.

The residual S - E already has team quality removed, because that is what
the ratings encode. So the question becomes: in long-travel games, does the
model systematically misprice the away team? A residual near zero in every
bucket means Elo has already absorbed whatever the feature does and there
is nothing left to add.

RESULT: neither feature clears the bar. Residual by bucket --

  rest diff   -30..-6  -6..-2  -2..-.5  -.5..5  .5..2  2..6  6..30
              -0.051   -0.019   +0.005  -0.001  +0.006 -0.023 +0.028

  travel      0-250  250-500  500-1k  1k-1.5k  1.5k-2k  2k-3.5k
              -0.019  -0.018  -0.003   +0.003   +0.047  -0.019

Rest is scatter, not a gradient, and the leftmost cell has the wrong sign
for a fatigue story. The direct test -- home off a bye against a team on
short rest -- is -0.1 SE on 58 games.

Travel has one cell at +2.8 SE (1500-2000 mi), but the bucket above it
reverses to -1.1 on a larger sample. Six buckets tested, one clearing 2 SE
is what chance produces.

Note the travel RAW rates climb cleanly from .548 to .615 across the first
five buckets, exactly the pattern the feature predicts -- while the
residuals stay flat. That is the confounding made visible: the model
already knew those away teams were good or bad, and the apparent travel
effect was team quality the whole time. Worth keeping as the clearest
example in this repo of why raw rates mislead.
"""
import numpy as np
import pandas as pd

from data import load_games
from elo import run_elo, QB_SCALE, QB_ALPHA
from qb_data import build_qb_map

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50

# Approximate stadium coordinates, city-level. Good enough for distances
# measured in hundreds of miles; not good enough for anything finer.
STADIUMS = {
    'ARI': (33.53, -112.26), 'ATL': (33.76, -84.40), 'BAL': (39.28, -76.62),
    'BUF': (42.77, -78.79),  'CAR': (35.23, -80.85), 'CHI': (41.86, -87.62),
    'CIN': (39.10, -84.52),  'CLE': (41.51, -81.70), 'DAL': (32.75, -97.09),
    'DEN': (39.74, -105.02), 'DET': (42.34, -83.05), 'GB':  (44.50, -88.06),
    'HOU': (29.68, -95.41),  'IND': (39.76, -86.16), 'JAX': (30.32, -81.64),
    'KC':  (39.05, -94.48),  'LA':  (33.95, -118.34),'LAC': (33.95, -118.34),
    'LV':  (36.09, -115.18), 'MIA': (25.96, -80.24), 'MIN': (44.97, -93.26),
    'NE':  (42.09, -71.26),  'NO':  (29.95, -90.08), 'NYG': (40.81, -74.07),
    'NYJ': (40.81, -74.07),  'PHI': (39.90, -75.17), 'PIT': (40.45, -80.02),
    'SEA': (47.59, -122.33), 'SF':  (37.40, -121.97),'TB':  (27.98, -82.50),
    'TEN': (36.17, -86.77),  'WAS': (38.91, -76.86),
}


def haversine(a, b):
    """Great-circle distance in miles between two (lat, lon) pairs."""
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(h))


def add_features(games):
    """Rest differential and away-team travel distance, per game."""
    g = games.copy()

    if {'home_rest', 'away_rest'}.issubset(g.columns):
        source = 'nflverse home_rest / away_rest'
    else:
        # fall back to computing days since each team's previous game
        source = 'computed from gameday'
        g['gameday'] = pd.to_datetime(g.gameday)
        last = {}
        hr, ar = [], []
        for r in g.itertuples():
            for team, out in ((r.home_team, hr), (r.away_team, ar)):
                prev = last.get(team)
                out.append((r.gameday - prev).days if prev is not None else np.nan)
                last[team] = r.gameday
        g['home_rest'], g['away_rest'] = hr, ar

    g['rest_diff'] = g.home_rest - g.away_rest

    g['travel'] = [
        haversine(STADIUMS[r.away_team], STADIUMS[r.home_team])
        if r.away_team in STADIUMS and r.home_team in STADIUMS else np.nan
        for r in g.itertuples()
    ]

    g['home_win'] = (g.result > 0).astype(float)
    return g, source


def bucket_report(g, col, bins, label):
    """
    Per bucket: raw home-win rate, and the mean model residual S - E.

    resid > 0 means the home team did better than the model expected, so a
    gradient in that column is information the model does not have. The
    raw rate is shown alongside for contrast -- where the two disagree, the
    raw one is being driven by which teams land in the bucket.
    """
    b = pd.cut(g[col], bins=bins)
    agg = g.groupby(b, observed=True).agg(
        n=('home_win', 'size'),
        raw=('home_win', 'mean'),
        resid=('resid', 'mean'),
    )
    agg['se'] = g.groupby(b, observed=True).resid.std() / np.sqrt(agg.n)

    print(f'\n{label}')
    print(f'{"bucket":>16} {"n":>6} {"raw":>7} {"resid":>8} {"SE":>7} {"t":>6}')
    for idx, r in agg.iterrows():
        t = r.resid / r.se if r.se else np.nan
        print(f'{str(idx):>16} {int(r.n):>6} {r.raw:>7.3f} {r.resid:>+8.4f} '
              f'{r.se:>7.4f} {t:>+6.1f}')


if __name__ == '__main__':
    games = load_games()
    g, source = add_features(games)

    # model residuals: what the current model gets wrong
    qb_map = build_qb_map(games, verbose=False)
    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)
    g['resid'] = S - E

    print(f'rest source: {source}')
    print(f'baseline home-win rate: {g.home_win.mean():.4f}  (n={len(g)})')
    print(f'mean residual overall:  {g.resid.mean():+.4f}  (should be ~0)')

    bucket_report(g, 'rest_diff', [-30, -6, -2, -0.5, 0.5, 2, 6, 30],
                  'by rest differential (home rest - away rest, days)')

    bucket_report(g, 'travel', [-1, 250, 500, 1000, 1500, 2000, 3500],
                  'by away-team travel distance (miles)')

    # Extreme cases, where any effect should be largest
    print('\nextremes')
    for label, mask in [
        ('home off a bye, away on short rest', (g.home_rest >= 10) & (g.away_rest <= 6)),
        ('away off a bye, home on short rest', (g.away_rest >= 10) & (g.home_rest <= 6)),
        ('away travelled 2000+ miles',         g.travel >= 2000),
        ('away travelled under 250 miles',     g.travel < 250),
    ]:
        sub = g[mask.fillna(False)]
        if len(sub) < 30:
            print(f'  {label:<38} n={len(sub)} — too few to read')
            continue
        se = sub.resid.std() / np.sqrt(len(sub))
        print(f'  {label:<38} n={len(sub):>5}  '
              f'raw {sub.home_win.mean():.3f}  '
              f'resid {sub.resid.mean():+.4f}  ({sub.resid.mean() / se:+.1f} SE)')

    print('\nThe resid column is the one that matters. A feature worth adding')
    print('shows a monotone gradient there and extremes several SE from zero.')
    print('A flat resid column means the model already accounts for it.')
