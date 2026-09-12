"""
Scan every available schedule column for signal the model is missing.

rest_travel.py established the method: compute the model residual S - E,
bucket by a candidate feature, and see whether the mean residual departs
from zero. Team quality is already removed, because that is what produced
E, so anything left over is information the model does not have.

Rather than build one script per idea, this runs every column nflverse
ships against that test at once. Cheap to run, and it says where to spend
an evening instead of guessing.

Read the t column. |t| under 2 is noise. Over 3 on a decent sample is worth
building. And a single cell clearing 2 inside a feature whose other cells
are flat is a multiple-comparisons artifact, not a finding -- travel's
1500-2000 mile bucket was exactly that.
"""
import numpy as np
import pandas as pd

from data import load_games
from elo import run_elo, QB_SCALE, QB_ALPHA
from qb_data import build_qb_map

K_STAR, H_STAR, RHO_STAR = 20, 50, 0.50
MIN_N = 100          # ignore buckets too small to read

# (column, bins or None for categorical, label)
CANDIDATES = [
    ('div_game',    None,                          'divisional game'),
    ('roof',        None,                          'roof type'),
    ('surface',     None,                          'playing surface'),
    ('weekday',     None,                          'day of week'),
    ('game_type',   None,                          'regular season vs playoff'),
    ('week',        [0, 4, 9, 13, 18, 25],         'week of season'),
    ('temp',        [-20, 32, 45, 60, 75, 120],    'temperature (F)'),
    ('wind',        [-1, 5, 10, 15, 40],           'wind (mph)'),
    ('total_line',  [0, 38, 42, 46, 50, 80],       'over/under total'),
    ('spread_line', [-30, -7, -3, 0, 3, 7, 30],    'closing spread'),
]


def scan(g, col, bins, label):
    if col not in g.columns:
        print(f'\n{label} — column "{col}" not present')
        return

    series = g[col]
    if series.isna().all():
        print(f'\n{label} — all missing')
        return

    key = pd.cut(series, bins=bins) if bins else series

    agg = g.groupby(key, observed=True).agg(
        n=('resid', 'size'),
        raw=('home_win', 'mean'),
        resid=('resid', 'mean'),
        sd=('resid', 'std'),
    )
    agg = agg[agg.n >= MIN_N]
    if agg.empty:
        print(f'\n{label} — no bucket reaches n={MIN_N}')
        return

    agg['se'] = agg.sd / np.sqrt(agg.n)
    agg['t'] = agg.resid / agg.se

    flag = ' <-- worth a look' if (agg.t.abs() >= 2.5).any() else ''
    print(f'\n{label}{flag}')
    print(f'{"bucket":>18} {"n":>6} {"raw":>7} {"resid":>8} {"t":>6}')
    for idx, r in agg.iterrows():
        print(f'{str(idx):>18} {int(r.n):>6} {r.raw:>7.3f} '
              f'{r.resid:>+8.4f} {r.t:>+6.1f}')


if __name__ == '__main__':
    games = load_games()

    qb_map = build_qb_map(games, verbose=False)
    E, S, _ = run_elo(games, k=K_STAR, H=H_STAR, rho=RHO_STAR, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

    g = games.copy()
    g['resid'] = S - E
    g['home_win'] = (g.result > 0).astype(float)

    print(f'n = {len(g)}, mean residual {g.resid.mean():+.4f} (should be ~0)')
    print(f'buckets under n={MIN_N} are suppressed')

    for col, bins, label in CANDIDATES:
        scan(g, col, bins, label)

    print('\n' + '=' * 60)
    print('Flagged features are candidates, not findings. Before building')
    print('one, check that the gradient is monotone and that the extreme')
    print('bucket is the strongest -- travel failed both of those tests.')
