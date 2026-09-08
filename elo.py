import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
from data import load_games

# ---------------------------------------------------------------- grids
# Theta = K_GRID x H_GRID x RHO_GRID, the set of hyperparameters searched.
K_GRID = [4, 6, 8, 10, 14, 20, 24, 28, 32, 36, 44, 48, 56, 64, 80]
H_GRID = [20, 30, 40, 50, 60, 70, 80]
RHO_GRID = [0.0, 0.15, 0.33, 0.5, 0.7]
WINDOW_GRID = [3, 5, 8, 12]     # seasons of history behind a rolling H

TEST_START = 2019   # held-out evaluation window: this season onward


def rolling_hfa(games, window=5, prior=0.5631):
    """
    Home-field advantage in rating points, one value per season, derived
    from the home-win RATE of the prior `window` seasons.

    Two steps: average the home-win rate over those prior seasons, then
    invert the Elo formula to express that rate as rating points. If home
    teams win a fraction p of games between evenly-rated opponents, then
    p = 1/(1 + 10^(-H/400)), so H = 400 * log10(p / (1 - p)).

    Only PRIOR seasons are used, which is what keeps this leakage-free:
    H for 2024 never sees a 2024 result. Using a season's own rate would
    mean predicting games with information that includes those games.
    `prior` is the fallback for the first season, where no history exists
    (the all-time home-win rate).
    """
    by_season = games.assign(hw=games.result > 0).groupby('season').hw.mean()

    hfa = {}
    for season in by_season.index:
        past = by_season[by_season.index < season].tail(window)
        p = past.mean() if len(past) else prior
        hfa[season] = 400 * np.log10(p / (1 - p))
    return hfa


def run_elo(games, k=20, H=55, rho=0.33, mov=False, skip_late=False,
            hfa=None):
    """
    Walk the games in chronological order. For each game: predict, record,
    then update. The prediction uses only ratings built from PRIOR games,
    so every E is an out-of-sample forecast.

    R          ratings, team -> rating (all start at 1500)
    d          home team's rating edge, including home field
    E          predicted P(home wins)
    S          observed outcome: 1 home win, 0.5 tie, 0 away win
    M          margin-of-victory multiplier on k (1.0 when mov=False)
    hfa        optional dict season -> H. When given it overrides the
               constant H, letting home-field advantage vary by era.
    skip_late  if True, still predict REG weeks 17-18 but don't learn from
               them. Tested and rejected (see RESULTS.md); kept for
               reproducibility.
    """
    R = defaultdict(lambda: 1500.0)
    E_all, S_all = [], []
    season = None
    H_now = H

    for g in games.itertuples():

        # --- offseason reversion -------------------------------------
        # Rosters turn over, so shrink every team toward the 1500 mean:
        #     R <- 1500 + (1 - rho)(R - 1500)
        # Recenter at 0, scale by (1 - rho), shift back. rho=0 never
        # forgets; rho=1 resets everyone to 1500.
        if g.season != season:
            for t in R:
                R[t] = 1500 + (1 - rho) * (R[t] - 1500)
            season = g.season
            if hfa is not None:
                H_now = hfa[season]

        # --- 1. predict ----------------------------------------------
        # d is the linear predictor: rating gap plus a home offset (the
        # intercept, in logistic-regression terms).
        d = R[g.home_team] + H_now - R[g.away_team]

        # Bradley-Terry in log space, i.e. a sigmoid with base 10:
        #     E = 1 / (1 + 10^(-d/400)) = sigma(c*d),  c = ln(10)/400
        # The 400 sets the scale, so a 400-point gap is 10:1 odds.
        E = 1 / (1 + 10 ** (-d / 400))

        # --- 2. observe ----------------------------------------------
        # result = home score - away score. Only its SIGN is used here;
        # the magnitude is handled separately by M below.
        S = 1.0 if g.result > 0 else 0.5 if g.result == 0 else 0.0

        # --- 3. record -----------------------------------------------
        E_all.append(E)
        S_all.append(S)

        # --- 4. update -----------------------------------------------
        if skip_late and g.game_type == 'REG' and g.week >= 17:
            continue

        # Margin-of-victory multiplier:
        #     M = ln(|y| + 1) * 2.2 / (0.001 * d_win + 2.2)
        # ln compresses -- 3 vs 10 points matters more than 35 vs 42.
        # The denominator discounts blowouts by an already-strong
        # favorite (d_win > 0) and boosts upsets (d_win < 0).
        if mov and g.result != 0:
            d_win = d if g.result > 0 else -d
            M = np.log(abs(g.result) + 1) * 2.2 / (0.001 * d_win + 2.2)
        else:
            M = 1.0

        # SGD step on log loss. The gradient of the per-game loss wrt
        # R_h is c(E - S), so descending gives +k*(S - E); k absorbs c
        # and the learning rate. The away team gets the mirror image
        # because dd/dR_a = -1, which is why ratings are zero-sum.
        R[g.home_team] += k * M * (S - E)
        R[g.away_team] -= k * M * (S - E)

    return np.array(E_all), np.array(S_all), dict(R)


def Accuracy(E, S):
    """
    (1/N) sum 1[(E - 1/2)(S - 1/2) > 0]

    The product is positive exactly when prediction and outcome fall on
    the same side of 1/2, i.e. when the pick was right.
    """
    return ((E - 0.5) * (S - 0.5) > 0).mean()


def L(E, S):
    """
    -(1/N) sum [ S ln E + (1 - S) ln(1 - E) ]

    S and (1-S) act as switches: exactly one term survives per game, so
    each term is the log of the probability assigned to what actually
    happened. Reference points: 0.6931 = always predicting 0.5, and 0 =
    perfect foresight. Lower is better.
    """
    return -(S * np.log(E) + (1 - S) * np.log(1 - E)).mean()


def season_mask(games, min_season=None, max_season=None):
    """Boolean mask selecting games within a season window."""
    m = np.ones(len(games), dtype=bool)
    if min_season is not None:
        m &= (games.season >= min_season).values
    if max_season is not None:
        m &= (games.season <= max_season).values
    return m


def tune_fixed(games, min_season=None, max_season=None):
    """Grid over (k, H, rho) with a constant home-field advantage."""
    window = season_mask(games, min_season, max_season)
    results = []
    for k in K_GRID:
        for H in H_GRID:
            for rho in RHO_GRID:
                E, S, _ = run_elo(games, k=k, H=H, rho=rho, mov=True)
                Ew, Sw = E[window], S[window]
                results.append((L(Ew, Sw), Accuracy(Ew, Sw), k, H, rho))
    results.sort()
    return results


def tune_rolling(games, min_season=None, max_season=None):
    """Grid over (k, rho, window) with a season-varying H."""
    mask = season_mask(games, min_season, max_season)
    hfas = {w: rolling_hfa(games, window=w) for w in WINDOW_GRID}

    results = []
    for k in K_GRID:
        for rho in RHO_GRID:
            for w in WINDOW_GRID:
                E, S, _ = run_elo(games, k=k, rho=rho, mov=True, hfa=hfas[w])
                Ew, Sw = E[mask], S[mask]
                results.append((L(Ew, Sw), Accuracy(Ew, Sw), k, w, rho))
    results.sort()
    return results


def report(label, results, cols=('k', 'H', 'rho')):
    """Print the top 5 of a tuning run."""
    print(f'\n{label} — best 5 of {len(results)}')
    print(f'{"L":>7} {"Acc":>7} {cols[0]:>4} {cols[1]:>4} {cols[2]:>6}')
    for loss, acc, a, b, c in results[:5]:
        print(f'{loss:7.4f} {acc:7.4f} {a:4d} {b:4d} {c:6.2f}')


if __name__ == '__main__':
    games = load_games()

    # --- what the rolling H actually looks like -----------------------
    # Each season's value comes from the prior 5 seasons' home-win rate.
    # Stable near 50 through 2019, then a sharp drop from 2020 onward.
    hfa5 = rolling_hfa(games, window=5)
    print('rolling H, each season from the prior 5 seasons:')
    for s in sorted(hfa5):
        print(f'  {s}  {hfa5[s]:5.1f}')

    # --- fixed vs rolling H, both tuned in-sample ---------------------
    fixed = tune_fixed(games)
    roll = tune_rolling(games)

    report('fixed H, in-sample', fixed, cols=('k', 'H', 'rho'))
    report('rolling H, in-sample', roll, cols=('k', 'win', 'rho'))
    print(f'\nimprovement in L from rolling H: {fixed[0][0] - roll[0][0]:+.4f}')

    # --- does rolling H close the held-out gap? -----------------------
    # In-sample the two are indistinguishable. The question is whether a
    # season-varying H transfers better to a period whose home-field
    # advantage differs from the historical average.
    test = season_mask(games, min_season=TEST_START)
    print(f'\n{"=" * 60}')
    print(f'held-out test: {TEST_START}+, {test.sum()} games')

    f_train = tune_fixed(games, max_season=TEST_START - 1)
    report(f'fixed H, tuned on <{TEST_START}', f_train, cols=('k', 'H', 'rho'))
    _, _, k_f, H_f, rho_f = f_train[0]
    Ef, Sf, _ = run_elo(games, k=k_f, H=H_f, rho=rho_f, mov=True)
    print(f'  test L {L(Ef[test], Sf[test]):.4f}  '
          f'Acc {Accuracy(Ef[test], Sf[test]):.4f}  '
          f'gap {L(Ef[test], Sf[test]) - fixed[0][0]:+.4f}')

    r_train = tune_rolling(games, max_season=TEST_START - 1)
    report(f'rolling H, tuned on <{TEST_START}', r_train, cols=('k', 'win', 'rho'))
    _, _, k_r, w_r, rho_r = r_train[0]
    Er, Sr, _ = run_elo(games, k=k_r, rho=rho_r, mov=True,
                        hfa=rolling_hfa(games, window=w_r))
    print(f'  test L {L(Er[test], Sr[test]):.4f}  '
          f'Acc {Accuracy(Er[test], Sr[test]):.4f}  '
          f'gap {L(Er[test], Sr[test]) - roll[0][0]:+.4f}')

    # --- final model: whichever won in-sample -------------------------
    if roll[0][0] < fixed[0][0]:
        _, _, k_s, w_s, rho_s = roll[0]
        E, S, R = run_elo(games, k=k_s, rho=rho_s, mov=True,
                          hfa=rolling_hfa(games, window=w_s))
        theta = f'k={k_s}, rolling H (window={w_s}), rho={rho_s}'
    else:
        _, _, k_s, H_s, rho_s = fixed[0]
        E, S, R = run_elo(games, k=k_s, H=H_s, rho=rho_s, mov=True)
        theta = f'k={k_s}, H={H_s}, rho={rho_s}'

    # --- Vegas benchmark ---------------------------------------------
    # spread_line is the market's predicted margin (positive = home
    # favored). Actual margins scatter around it roughly normally with
    # SD ~13.5 points, so P(home wins) = P(margin > 0) = Phi(spread/13.5).
    spread = games.spread_line.values
    mask = ~np.isnan(spread)
    vegas = norm.cdf(spread / 13.5)

    print(f'\n{"=" * 60}')
    print(f'games with a line: {mask.sum()}')
    print(f'elo    L {L(E[mask], S[mask]):.4f}  Acc {Accuracy(E[mask], S[mask]):.4f}')
    print(f'vegas  L {L(vegas[mask], S[mask]):.4f}  Acc {Accuracy(vegas[mask], S[mask]):.4f}')

    # --- calibration --------------------------------------------------
    # Bucket by predicted probability; within each bucket the mean
    # prediction should match the observed win rate.
    df = pd.DataFrame({'E': E, 'S': S})
    df['bucket'] = pd.cut(df.E, bins=[0, .3, .4, .5, .6, .7, .8, 1.0])
    print()
    print(df.groupby('bucket', observed=True).agg(
        n=('S', 'size'),
        predicted=('E', 'mean'),
        actual=('S', 'mean')).round(3))

    print(f'\nbaseline (always home)  Acc {(S > 0.5).mean():.4f}')
    print(f'elo at theta*           Acc {Accuracy(E, S):.4f}  L {L(E, S):.4f}')
    print(f'theta* = ({theta})\n')

    for team, rating in sorted(R.items(), key=lambda x: -x[1]):
        print(f'  {team} {rating:.0f}')
