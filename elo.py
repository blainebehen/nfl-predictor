import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
from data import load_games

# ---------------------------------------------------------------- grids
# Theta = K_GRID x H_GRID x RHO_GRID, the set of hyperparameters searched.
K_GRID = [4, 6, 8, 10, 14, 20, 24, 28, 32, 36, 44, 48, 56, 64, 80]
H_GRID = [20, 30, 40, 50, 60]
RHO_GRID = [0.0, 0.15, 0.33, 0.5, 0.7]


def run_elo(games, k=20, H=55, rho=0.33, mov=False, skip_late=False):
    """
    Walk the games in chronological order. For each game: predict, record,
    then update. The prediction uses only ratings built from PRIOR games,
    so every E is an out-of-sample forecast.

    R          ratings, team -> rating (all start at 1500)
    d          home team's rating edge, including home field
    E          predicted P(home wins)
    S          observed outcome: 1 home win, 0.5 tie, 0 away win
    M          margin-of-victory multiplier on k (1.0 when mov=False)
    skip_late  if True, still predict REG weeks 17-18 but don't learn from
               them -- those games are contaminated by teams resting
               starters once playoff seeding is locked
    """
    R = defaultdict(lambda: 1500.0)
    E_all, S_all = [], []
    season = None

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

        # --- 1. predict ----------------------------------------------
        # d is the linear predictor: rating gap plus a constant home
        # offset H (the intercept, in logistic-regression terms).
        d = R[g.home_team] + H - R[g.away_team]

        # Bradley-Terry in log space, i.e. a sigmoid with base 10:
        #     E = 1 / (1 + 10^(-d/400)) = sigma(c*d),  c = ln(10)/400
        # The 400 sets the scale, so a 400-point gap is 10:1 odds.
        E = 1 / (1 + 10 ** (-d / 400))

        # --- 2. observe ----------------------------------------------
        # result = home score - away score. Only its SIGN is used here;
        # the magnitude is handled separately by M below.
        S = 1.0 if g.result > 0 else 0.5 if g.result == 0 else 0.0

        # --- 3. record -----------------------------------------------
        # Always record, even when we don't learn from the game -- the
        # model is still being scored on it.
        E_all.append(E)
        S_all.append(S)

        # --- 4. update -----------------------------------------------
        # Skip learning from rested-starter games if requested. Playoffs
        # (game_type != 'REG') are always learned from.
        if skip_late and g.game_type == 'REG' and g.week >= 17:
            continue

        # Margin-of-victory multiplier:
        #     M = ln(|y| + 1) * 2.2 / (0.001 * d_win + 2.2)
        # ln compresses -- 3 vs 10 points matters more than 35 vs 42.
        # The denominator discounts blowouts by an already-strong
        # favorite (d_win > 0) and boosts upsets (d_win < 0), which
        # stops good teams from inflating off expected blowouts.
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


def tune(games, mov, skip_late):
    """Evaluate L at every theta in the grid; return sorted best-first."""
    results = []
    for k in K_GRID:
        for H in H_GRID:
            for rho in RHO_GRID:
                E, S, _ = run_elo(games, k=k, H=H, rho=rho,
                                  mov=mov, skip_late=skip_late)
                results.append((L(E, S), Accuracy(E, S), k, H, rho))
    results.sort()
    return results


def report(label, results):
    """Print the top 5 and flag any winner sitting at a grid boundary."""
    print(f'\n{label} — best 5 of {len(results)}')
    print(f'{"L":>7} {"Acc":>7} {"k":>4} {"H":>4} {"rho":>6}')
    for loss, acc, k, H, rho in results[:5]:
        print(f'{loss:7.4f} {acc:7.4f} {k:4d} {H:4d} {rho:6.2f}')

    _, _, k, H, rho = results[0]
    edges = []
    if k in (K_GRID[0], K_GRID[-1]):
        edges.append('k')
    if H in (H_GRID[0], H_GRID[-1]):
        edges.append('H')
    if rho in (RHO_GRID[0], RHO_GRID[-1]):
        edges.append('rho')
    if edges:
        print(f'  warning: {", ".join(edges)} at grid edge — extend the range')


if __name__ == '__main__':
    games = load_games()

    # Both variants use MOV (settled earlier: it beats no-MOV by 0.0059).
    # Each is tuned on its own grid so the comparison is fair.
    base = tune(games, mov=True, skip_late=False)
    late = tune(games, mov=True, skip_late=True)

    report('mov', base)
    report('mov + skip late', late)
    print(f'\nimprovement in L from skipping late games: '
          f'{base[0][0] - late[0][0]:+.4f}')

    # theta* = argmin over the better of the two variants
    best = late if late[0][0] < base[0][0] else base
    use_skip = best is late
    _, _, k_star, H_star, rho_star = best[0]

    E, S, R = run_elo(games, k=k_star, H=H_star, rho=rho_star,
                      mov=True, skip_late=use_skip)

    # --- Vegas benchmark ---------------------------------------------
    # spread_line is the market's predicted margin (positive = home
    # favored). Actual margins scatter around it roughly normally with
    # SD ~13.5 points, so P(home wins) = P(margin > 0) = Phi(spread/13.5).
    spread = games.spread_line.values
    mask = ~np.isnan(spread)
    vegas = norm.cdf(spread / 13.5)

    print(f'\ngames with a line: {mask.sum()}')
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

    # --- where we disagree most with the market -----------------------
    diff = np.abs(E - vegas)
    worst = np.argsort(-diff)[:20]
    cols = ['season', 'week', 'game_type', 'away_team', 'home_team',
            'result', 'spread_line']
    out = games.iloc[worst][cols].copy()
    out['elo'] = E[worst].round(3)
    out['vegas'] = vegas[worst].round(3)
    print()
    print(out.to_string())

    print(f'\nbaseline (always home)  Acc {(S > 0.5).mean():.4f}')
    print(f'elo at theta*           Acc {Accuracy(E, S):.4f}  L {L(E, S):.4f}')
    print(f'theta* = (k={k_star}, H={H_star}, rho={rho_star}, '
          f'skip_late={use_skip})\n')

    for team, rating in sorted(R.items(), key=lambda x: -x[1]):
        print(f'  {team} {rating:.0f}')
        