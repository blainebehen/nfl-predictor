import numpy as np
from collections import defaultdict
from data import load_games
import pandas as pd
from scipy.stats import norm

K_GRID = [4, 6, 8, 10, 14, 20, 24, 28, 32, 36, 44, 48, 56, 64, 80]
H_GRID = [20, 30, 40, 50, 60]
RHO_GRID = [0.0, 0.15, 0.33, 0.5, 0.7]


def run_elo(games, k=20, H=55, rho=0.33, mov=False):
    """
    R      ratings (dict: team -> rating)
    d      home team's rating edge, incl. home field
    E      predicted P(home wins)
    S      observed outcome: 1 home win, 0.5 tie, 0 away win
    M      margin-of-victory multiplier on k (1.0 if mov=False)
    """
    R = defaultdict(lambda: 1500.0)
    E_all, S_all = [], []
    season = None

    for g in games.itertuples():

        # R <- 1500 + (1 - rho)(R - 1500)
        if g.season != season:
            for t in R:
                R[t] = 1500 + (1 - rho) * (R[t] - 1500)
            season = g.season

        # d = R_h + H - R_a
        d = R[g.home_team] + H - R[g.away_team]

        # E = 1 / (1 + 10^(-d/400))
        E = 1 / (1 + 10 ** (-d / 400))

        # S from sign of result
        S = 1.0 if g.result > 0 else 0.5 if g.result == 0 else 0.0

        # M = ln(|y| + 1) * 2.2 / (0.001 * d_win + 2.2)
        if mov and g.result != 0:
            d_win = d if g.result > 0 else -d
            M = np.log(abs(g.result) + 1) * 2.2 / (0.001 * d_win + 2.2)
        else:
            M = 1.0

        E_all.append(E)
        S_all.append(S)

        # R_h <- R_h + k*M*(S - E),  R_a <- R_a - k*M*(S - E)
        R[g.home_team] += k * M * (S - E)
        R[g.away_team] -= k * M * (S - E)

    return np.array(E_all), np.array(S_all), dict(R)


def Accuracy(E, S):
    """(1/N) sum 1[(E - 1/2)(S - 1/2) > 0]"""
    return ((E - 0.5) * (S - 0.5) > 0).mean()


def L(E, S):
    """-(1/N) sum [ S ln E + (1 - S) ln(1 - E) ]"""
    return -(S * np.log(E) + (1 - S) * np.log(1 - E)).mean()


def tune(games, mov):
    """Evaluate L over the full grid; returns results sorted best-first."""
    results = []
    for k in K_GRID:
        for H in H_GRID:
            for rho in RHO_GRID:
                E, S, _ = run_elo(games, k=k, H=H, rho=rho, mov=mov)
                results.append((L(E, S), Accuracy(E, S), k, H, rho))
    results.sort()
    return results


def report(label, results):
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

    plain = tune(games, mov=False)
    withmov = tune(games, mov=True)

    report('no mov', plain)
    report('with mov', withmov)

    print(f'\nimprovement in L: {plain[0][0] - withmov[0][0]:+.4f}')

    # theta* = argmin over the better of the two
    best = withmov if withmov[0][0] < plain[0][0] else plain
    use_mov = best is withmov
    _, _, k_star, H_star, rho_star = best[0]

    E, S, R = run_elo(games, k=k_star, H=H_star, rho=rho_star, mov=use_mov)
    
    spread = games.spread_line.values
    mask = ~np.isnan(spread)

    vegas = norm.cdf(spread / 13.5)
    
    diff = np.abs(E - vegas)
    worst = np.argsort(-diff)[:20]
    cols = ['season', 'week', 'away_team', 'home_team', 'result', 'spread_line']
    out = games.iloc[worst][cols].copy()
    out['elo'] = E[worst].round(3)
    out['vegas'] = vegas[worst].round(3)
    print(out.to_string())

    print(f'\ngames with a line: {mask.sum()}')
    print(f'elo    L {L(E[mask], S[mask]):.4f}  Acc {Accuracy(E[mask], S[mask]):.4f}')
    print(f'vegas  L {L(vegas[mask], S[mask]):.4f}  Acc {Accuracy(vegas[mask], S[mask]):.4f}')
    
    df = pd.DataFrame({'E': E, 'S': S})
    df['bucket'] = pd.cut(df.E, bins=[0, .3, .4, .5, .6, .7, .8, 1.0])

    print(df.groupby('bucket', observed=True).agg(
        n=('S', 'size'),
        predicted=('E', 'mean'),
        actual=('S', 'mean')).round(3))

    print(f'\nbaseline (always home)  Acc {(S > 0.5).mean():.4f}')
    print(f'elo at theta*           Acc {Accuracy(E, S):.4f}  L {L(E, S):.4f}')
    print(f'theta* = (k={k_star}, H={H_star}, rho={rho_star}, mov={use_mov})\n')

    for team, rating in sorted(R.items(), key=lambda x: -x[1])[:32]:
        print(f'  {team} {rating:.0f}')
        
        
        