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

# Adopted QB adjustment. Tuned in qb.py, validated across six held-out
# windows in qb_windows.py -- every window positive, +0.0037 to +0.0121.
QB_SCALE = 600      # rating points per unit of EPA/dropback
QB_ALPHA = 0.02     # EWMA rate for a quarterback's rating

# Adopted team-EPA adjustment. Tuned in epa.py, validated across the same
# six windows in epa_windows.py -- every window positive, +0.0011 to
# +0.0034, with all six independently selecting scale=200.
EPA_SCALE = 200     # rating points per unit of net EPA/play
EPA_ALPHA = 0.15    # EWMA rate for a team's offensive and defensive EPA

# Clinch adjustment. CANDIDATE, NOT IN theta*. A team whose playoff seed
# can no longer move underperforms its rating in REG weeks 15+ -- it rests
# starters, and the ratings cannot see it.
#
# Use the RANK-FROZEN flag, not "has clinched a berth". A team that has
# secured a place but is still fighting for seeding has everything to play
# for; pooling it in dilutes the effect roughly fourfold. Build the map
# with clinch_wide.build_wide_map and keep the 'locked' label:
#
#   lab = build_wide_map(games)
#   cmap = {k: 1.0 for k, v in lab.items() if v == 'locked'}
#   run_elo(..., clinch_map=cmap, clinch_scale=CLINCH_SCALE)
#
# Four of six windows positive, but five of six independently selected
# 130 -- the only stable parameter any version of this produced. Nothing
# uses it by default. Read the three Clinch status sections of RESULTS.md
# before turning it on: the evidence is exploratory, not confirmatory.
#
# Do NOT reach for skip_games instead. Excluding these games from the
# update was tested on exactly this flag and loses, 1 of 6 windows.
CLINCH_SCALE = 130  # rating points docked from a rank-frozen team


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

    TESTED, NOT ADOPTED. theta* uses a fixed H. Across six held-out test
    windows (holdout_windows.py) this beat a fixed H on exactly one --
    2019-2021 -- and was within +/-0.0017 or worse on the rest. Kept here
    so the comparison stays reproducible.
    """
    by_season = games.assign(hw=games.result > 0).groupby('season').hw.mean()

    hfa = {}
    for season in by_season.index:
        past = by_season[by_season.index < season].tail(window)
        p = past.mean() if len(past) else prior
        hfa[season] = 400 * np.log10(p / (1 - p))
    return hfa


def run_elo(games, k=20, H=55, rho=0.33, mov=False, skip_late=False,
            hfa=None, qb_map=None, qb_scale=0.0, qb_alpha=0.10,
            qb_beta=0.03, qb_init=0.0, epa_map=None, epa_scale=0.0,
            epa_alpha=0.15, clinch_map=None, clinch_scale=0.0,
            skip_games=None):
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
    qb_map     optional dict (game_id, team) -> (player_id, epa_per_dropback)
               from qb.build_qb_map. Enables the QB adjustment.
    qb_scale   rating points per unit of EPA/dropback deviation. 0 disables
               the adjustment even when qb_map is supplied, which is the
               control condition for measuring whether it helps.
    qb_alpha   EWMA rate for a quarterback's own rating Q.
    qb_beta    EWMA rate for a team's starter baseline T. Smaller than
               alpha on purpose: T should represent what a team normally
               gets at the position, so it should not chase a backup's
               couple of starts.
    qb_init    starting Q for a quarterback with no history, and starting
               T for a team. League mean EPA/dropback is about 0.042;
               0.0 is a mild penalty reflecting that debut starters are
               usually below average. Not tuned.
    epa_map    optional dict (game_id, team) -> (off_epa, def_epa) per play,
               from epa_test.team_game_epa. Enables the EPA adjustment.
    epa_scale  rating points per unit of net-EPA differential. 0 disables
               it, which is the control condition.
    epa_alpha  EWMA rate for a team's offensive and defensive EPA.
    clinch_map optional dict (game_id, team) -> 1.0 for a team that has
               clinched a playoff berth entering that game, from
               clinch.build_clinch_map. Absent keys count as 0.
    clinch_scale rating points docked from a team that has clinched. 0
               disables it, which is the control condition. Eliminated
               teams are deliberately NOT flagged -- they show the
               opposite sign, and pooling the two cancels both.
    skip_games optional set of game_ids to predict and score, but NOT
               learn from. The targeted form of skip_late: instead of
               discarding whole weeks, discard only the games a flag says
               are contaminated. A rested-starter loss teaches the ratings
               something false, and unlike the clinch adjustment -- which
               fixes the prediction -- this protects the rating itself.
    skip_late  if True, still predict REG weeks 17-18 but don't learn from
               them. Tested and rejected (see RESULTS.md); kept for
               reproducibility.
    """
    R = defaultdict(lambda: 1500.0)
    E_all, S_all = [], []
    season = None
    H_now = H

    # QB state: Q is per quarterback, T is per team. Both are EWMAs of
    # EPA per dropback, read before a game and updated after it.
    use_qb = qb_map is not None and qb_scale != 0.0
    Q = defaultdict(lambda: qb_init)
    T = defaultdict(lambda: qb_init)

    # EPA state: per-play offensive and defensive EWMAs. Net rating is
    # offence minus defence allowed, so higher is better.
    use_epa = epa_map is not None and epa_scale != 0.0
    OFF, DEF = {}, {}

    # Clinch state is precomputed from standings, so there is nothing to
    # carry between games -- just a lookup.
    use_clinch = clinch_map is not None and clinch_scale != 0.0

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
        # QB adjustment: how far each team's starter sits from what that
        # team normally gets, differenced so it enters d as a home edge.
        # Team Elo already absorbs a franchise's average QB quality, so
        # only the deviation is new information. Missing QB data (0.3% of
        # team-games) contributes nothing.
        adj = 0.0
        if use_qb:
            h_qb = qb_map.get((g.game_id, g.home_team))
            a_qb = qb_map.get((g.game_id, g.away_team))
            dev_h = Q[h_qb[0]] - T[g.home_team] if h_qb else 0.0
            dev_a = Q[a_qb[0]] - T[g.away_team] if a_qb else 0.0
            adj = qb_scale * (dev_h - dev_a)

        # EPA adjustment: Elo updates on who won, scaled by margin. EPA
        # per play measures how a team actually moved the ball, which is
        # less noisy than the scoreboard. The two correlate ~0.8; the
        # residual test (epa_test.py) shows Elo is wrong where they part.
        if use_epa:
            net_h = OFF.get(g.home_team, 0.0) - DEF.get(g.home_team, 0.0)
            net_a = OFF.get(g.away_team, 0.0) - DEF.get(g.away_team, 0.0)
            adj += epa_scale * (net_h - net_a)

        # Clinch adjustment: a team that has already secured a playoff
        # berth underperforms its rating in late-season games -- starters
        # rest, and the ratings cannot see it. Docked from whichever side
        # has clinched, so the sign is (away - home). Nonzero only on REG
        # weeks 15+; every other game has no entry in clinch_map.
        if use_clinch:
            c_h = clinch_map.get((g.game_id, g.home_team), 0.0)
            c_a = clinch_map.get((g.game_id, g.away_team), 0.0)
            adj += clinch_scale * (c_a - c_h)

        # d is the linear predictor: rating gap plus a home offset (the
        # intercept, in logistic-regression terms) plus any adjustments.
        d = R[g.home_team] + H_now - R[g.away_team] + adj

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
        if skip_games is not None and g.game_id in skip_games:
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

        # QB update: move each starter's rating and his team's baseline
        # toward what he actually did. Same shape as the Elo step -- a
        # fraction of the way toward the new observation.
        if use_qb:
            for qb, team in ((h_qb, g.home_team), (a_qb, g.away_team)):
                if qb is None:
                    continue
                pid, val = qb
                Q[pid] += qb_alpha * (val - Q[pid])
                T[team] += qb_beta * (val - T[team])

        if use_epa:
            for team in (g.home_team, g.away_team):
                entry = epa_map.get((g.game_id, team))
                if entry is None:
                    continue
                o, d_ = entry
                OFF[team] = OFF.get(team, o) + epa_alpha * (o - OFF.get(team, o))
                DEF[team] = DEF.get(team, d_) + epa_alpha * (d_ - DEF.get(team, d_))

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
    from qb_data import build_qb_map
    from epa_test import team_game_epa

    games = load_games()

    # --- what the rolling H looks like --------------------------------
    # Each season's value comes from the prior 5 seasons' home-win rate.
    # NOTE: this smoothed view once suggested a permanent collapse after
    # 2020. The raw per-season rates (hfa_trend.py) show a 2019-2021 dip
    # that recovered -- the "collapse" was the window lagging that dip.
    # holdout_windows.py then showed rolling H only beats a fixed H on
    # windows containing 2019-2021, so it is not adopted -- see below.
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

    # --- which transfers better to unseen seasons? --------------------
    # In-sample the two are indistinguishable. The test window opens on
    # the depressed 2019-2021 seasons, which a rolling H can adapt to and
    # a fixed H cannot -- see holdout_windows.py for the same comparison
    # across five other windows, where the gain largely disappears.
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

    # --- final model: fixed H + QB adjustment -------------------------
    # Rolling H is not adopted. It wins in-sample by 0.0001 and held out
    # by 0.0033, but holdout_windows.py shows that entire gain sits in the
    # 2019-2021 window and vanishes elsewhere. A fixed H is simpler and
    # the evidence does not clear the bar for adding a parameter.
    #
    # The QB adjustment IS adopted: it wins all six held-out windows
    # (qb_windows.py) and both directions agree, which is what separates
    # it from rolling H.
    _, _, k_s, H_s, rho_s = fixed[0]
    qb_map = build_qb_map(games, verbose=False)
    epa_map = team_game_epa()
    E, S, R = run_elo(games, k=k_s, H=H_s, rho=rho_s, mov=True,
                      qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA,
                      epa_map=epa_map, epa_scale=EPA_SCALE,
                      epa_alpha=EPA_ALPHA)
    theta = (f'k={k_s}, H={H_s}, rho={rho_s}, '
             f'qb_scale={QB_SCALE}, qb_alpha={QB_ALPHA}, '
             f'epa_scale={EPA_SCALE}, epa_alpha={EPA_ALPHA}')

    # references on the same games, so the header numbers are comparable
    E0, S0, _ = run_elo(games, k=k_s, H=H_s, rho=rho_s, mov=True)
    Eq, Sq, _ = run_elo(games, k=k_s, H=H_s, rho=rho_s, mov=True,
                        qb_map=qb_map, qb_scale=QB_SCALE, qb_alpha=QB_ALPHA)

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
    print(f'elo + mov               Acc {Accuracy(E0, S0):.4f}  L {L(E0, S0):.4f}')
    print(f'  + qb                  Acc {Accuracy(Eq, Sq):.4f}  L {L(Eq, Sq):.4f}')
    print(f'  + epa (theta*)        Acc {Accuracy(E, S):.4f}  L {L(E, S):.4f}')
    print(f'theta* = ({theta})\n')

    for team, rating in sorted(R.items(), key=lambda x: -x[1]):
        print(f'  {team} {rating:.0f}')
