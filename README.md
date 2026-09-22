# NFL win predictor

A model that predicts who wins NFL games, built on Elo ratings with two
extra pieces: an adjustment for who's starting at quarterback, and one for
how efficiently each team has been playing (EPA per play). I compare it
against the Vegas line and track its picks live during the 2026 season.

## How it works

Every team has a rating, starting at 1500. To predict a game:

1. Take the home team's rating minus the away team's, plus a bonus for
   home field.
2. Adjust for the quarterbacks. Each QB has a rating based on his recent
   EPA per dropback. The model only uses how far a team's starter is from
   who they *normally* play, since the team rating already includes that.
   This is what lets it react when a backup has to start.
3. Adjust for team EPA per play, offense and defense. EPA measures how
   well a team actually moved the ball, which is less noisy than the final
   score.
4. Convert that to a win probability with the Elo formula:
   `P(home wins) = 1 / (1 + 10^(-edge/400))`.

After each game, both teams' ratings move based on how surprising the
result was. Winning by more moves them more, and ratings shrink back
toward average each offseason as rosters change.

Every game is predicted using only games played before it, so the whole
backtest is out of sample.

## Results

7,278 games, 1999-2025:

| model                 | accuracy | log loss |
|-----------------------|----------|----------|
| always guess 50%      | 0.500    | 0.693    |
| always pick home team | 0.563    | —        |
| Elo + margin of victory | 0.641  | 0.631    |
| + QB adjustment       | 0.643    | 0.629    |
| + team EPA            | 0.644    | 0.627    |
| Vegas closing line    | 0.661    | 0.614    |

Log loss is the main score (lower is better). It rewards being confident
and right, and punishes being confident and wrong, which accuracy doesn't.

The model tunings above were picked using all these games, so they're a
little optimistic. As a fairer test, I tuned only on 1999-2018 and scored
2019-2025: log loss 0.640 → 0.634 → 0.632 for the three versions. Same
order, same improvements.

The model is also well calibrated: when it says 70%, the home team wins
about 70% of the time.

It doesn't beat Vegas, and I didn't expect it to. It gets about 80% of the
way from a coin flip to the Vegas line.

## What I learned

**The QB adjustment was the biggest single improvement.** I tested it on
six different stretches of seasons, each time tuning only on earlier
years. It helped in all six.

**Team EPA helped a little, but every time.** Smaller than QB, but it
also helped in all six test stretches.

**Some ideas looked good once and didn't hold up.**
- Letting home-field advantage change by era looked like a clear win on
  2019-2025. Tested across all six stretches, the whole gain came from
  2019-2021, when home teams won unusually rarely (including the COVID
  seasons with few or no fans). I didn't keep it.
- Travel distance: home teams do win more when the away team travels
  farther, but that's because the same far-away teams make those trips
  every year, and the ratings already account for how good they are.
- Teams resting starters after clinching their playoff seed: this effect
  is real and big, but it only happens in about two games a season. Not
  enough to tune it reliably, so it's not in the model.

The main lesson: one good test isn't enough. Checking across several time
periods caught two features that would have looked great if I'd stopped
early.

## 2026 season

Each week I run `predict.py` and commit the picks before kickoff, so the
commit timestamps prove they came first. I run two versions side by side,
with and without team EPA, to see if the backtest difference holds up on
real games. `score.py` scores them against results and the Vegas line.

Sixteen games a week is a small sample, so it'll take most of the season
before the record says much.

## Files

    data.py        loads game schedules and results (nflverse)
    features.py    QB and team EPA values for each game
    elo.py         the model, tuning, testing, Vegas comparison
    predict.py     this week's predictions
    score.py       scores saved predictions
    find_qb.py     helper to look up QB ids for manual starter picks

    research/      the experiments behind the results above, including
                   the ideas that didn't make it. Full notes in
                   research/RESULTS.md

## Running it

    pip install nflreadpy pandas numpy scipy
    python elo.py          # tune and evaluate (takes a few minutes)
    python predict.py 4    # predict week 4
    python score.py        # score predictions so far

Data comes from [nflverse](https://github.com/nflverse) via `nflreadpy`.
