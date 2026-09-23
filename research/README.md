# Research

The tests behind the choices in the model. Every test works the same way:
tune on earlier seasons, then score seasons the model hasn't seen. Since
one test period can get lucky, I check six different stretches of seasons
(2004-09, 2010-15, 2016-18, 2019-21, 2022-25, and 2019-25).

Scores are log loss (lower is better). "Diff" is how much the change
improved it.

## Kept

**QB adjustment** (`qb.py`). Each QB gets a rating from his recent EPA per
dropback, and the model adds the difference between the two starters.

| stretch   | no QB  | with QB | diff    |
|-----------|--------|---------|---------|
| 2004-2009 | 0.6249 | 0.6212  | +0.0036 |
| 2010-2015 | 0.6265 | 0.6186  | +0.0080 |
| 2016-2018 | 0.6271 | 0.6211  | +0.0060 |
| 2019-2021 | 0.6421 | 0.6300  | +0.0121 |
| 2022-2025 | 0.6339 | 0.6280  | +0.0059 |
| 2019-2025 | 0.6374 | 0.6302  | +0.0071 |

Helped every time. I also tried comparing each starter to his team's usual
QB. Tuning always turned that off, so I use the raw ratings.

**Raw QB fix** (`raw_qb_retune.py`). I'd decided on raw QB ratings, but the
model code still had the team-comparison version on. This script re-checks
everything after fixing it.

## Tested, basically no effect

**Team EPA** (`epa.py`). EPA per play, offense and defense, on top of QB:

| stretch   | QB only | QB + EPA | diff    |
|-----------|---------|----------|---------|
| 2004-2009 | 0.6193  | 0.6189   | +0.0004 |
| 2010-2015 | 0.6185  | 0.6186   | -0.0001 |
| 2016-2018 | 0.6214  | 0.6215   | -0.0001 |
| 2019-2021 | 0.6295  | 0.6298   | -0.0003 |
| 2022-2025 | 0.6280  | 0.6275   | +0.0005 |
| 2019-2025 | 0.6286  | 0.6284   | +0.0002 |

With the old QB version, EPA helped a little every time. With raw QB
ratings, it's a wash. It's still in the model at a small weight, and I
track it live to see if that changes.

## Tested, not kept

**Home-field advantage that changes over time** (`home_field.py`). Looked
good on 2019-2025, but only because 2019-2021 had unusually few home wins.
No real help in the other stretches.

**Travel distance and rest days.** Home teams win more when the visitor
travels farther, but that's because the same far-away teams make those
trips every year, and the ratings already know how good they are.

**Teams resting starters after clinching.** Real and big, but only about
two games a season, which isn't enough to tune it reliably.

## Running

    python research/qb.py windows
    python research/epa.py windows
    python research/raw_qb_retune.py
    python research/home_field.py      (slow, about 10 minutes)
