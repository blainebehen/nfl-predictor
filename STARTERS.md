# Starter log, 2026

For games that haven't been played, the model has to guess each team's
starting QB (see predict.py). This tracks where the guess was wrong, so
it's clear which predictions were made with the wrong quarterback.

"Actual" is the QB with the most pass attempts in the game.

## Misses

| week | team | predicted with | actual starter | why |
|------|------|----------------|----------------|-----|
| 1 | ATL | Tua Tagovailoa | Cooper Rush | Rush named starter; Penix not ready |
| 1 | SEA | Sam Darnold | Drew Lock | Darnold hurt early in the game (2 attempts) |
| 1 | MIN | Kyler Murray | Carson Wentz | Murray hurt early in the game (5 attempts, concussion) |
| 2 | ATL | Tua Tagovailoa | Cooper Rush | Rush started again |
| 2 | MIN | Kyler Murray | Carson Wentz | Murray out, concussion protocol |
| 2 | SEA | Sam Darnold | Drew Lock | Darnold out, oblique injury |
| 2 | NYG | Jaxson Dart | Jameis Winston | Dart started, hurt early (5 attempts, knee) |

Week 1: 1 real miss (ATL) plus 2 in-game injuries.
Week 2: 3 real misses (ATL, MIN, SEA) plus 1 in-game injury.

## Overrides set before each week

| week | team | set to | reason |
|------|------|--------|--------|
| 3 | ATL | Michael Penix Jr. | named Week 3 starter |
| 3 | NYG | Jameis Winston | Dart knee injury |
| 3 | MIN | Kyler Murray | cleared concussion protocol, confirmed starter |

Lesson: check injury news for every team before running predict.py, not
just the teams I already had overrides for.
