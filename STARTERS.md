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

| 4 | TB | Baker Mayfield | *(pending)* | ruled Out Friday (thumb); forecast was committed Sunday morning without checking the report |

Week 1: 1 real miss (ATL) plus 2 in-game injuries.
Week 2: 3 real misses (ATL, MIN, SEA) plus 1 in-game injury.
Week 4: 1 real miss (TB) — and it was avoidable.

## Overrides set before each week

| week | team | set to | reason |
|------|------|--------|--------|
| 3 | ATL | Michael Penix Jr. | named Week 3 starter |
| 3 | NYG | Jameis Winston | Dart knee injury |
| 3 | MIN | Kyler Murray | cleared concussion protocol, confirmed starter |

Lesson: check injury news for every team before running predict.py, not
just the teams I already had overrides for.

## Reading the injury report (from Week 5)

Every miss above except the in-game injuries was listed on the official
injury report before kickoff. Week 4 is the clearest case: Mayfield
(thumb), Caleb Williams (hamstring) and Jayden Daniels (elbow) were all
Out, and attempts alone would have started two of them.

predict.py now loads `nfl.load_injuries()` for the week and drops anyone
Out or Doubtful before ranking, flags Questionable, and refuses to let a
stale STARTER_OVERRIDES entry reinstate a quarterback who cannot play —
the WAS override was pointing at Jayden Daniels while he was ruled out.

Caveat: the report fills in through the week. Practice participation
appears midweek, firm Out/Questionable designations land Friday. Running
Tuesday gets a thinner report than running Saturday, so the script prints
how many statuses it found.
