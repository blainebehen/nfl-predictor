# Starter log, 2026

For games that haven't been played, the model has to guess each team's
starting QB (see predict.py). This tracks where the guess was wrong, so
it's clear which predictions were made with the wrong quarterback.

"Actual" is the QB with the most pass attempts in the game.

Through Week 6: **13 wrong out of 152 team-game forecasts, 8.6%.** Ten of
those are real misses — the predicted QB never played. The other three
are in-game injuries, where he did start and got hurt early; nothing a
pre-game guess could have done about those.

Week 5 isn't here because no forecast was committed for it.

## Misses

| week | team | predicted with | actual starter | att | why |
|------|------|----------------|----------------|-----|-----|
| 1 | ATL | Tua Tagovailoa | Cooper Rush | 22 | Rush named starter; Penix not ready |
| 1 | SEA | Sam Darnold | Drew Lock | 22 | in-game: Darnold hurt early (2 attempts) |
| 1 | MIN | Kyler Murray | Carson Wentz | 19 | in-game: Murray hurt early (5 attempts, concussion) |
| 2 | ATL | Tua Tagovailoa | Cooper Rush | 17 | Rush started again |
| 2 | MIN | Kyler Murray | Carson Wentz | 20 | Murray out, concussion protocol |
| 2 | SEA | Sam Darnold | Drew Lock | 26 | Darnold out, oblique injury |
| 2 | NYG | Jaxson Dart | Jameis Winston | 27 | in-game: Dart hurt early (5 attempts, knee) |
| 3 | WAS | Jayden Daniels | Marcus Mariota | 31 | Daniels out; no override set |
| 3 | CHI | Caleb Williams | Case Keenum | 34 | Williams out (hamstring) |
| 4 | CHI | Case Keenum | Tyson Bagent | 34 | third CHI starter in three weeks |
| 4 | TB | Baker Mayfield | Jalon Daniels | 27 | ruled Out Friday (thumb); forecast committed Sunday morning without checking the report |
| 6 | CHI | Tyson Bagent | Caleb Williams | 36 | Williams back from the hamstring |
| 6 | PHI | Jalen Hurts | Tanner McKee | 45 | Hurts out |

| week | forecasts | real misses | in-game injuries |
|------|-----------|-------------|------------------|
| 1 | 32 | 1 | 2 |
| 2 | 32 | 3 | 1 |
| 3 | 32 | 2 | 0 |
| 4 | 28 | 2 | 0 |
| 6 | 28 | 2 | 0 |

## Chicago is the whole problem

Four of the ten real misses are one team. Chicago has started three
different quarterbacks in the weeks I forecast — Williams, then Keenum,
then Bagent, then Williams again — and the model was wrong every single
week from 3 on, including Week 6, when it picked the man who had started
the most recent game and Williams returned instead.

That is the known failure mode of the recency rule, stated in predict.py
and then observed exactly as described: a fill-in gets picked after the
starter comes back. The 2x fill-in flag did not fire in Week 6 because
Bagent's attempts were close enough to Williams's by then.

No rule based on past attempts fixes this. A team cycling quarterbacks
weekly is decided by injury news, not by history.

## Overrides set before each week

| week | team | set to | reason |
|------|------|--------|--------|
| 3 | ATL | Michael Penix Jr. | named Week 3 starter |
| 3 | NYG | Jameis Winston | Dart knee injury |
| 3 | MIN | Kyler Murray | cleared concussion protocol, confirmed starter |

The overrides that worked were the ones set from injury news. The ones
that went stale were worse than nothing — an override beats every rule in
predict.py, so the WAS entry kept pointing at Jayden Daniels while he was
ruled out, which is what produced the Week 3 miss.

## Reading the injury report (from Week 5)

Every real miss above was listed on the official injury report before
kickoff. Week 4 is the clearest case: Mayfield (thumb), Caleb Williams
(hamstring) and Jayden Daniels (elbow) were all Out, and attempts alone
would have started two of them.

predict.py now loads `nfl.load_injuries()` for the week and drops anyone
Out or Doubtful before ranking, flags Questionable, and refuses to let a
stale STARTER_OVERRIDES entry reinstate a quarterback who cannot play.

Caveat, and it bit immediately: the report fills in through the week.
Practice participation appears midweek, firm Out/Questionable
designations land Friday. The Week 6 run on Thursday evening printed "no
week-6 QB injury report yet" and fell back to attempts alone, which is
how PHI went in with Hurts. The filter is only worth having if the run
happens late enough for nflverse to have the data.

So the schedule is a tradeoff, not a best practice. Running early
protects the commit timestamp and gets a thin injury report; running
Saturday gets the full report and leaves almost no margin before Sunday.
Friday is the compromise.
