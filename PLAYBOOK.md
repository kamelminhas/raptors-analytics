# Season playbook

How to run this project for a full season and turn it into something that gets you hired.

## How we split the work

- **Claude writes and maintains the code:** new metrics, charts, fixes, the quarter-season report. When something breaks or you want a new view, describe it and it gets built.
- **You do the analysis:** reading the numbers, watching the games, explaining *why*, deciding what matters, and writing it up. That is the part teams hire for.
- **Own the concepts anyway.** Interviewers will ask "how did you calculate that?" You don't need to write the code, but you should be able to explain every number in a sentence. The glossary at the bottom gives you that sentence for each one.

## The calendar

| When | Game | What to do |
|---|---|---|
| Oct 3–16 | Preseason | **Dry run.** Five games: Oct 3 vs MIA (Quebec City), Oct 10 vs LAC (Vancouver), Oct 13 vs NY, Oct 15 at NY, Oct 16 at DET. Run `python run.py game --date 2026-10-03` the morning after each one, practise the routine, and report anything that breaks. |
| Oct 21 | 1, vs CHI | Opening night. First real report. |
| Oct 30 – Nov 27 | | NBA Cup group stage (4 Raptors games). |
| Dec 3 | 21, at SA | **Quarter mark.** |
| Dec 4–12 | | Cup knockouts: at most 2 Raptors games. **Write the quarter-season roundup.** |
| mid-January | ~41 | Halfway. |
| Feb 18–25 | ~59 | All-Star break, no games. **Midseason report.** |
| early March | ~62 | Three-quarter mark. |
| Apr 11 | 82, at BKN | Finale. **End-of-season review** during the playoffs. |

The schedule has 15 back-to-backs, 40 home and 40 away games (plus the 2 December games still to be scheduled).

## After every game (30–45 minutes)

1. **Run it:** `python run.py game`.
2. **Read the report top to bottom** (5 min): efficiency → what decided it → game flow → players → shots.
3. **Pick one or two of the questions** in "Questions for your notes" and chase the answer (10–20 min). Useful places to look:
   - the play-by-play (hover the game-flow line)
   - the box score
   - the condensed game or highlights
4. **Write the notes file** it created in `notes/2026-27/` (10 min):
   - **Headline:** one sentence, what happened and why.
   - **What I noticed:** 2–3 bullets. Each one is a number from the report plus your explanation.
   - **What to watch next game:** one prediction, with the reason.
5. **Run it again,** and your notes appear in the report.
6. **Optional:** post one chart (the PNGs are in `reports/2026-27/img/`) with a two-sentence caption.

Skipping a game is fine; skipping a month is not. A short note on every game beats a long essay on a few.

## Weekly (1–2 hours)

- Open the season tracker:
  - What did the 10-game line do this week?
  - Whose "last 10" columns changed? Minutes show role changes; TS% shows hot and cold stretches.
- Pick **one deeper question** and write a short post: about 300 words and one chart. Starter ideas:
  - How is the offense different with Kawhi Leonard on the floor versus off it?
  - Is Collin Murray-Boyles' year-two jump real? Look at role (minutes, usage) versus efficiency.
  - How do the Raptors do on the second night of back-to-backs?
  - Which lineups are winning their minutes? (Ready once lineups are built; see the roadmap.)
  - Shot profile: are they getting more corner threes and rim attempts than last season?

## Six habits of a good analyst

1. **"Compared to what?"** A number means nothing without a baseline: season average, league average, or last season. Every report shows one; always say which you're using.
2. **Per possession, not per game.** A fast-paced game inflates everything. Ratings (per 100 possessions) take pace out.
3. **Small samples lie.** A true 35% three-point shooter shoots somewhere between about 22% and 48% over 50 attempts, just by chance.
   - Why: it's a binomial. The standard deviation is √(0.35 × 0.65 / 50) ≈ 6.7 percentage points, and ±2 SD covers roughly 95%.
   - Before calling anyone "hot", count the attempts.
4. **Separate what a team controls from what it doesn't.** On any given night, opponent three-point percentage is mostly luck. How many threes you allow, and how open they are, is defense.
5. **Luck regresses.** Teams that win more games than their point differential predicts (Pythagorean record) usually drift back toward it. So do close-game records.
6. **Numbers say what; film says why.** The report flags what was unusual. Your notes explain it. That second step is what makes it analysis.

## The quarter-season roundup (write it Dec 4–12)

**Deliverable:** one polished report or web page (the season tracker plus your writing), and one post with the three best charts.

1. **How good are they, really?**
   - Record against Pythagorean record, and net rating against SRS (which adjusts for schedule).
   - If the record and the point differential disagree, which do you believe, and why?
2. **Identity.** Their four-factor ranks on offense and defense this season, against 2025-26 (offense 15th, defense 5th). What changed with Leonard in and Brandon Ingram out?
3. **The Kawhi effect.**
   - Team results and ratings with and without him, then his on/off numbers once lineups are built.
   - Workload: minutes, back-to-backs, games missed.
4. **Development.** Murray-Boyles (year two), Ja'Kobe Walter, and rookie Allen Graves. Track role (minutes, usage) and efficiency, and whether each is trending up.
5. **Schedule.** Strength of opponents so far against what's left.
6. **What's next.** A projection for the rest of the season, with uncertainty, and the three things to watch.
7. **Methods appendix.** How every number was computed (the glossary below). Recruiters notice this.

## What gets built next (code roadmap)

| When | What | Why it matters |
|---|---|---|
| Preseason | Fixes from your dry run | Smooth routine before it counts |
| By mid-November | **Lineups and on/off.** Reconstruct who was on the floor from the substitutions. | 5-man lineup ratings; each player's impact on team net rating |
| By Dec 3 | **Season projection.** Simulate the rest of the schedule thousands of times from SRS ratings. | "X% chance of a top-6 seed", with honest uncertainty |
| January | **Your own win-probability model.** Logistic regression on score margin and time left, trained on last season's play-by-play. | A model you can explain end to end: ideal interview material |
| All-Star break | **Shooting regression.** How much to trust a hot start (beta-binomial shrinkage). | Separating skill from noise, the core of player evaluation |
| April | **Season review,** including checking whether the projections were calibrated | Showing you evaluate your own work |

## Turning it into a portfolio

- **GitHub:** make this project a public repository and turn on GitHub Pages (steps in the README). The site with every game report is the link for your résumé.
- **Posting:** once a week on LinkedIn and/or X, share one chart and two or three sentences of insight. Consistency beats volume.
- **Flagship pieces:** the quarter roundup (December), midseason report (February) and season review (April). These are what you'll send with applications.
- **Résumé line:** once it's true, something like "Built a Python pipeline analyzing every 2026-27 Toronto Raptors game (possession-based ratings, four-factor regression, shot profiles); published 80+ game reports and three season reviews."
- **Where it points:** MLSE (the Raptors' owner) and Raptors 905 are the obvious fits. Beyond them:
  - Hamilton's own teams: the Tiger-Cats and Forge FC.
  - Team analytics departments across leagues.
  - Sports data companies.

## Glossary: one-line answers

| Stat | How it's calculated | In words |
|---|---|---|
| Possessions | FGA + 0.4·FTA − 1.07·(ORB share)·(missed FG) + TOV, averaged over both teams | Basketball-Reference's estimate of how many times each team had the ball |
| Offensive / defensive rating | 100 × points ÷ possessions | Points scored / allowed per 100 possessions |
| Net rating | ORtg − DRtg | Points per 100 possessions better than the opponent |
| Pace | 48 × possessions ÷ minutes | Possessions per 48 minutes |
| eFG% | (FGM + 0.5·3PM) ÷ FGA | Field-goal % where a three counts 1.5 times as much as a two |
| TOV% | TOV ÷ (FGA + 0.44·FTA + TOV) | Share of plays that end in a turnover |
| ORB% | ORB ÷ (ORB + opponent DRB) | Share of your own misses you rebound |
| FT/FGA | FTM ÷ FGA | Free throws made per shot attempt |
| Four-factor model | OLS of net rating on the four team-minus-opponent gaps, ~4,900 team-games, R² ≈ 0.95 | Turns "we won the rebounding battle by 10 points of ORB%" into points per 100 possessions |
| TS% | PTS ÷ (2 × (FGA + 0.44·FTA)) | Scoring efficiency, counting threes and free throws |
| USG% | share of team plays (shots, free-throw trips, turnovers) a player used while on the floor | How much of the offence runs through him |
| Game Score | PTS + 0.4·FG − 0.7·FGA − 0.4·(FTA−FT) + 0.7·ORB + 0.3·DRB + STL + 0.7·AST + 0.7·BLK − 0.4·PF − TOV | John Hollinger's single number for a box-score line; about 10 is an average game |
| Pythagorean record | PTS¹⁴ ÷ (PTS¹⁴ + PA¹⁴) × games | The record the point differential "deserves" |
| SRS | Least-squares solution of margin = rating(home) − rating(away) + home court, over every game | Average margin, adjusted for opponent strength |
| Clutch time | Last 5 minutes of the 4th quarter or overtime, score within 5 | The NBA's own definition |
