# Raptors Analytics

Game-by-game analysis of the Toronto Raptors' 2026-27 season. After every game, one command pulls the data and builds a report:

- **Efficiency:** points scored and allowed per 100 possessions, vs. the season average.
- **What decided it:** the four factors, each converted to points by a regression fitted on two seasons of NBA games.
- **Game flow:** the lead after every basket, the biggest runs, and clutch time.
- **Players:** the box score plus true shooting, usage, Game Score, and how each player compared with his normal.
- **Shot profile:** a shot chart and shooting by zone, vs. the season and the league.
- **Your analysis:** what stood out statistically, questions to investigate, and your own written notes.

A season tracker updates alongside it: record, ratings and NBA ranks, Pythagorean record, SRS, rolling form, the roster, and upcoming games.

Every chart is interactive in the report and is also saved as a PNG ready to post.

---

## One-time setup

1. **Install Python 3.10 or newer** from [python.org](https://www.python.org/downloads/). On Windows, tick "Add Python to PATH".
2. Open a terminal in this folder and install the packages:
   ```
   python -m pip install -r requirements.txt
   ```
   (On a Mac, use `python3` everywhere this README says `python`.)
3. Run setup. It downloads last season's data (for comparisons), checks that ESPN is reachable, and builds the season page:
   ```
   python run.py setup
   ```
4. See what you'll get: this builds sample reports from last season's final games and the 7-game Cleveland series.
   ```
   python run.py demo
   ```

## After every game

```
python run.py game
```

That's it. It finds the most recent Raptors game, builds its report, updates the season tracker, and opens the report in your browser. Then:

1. Read the report top to bottom (5 minutes).
2. Open the notes file it created in `notes/2026-27/` and write your analysis. The questions from the report are already in it.
3. Run `python run.py game` again, and your notes appear in the report.

The full routine, and how to build it into a portfolio, is in **[PLAYBOOK.md](PLAYBOOK.md)**.

## Commands

| Command | What it does |
|---|---|
| `python run.py game` | Report for the most recent game |
| `python run.py game --date 2026-10-21` | The game on a given date |
| `python run.py game --id 401909837` | One game, by ESPN game id (in the ESPN game URL) |
| `python run.py season` | Rebuild the season tracker |
| `python run.py rebuild` | Re-render every report this season (after you change a template) |
| `python run.py demo` | Sample reports from last season |
| `python run.py setup` | First-time setup |

Add `--season 2026` to any command to work on 2025-26 (seasons are named by the year they end). Add `--refresh` to force fresh data, and `--no-open` to skip opening the browser.

## What's where

```
run.py                  the commands above
config.py               settings: team, season, your name on the charts (CREDIT)
raptors/
  data.py               downloading + cleaning (archive and ESPN), one format for both
  metrics.py            possessions, ratings, four factors (+ the regression), player stats, SRS
  pbp.py                play-by-play: shot zones, game flow, runs, clutch time
  charts.py             every chart (Altair / Vega-Lite), HTML + PNG
  insights.py           "what stands out" rules and the questions for your notes
  report.py             assembles the reports
  templates/            the HTML pages
notes/2026-27/          YOUR analysis, one markdown file per game
reports/                the generated site: open reports/index.html
data/                   downloaded data (safe to delete; it re-downloads)
```

## Where the data comes from

- **League-wide box scores and play-by-play:** the [sportsdataverse](https://github.com/sportsdataverse) archive, which republishes ESPN's data as files and refreshes through the season.
- **The game you're reporting on:** straight from ESPN's public game feed, available minutes after the final buzzer. If ESPN can't be reached, the archive's GitHub copy is used instead, which can lag by up to a day.

Everything was checked against Basketball-Reference on last season's data. Team ratings, pace, the four factors, the Pythagorean record, SRS and the league ranks all match, to within 0.1. Shot counts from the play-by-play match the official box score in 97% of team-games; the rest are feed quirks.

One difference to know about: this project's free-throw factor is **made** free throws per field-goal attempt (Dean Oliver's original definition). Some sites, including Basketball-Reference box scores, show **attempts** per field-goal attempt, so their number is higher.

## Put it online (optional, recommended)

The `reports/` folder is a complete static website. To publish it free with GitHub Pages:

1. In `config.py`, change `REPORTS_DIR = ROOT / "reports"` to `REPORTS_DIR = ROOT / "docs"` (GitHub Pages can publish a folder named `docs`), then run `python run.py rebuild`.
2. Push the project to a public GitHub repository.
3. On GitHub: Settings → Pages → "Deploy from a branch" → branch `main`, folder `/docs`.

You get a public URL with every game report, which is the link to put on your résumé.

## If something goes wrong

- **"Missing a package"**: run `python -m pip install -r requirements.txt` again.
- **"Game … isn't final yet"**: the game is still on, or ESPN hasn't marked it final. Try again in 15 minutes.
- **ESPN not reachable**: reports still build from the archive's copy, usually by the next morning.
- **Charts don't show**: each report folder has its own copy of the charting library (`vega-bundle.js`, next to the pages; Safari only loads files from a page's own folder). If it's missing, re-run the command that built the page. The PNGs in `reports/<season>/img/` always work.
- **PNG charts use a plain font**: setup couldn't download the Barlow font. Download it from Google Fonts and put the `.ttf` files in `assets/fonts/`.
