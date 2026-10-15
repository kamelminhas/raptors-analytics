"""Project settings. Everything you might want to change lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # downloaded data (safe to delete; it re-downloads)
REPORTS_DIR = ROOT / "docs"    # generated HTML reports + PNG charts
NOTES_DIR = ROOT / "notes"        # YOUR written analysis, one markdown file per game

TEAM = "TOR"                      # ESPN team abbreviation
TEAM_NAME = "Raptors"
TEAM_CITY = "Toronto"

# Seasons are labelled by the year they END in: 2027 = the 2026-27 season.
CURRENT_SEASON = 2027

# League-wide data older than this (hours) is re-downloaded before a report is built.
ARCHIVE_MAX_AGE_HOURS = 6

# Printed in the corner of every exported PNG chart. Change to your name or social handle.
CREDIT = "Kamel Minhas"
DATA_CREDIT = "Data: ESPN via sportsdataverse"

# Early in the season a team or player has few games, so "vs average" comparisons
# use last season's numbers until this many games have been played this season.
MIN_GAMES_FOR_SEASON_BASELINE = 5


def season_label(season: int) -> str:
    """2027 -> '2026-27'"""
    return f"{season - 1}-{str(season)[-2:]}"
