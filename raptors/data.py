"""Getting the data.

Two sources, one format:

1. The sportsdataverse archive (github.com/sportsdataverse): every NBA game's box score
   and play-by-play, originally from ESPN, published as parquet files and refreshed
   through the season. Used for league-wide context (all 30 teams).

2. One game's full ESPN feed: fetched straight from ESPN right after the game (works on
   your own computer), or from the archive's GitHub mirror if ESPN can't be reached.

Both are converted into the same three tables so the rest of the code never has to care
where a number came from:

    team_box    one row per team per game
    player_box  one row per player per game
    pbp         one row per play (with shot locations)
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config

ARCHIVE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
ARCHIVE_FILES = {
    "team_box": "espn_nba_team_boxscores/team_box_{season}.parquet",
    "player_box": "espn_nba_player_boxscores/player_box_{season}.parquet",
    "pbp": "espn_nba_pbp/play_by_play_{season}.parquet",
    "schedule": "espn_nba_schedules/nba_schedule_{season}.parquet",
}
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
ESPN_TEAM_SCHEDULE = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/"
                      "{team}/schedule?season={season}&seasontype={stype}")
GITHUB_GAME_JSON = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-raw/main/nba/json/final/{game_id}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (raptors-analytics student project)"}
EASTERN = ZoneInfo("America/New_York")

NBA_TEAM_IDS = set(range(1, 31))   # ESPN ids 1-30 are the 30 NBA teams (All-Star teams use others)
REGULAR, PLAYOFFS, PLAY_IN, PRESEASON = 2, 3, 5, 1
SEASON_TYPE_NAMES = {PRESEASON: "Preseason", REGULAR: "Regular season", PLAYOFFS: "Playoffs", PLAY_IN: "Play-in"}

TEAM_BOX_COLS = ["game_id", "season", "season_type", "game_date", "team_id", "team_abbr", "team_name",
                 "home_away", "pts", "opp_id", "opp_abbr", "opp_pts", "win",
                 "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
                 "ast", "stl", "blk", "tov", "tov_team", "pf", "pts_paint", "pts_fb", "largest_lead"]
PLAYER_BOX_COLS = ["game_id", "season", "season_type", "game_date", "team_id", "team_abbr", "opp_abbr",
                   "home_away", "athlete_id", "name", "short_name", "jersey", "position", "starter",
                   "dnp", "reason", "min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                   "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pm"]
PBP_COLS = ["game_id", "season", "season_type", "seq", "period", "clock", "t", "type_text", "text",
            "team_id", "a1", "a2", "a3", "home_score", "away_score", "scoring", "shooting",
            "value", "pts_att", "x", "y", "home_id", "away_id"]


class GameNotAvailable(Exception):
    """The game can't be fetched yet (not finished, or no source reachable)."""


def log(msg: str) -> None:
    print(f"  · {msg}", flush=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _snake(name: str) -> str:
    """'fieldGoalsMade' -> 'field_goals_made'"""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _num(v, default=0.0):
    """ESPN numbers arrive as strings like '12', '+7', '-3', '--', '36:12'."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return default if (isinstance(v, float) and math.isnan(v)) else v
    s = str(v).strip()
    if s in ("", "--", "-"):
        return default
    if ":" in s:                                    # minutes as mm:ss
        m, sec = s.split(":", 1)
        return float(m) + float(sec) / 60
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return default


def _int(v):
    try:
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else int(float(v))
    except (TypeError, ValueError):
        return None


def _get(obj, path: str, default=None):
    """Read 'a.b.0.c' from nested JSON. Also accepts pre-flattened keys ('type.text'),
    because the GitHub mirror stores plays flattened while ESPN's live feed nests them."""
    if isinstance(obj, dict) and path in obj:
        v = obj[path]
        return default if v is None else v
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            i = int(part)
            cur = cur[i] if i < len(cur) else None
        else:
            return default
        if cur is None:
            return default
    return cur


def clock_seconds(clock: str) -> float:
    """'10:34' -> 634.0, '45.2' -> 45.2"""
    s = str(clock or "0").strip()
    try:
        if ":" in s:
            parts = s.split(":")          # a handful of feed rows look like '1:39:00'
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except ValueError:
        return 0.0


def elapsed_seconds(period: int, clock: str) -> float:
    """Seconds since tip-off. Quarters are 12:00, overtimes 5:00."""
    left = clock_seconds(clock)
    if period <= 4:
        return (period - 1) * 720 + (720 - left)
    return 2880 + (period - 5) * 300 + (300 - left)


def game_length_seconds(periods: int) -> int:
    return 2880 + max(0, periods - 4) * 300


def _eastern_date(iso_utc: str) -> str:
    """'2026-10-21T23:30Z' -> '2026-10-21' (US Eastern calendar date, like the NBA uses)"""
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        return dt.astimezone(EASTERN).date().isoformat()
    except ValueError:
        return str(iso_utc)[:10]


# ---------------------------------------------------------------------------
# 1. the archive (league-wide)
# ---------------------------------------------------------------------------
def fetch_archive(kind: str, season: int, refresh: bool = False) -> pd.DataFrame | None:
    """Download (or reuse the cached copy of) one archive file. Returns None when the
    file doesn't exist yet, e.g. a season's box scores before opening night."""
    path = config.DATA_DIR / "archive" / f"{kind}_{season}.parquet"
    current = season >= config.CURRENT_SEASON
    max_age = config.ARCHIVE_MAX_AGE_HOURS * 3600 if current else float("inf")
    if path.exists() and not refresh and (time.time() - path.stat().st_mtime) < max_age:
        return pd.read_parquet(path)
    url = f"{ARCHIVE}/{ARCHIVE_FILES[kind].format(season=season)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=180)
        if r.status_code == 404:  # not published yet (e.g. box scores before opening night)
            return pd.read_parquet(path) if path.exists() else None
        r.raise_for_status()
        log(f"downloaded {kind} {config.season_label(season)} from the archive")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    except requests.RequestException as e:
        if path.exists():
            log(f"couldn't refresh {kind} ({type(e).__name__}); using the cached copy")
            return pd.read_parquet(path)
        raise
    return pd.read_parquet(path)


def _team_box_from_archive(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "game_id": df["game_id"].astype("int64"),
        "season": df["season"].astype(int),
        "season_type": df["season_type"].astype(int),
        "game_date": df["game_date"].astype(str).str[:10],
        "team_id": df["team_id"].astype(int),
        "team_abbr": df["team_abbreviation"].astype(str),
        "team_name": df["team_display_name"].astype(str),
        "home_away": df["team_home_away"].astype(str),
        "pts": df["team_score"].astype(int),
        "opp_id": df["opponent_team_id"].astype(int),
        "opp_abbr": df["opponent_team_abbreviation"].astype(str),
        "opp_pts": df["opponent_team_score"].astype(int),
        "win": df["team_winner"].astype(bool),
    })
    cols = {"fgm": "field_goals_made", "fga": "field_goals_attempted",
            "fg3m": "three_point_field_goals_made", "fg3a": "three_point_field_goals_attempted",
            "ftm": "free_throws_made", "fta": "free_throws_attempted",
            "oreb": "offensive_rebounds", "dreb": "defensive_rebounds", "reb": "total_rebounds",
            "ast": "assists", "stl": "steals", "blk": "blocks", "tov_team": "team_turnovers",
            "pf": "fouls", "pts_paint": "points_in_paint", "pts_fb": "fast_break_points",
            "largest_lead": "largest_lead"}
    for new, old in cols.items():
        out[new] = pd.to_numeric(df[old], errors="coerce").fillna(0) if old in df else 0
    tov = df["total_turnovers"] if "total_turnovers" in df else df["turnovers"]
    out["tov"] = pd.to_numeric(tov, errors="coerce").fillna(pd.to_numeric(df["turnovers"], errors="coerce"))
    return out[TEAM_BOX_COLS]


def _player_box_from_archive(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "game_id": df["game_id"].astype("int64"),
        "season": df["season"].astype(int),
        "season_type": df["season_type"].astype(int),
        "game_date": df["game_date"].astype(str).str[:10],
        "team_id": df["team_id"].astype(int),
        "team_abbr": df["team_abbreviation"].astype(str),
        "opp_abbr": df["opponent_team_abbreviation"].astype(str),
        "home_away": df["home_away"].astype(str),
        "athlete_id": pd.to_numeric(df["athlete_id"], errors="coerce").astype("Int64"),
        "name": df["athlete_display_name"].astype(str),
        "short_name": df["athlete_short_name"].astype(str),
        "jersey": df["athlete_jersey"].astype(str),
        "position": df["athlete_position_abbreviation"].astype(str),
        "starter": df["starter"].astype(bool),
        "reason": df["reason"].fillna("").astype(str),
        "min": pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0),
    })
    stat_cols = {"pts": "points", "fgm": "field_goals_made", "fga": "field_goals_attempted",
                 "fg3m": "three_point_field_goals_made", "fg3a": "three_point_field_goals_attempted",
                 "ftm": "free_throws_made", "fta": "free_throws_attempted",
                 "oreb": "offensive_rebounds", "dreb": "defensive_rebounds", "reb": "rebounds",
                 "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers", "pf": "fouls"}
    for new, old in stat_cols.items():
        out[new] = pd.to_numeric(df[old], errors="coerce").fillna(0)
    out["pm"] = df["plus_minus"].map(lambda v: _num(v, 0.0))
    out["dnp"] = df["did_not_play"].astype(bool) | (out["min"] <= 0)
    out = out[out["athlete_id"].notna()]
    return out[PLAYER_BOX_COLS]


def _pbp_from_archive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_id", "game_play_number"])
    out = pd.DataFrame({
        "game_id": df["game_id"].astype("int64"),
        "season": df["season"].astype(int),
        "season_type": df["season_type"].astype(int),
        "seq": df["game_play_number"].astype(int),
        "period": df["period_number"].astype(int),
        "clock": df["clock_display_value"].astype(str),
        "type_text": df["type_text"].fillna("").astype(str),
        "text": df["text"].fillna("").astype(str),
        "team_id": pd.to_numeric(df["team_id"], errors="coerce").astype("Int64"),
        "a1": pd.to_numeric(df["athlete_id_1"], errors="coerce").astype("Int64"),
        "a2": pd.to_numeric(df["athlete_id_2"], errors="coerce").astype("Int64"),
        "a3": pd.to_numeric(df["athlete_id_3"], errors="coerce").astype("Int64"),
        "home_score": pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int),
        "away_score": pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int),
        "scoring": df["scoring_play"].fillna(False).astype(bool),
        "shooting": df["shooting_play"].fillna(False).astype(bool),
        "value": pd.to_numeric(df["score_value"], errors="coerce").fillna(0).astype(int),
        "pts_att": pd.to_numeric(df["points_attempted"], errors="coerce"),
        "x": pd.to_numeric(df["coordinate_x_raw"], errors="coerce"),
        "y": pd.to_numeric(df["coordinate_y_raw"], errors="coerce"),
        "home_id": df["home_team_id"].astype(int),
        "away_id": df["away_team_id"].astype(int),
    })
    return _finish_pbp(out)


def _finish_pbp(out: pd.DataFrame) -> pd.DataFrame:
    """Shared clean-up for both sources: bad coordinates -> NaN, elapsed game time."""
    bad = (out["x"].abs() > 1000) | (out["y"].abs() > 1000)
    out.loc[bad, ["x", "y"]] = float("nan")
    out["t"] = [elapsed_seconds(p, c) for p, c in zip(out["period"], out["clock"])]
    # points attempted is missing in some feeds: infer it for field goals from the text
    out["pts_att"] = pd.to_numeric(out["pts_att"], errors="coerce").astype(float)
    fga = out["shooting"] & ~out["type_text"].str.contains("Free Throw", case=False)
    missing = out["pts_att"].isna() & fga
    if missing.any():
        out.loc[missing, "pts_att"] = infer_points_attempted(out.loc[missing]).astype(float)
    return out[PBP_COLS]


def infer_points_attempted(shots: pd.DataFrame) -> pd.Series:
    """2 or 3, for feeds without a points-attempted field: the text when it says so,
    otherwise the shot location (beyond the 23'9" arc, or in a corner past 22 ft).
    Rim position calibrated in pbp.py (x = 25, y = 0.84)."""
    text = shots["text"].str.lower()
    dist_xy = ((shots["x"] - 25.0) ** 2 + (shots["y"] - 0.84) ** 2) ** 0.5
    stated = pd.to_numeric(text.str.extract(r"(\d+)-foot")[0], errors="coerce")
    corner = ((shots["x"] - 25.0).abs() >= 21.5) & (shots["y"] <= 9.8)
    three = (text.str.contains("three point") | text.str.contains("heave") | (dist_xy >= 23.4)
             | corner | (stated >= 24))
    two_said = text.str.contains("two point")
    return pd.Series(np.where(three & ~two_said, 3.0, 2.0), index=shots.index)


def _schedule_from_archive(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "game_id": df["game_id"].astype("int64"),
        "start_utc": pd.to_datetime(df["date"], utc=True, errors="coerce"),
        "game_date": df["game_date"].astype(str).str[:10],
        "season_type": df["season_type"].astype(int),
        "home_id": pd.to_numeric(df["home_id"], errors="coerce").astype("Int64"),
        "away_id": pd.to_numeric(df["away_id"], errors="coerce").astype("Int64"),
        "home_abbr": df["home_abbreviation"].astype(str),
        "away_abbr": df["away_abbreviation"].astype(str),
        "venue": df["venue_full_name"].fillna("").astype(str),
        "neutral": df["neutral_site"].fillna(False).astype(bool),
        "note": df["notes_headline"].fillna("").astype(str),
        "completed": df["status_type_completed"].fillna(False).astype(bool),
    })
    return out.sort_values("start_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. one game's full ESPN feed
# ---------------------------------------------------------------------------
def _espn_get(url: str, timeout: int = 30) -> requests.Response:
    """GET from ESPN's public API. Some networks and ESPN's CDN reject unfamiliar user
    agents, so this tries ours first, then requests' default. Raises with the reason."""
    reasons = []
    for headers in (HEADERS, None):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            reasons.append(f"{type(e).__name__}: {str(e)[:140]}")
            continue
        if r.status_code == 200:
            return r
        reasons.append(f"HTTP {r.status_code}")
    raise requests.RequestException(" / ".join(dict.fromkeys(reasons)))


def check_espn() -> tuple[bool, str]:
    """Can this computer reach ESPN's game feed? Returns (ok, reason if not)."""
    try:
        _espn_get(ESPN_SUMMARY.format(game_id=401869384), timeout=20)
        return True, ""
    except requests.RequestException as e:
        return False, str(e)


def _game_path(game_id) -> Path:
    return config.DATA_DIR / "games" / f"{game_id}.json"


def _is_final(j: dict) -> bool:
    return bool(_get(j, "header.competitions.0.status.type.completed", False))


def fetch_game_json(game_id, refresh: bool = False) -> dict:
    """The full feed for one game (box score + every play). Cached after the first fetch."""
    path = _game_path(game_id)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for source, url in (("ESPN", ESPN_SUMMARY), ("GitHub mirror", GITHUB_GAME_JSON)):
        try:
            if source == "ESPN":
                r = _espn_get(url.format(game_id=game_id), timeout=45)
            else:
                r = requests.get(url.format(game_id=game_id), headers=HEADERS, timeout=45)
        except requests.RequestException as e:
            problems.append(f"{source}: {e}")
            continue
        if r.status_code != 200:
            problems.append(f"{source}: HTTP {r.status_code}")
            continue
        j = r.json()
        if not _is_final(j):
            problems.append(f"{source}: game isn't final yet")
            continue
        log(f"game {game_id}: fetched from {source}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(j), encoding="utf-8")
        return j
    raise GameNotAvailable(f"Couldn't get game {game_id} ({'; '.join(problems)}).")


@dataclass
class Game:
    """Everything about one game, parsed from its ESPN feed."""
    meta: dict
    team_box: pd.DataFrame
    player_box: pd.DataFrame
    pbp: pd.DataFrame


def parse_game(j: dict) -> Game:
    comp = _get(j, "header.competitions.0", {})
    season = int(_get(j, "header.season.year", 0))
    season_type = int(_get(j, "header.season.type", REGULAR))
    game_id = int(_get(j, "header.id", 0) or comp.get("id") or j.get("gameId"))
    game_date = _eastern_date(comp.get("date", ""))
    competitors = {str(c["id"]): c for c in comp.get("competitors", [])}
    home = next(c for c in competitors.values() if c.get("homeAway") == "home")
    away = next(c for c in competitors.values() if c.get("homeAway") == "away")

    # ---- team box --------------------------------------------------------
    rows = []
    for t in _get(j, "boxscore.teams", []):
        tid = str(_get(t, "team.id"))
        c = competitors.get(tid, {})
        stats = {}
        for s in t.get("statistics", []):
            name, val = s.get("name", ""), str(s.get("displayValue", ""))
            if "-" in name and "-" in val.lstrip("-"):
                (n1, n2), (v1, v2) = name.split("-", 1), val.split("-", 1)
                stats[_snake(n1)], stats[_snake(n2)] = _num(v1), _num(v2)
            else:
                stats[_snake(name)] = _num(val)
        rows.append({
            "game_id": game_id, "season": season, "season_type": season_type, "game_date": game_date,
            "team_id": int(tid), "team_abbr": _get(t, "team.abbreviation", ""),
            "team_name": _get(t, "team.displayName", ""),
            "home_away": c.get("homeAway") or t.get("homeAway", ""),
            "pts": int(_num(c.get("score"))), "win": bool(c.get("winner", False)),
            "fgm": stats.get("field_goals_made", 0), "fga": stats.get("field_goals_attempted", 0),
            "fg3m": stats.get("three_point_field_goals_made", 0),
            "fg3a": stats.get("three_point_field_goals_attempted", 0),
            "ftm": stats.get("free_throws_made", 0), "fta": stats.get("free_throws_attempted", 0),
            "oreb": stats.get("offensive_rebounds", 0), "dreb": stats.get("defensive_rebounds", 0),
            "reb": stats.get("total_rebounds", 0), "ast": stats.get("assists", 0),
            "stl": stats.get("steals", 0), "blk": stats.get("blocks", 0),
            "tov": stats.get("total_turnovers", stats.get("turnovers", 0)),
            "tov_team": stats.get("team_turnovers", 0), "pf": stats.get("fouls", 0),
            "pts_paint": stats.get("points_in_paint", 0), "pts_fb": stats.get("fast_break_points", 0),
            "largest_lead": stats.get("largest_lead", 0),
        })
    tb = pd.DataFrame(rows)
    other = tb[["team_id", "team_abbr", "pts"]].rename(columns={"team_id": "opp_id", "team_abbr": "opp_abbr", "pts": "opp_pts"})
    tb = pd.concat([tb, other.iloc[::-1].reset_index(drop=True)], axis=1)[TEAM_BOX_COLS]
    abbr = dict(zip(tb["team_id"], tb["team_abbr"]))

    # ---- player box ------------------------------------------------------
    prow = []
    for tp in _get(j, "boxscore.players", []):
        tid = int(_get(tp, "team.id"))
        ha = competitors.get(str(tid), {}).get("homeAway", "")
        opp = next((a for i, a in abbr.items() if i != tid), "")
        for grp in tp.get("statistics", []):
            keys = grp.get("keys") or []
            for a in grp.get("athletes", []):
                v = dict(zip(keys, a.get("stats") or []))

                def pair(key):
                    raw = str(v.get(key, "0-0"))
                    return (_num(raw.split("-")[0]), _num(raw.split("-")[1])) if "-" in raw else (0, 0)
                fgm, fga = pair("fieldGoalsMade-fieldGoalsAttempted")
                fg3m, fg3a = pair("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
                ftm, fta = pair("freeThrowsMade-freeThrowsAttempted")
                minutes = _num(v.get("minutes"), 0.0)
                prow.append({
                    "game_id": game_id, "season": season, "season_type": season_type, "game_date": game_date,
                    "team_id": tid, "team_abbr": abbr.get(tid, ""), "opp_abbr": opp, "home_away": ha,
                    "athlete_id": _int(_get(a, "athlete.id")), "name": _get(a, "athlete.displayName", ""),
                    "short_name": _get(a, "athlete.shortName", ""), "jersey": str(_get(a, "athlete.jersey", "")),
                    "position": _get(a, "athlete.position.abbreviation", ""),
                    "starter": bool(a.get("starter", False)),
                    "dnp": bool(a.get("didNotPlay", False)) or minutes <= 0,
                    "reason": a.get("reason") or "", "min": minutes,
                    "pts": _num(v.get("points")), "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a,
                    "ftm": ftm, "fta": fta, "oreb": _num(v.get("offensiveRebounds")),
                    "dreb": _num(v.get("defensiveRebounds")), "reb": _num(v.get("rebounds")),
                    "ast": _num(v.get("assists")), "stl": _num(v.get("steals")), "blk": _num(v.get("blocks")),
                    "tov": _num(v.get("turnovers")), "pf": _num(v.get("fouls")), "pm": _num(v.get("plusMinus")),
                })
    pb = pd.DataFrame(prow, columns=PLAYER_BOX_COLS)
    pb["athlete_id"] = pb["athlete_id"].astype("Int64")

    # ---- play-by-play ----------------------------------------------------
    plays = j.get("plays") or []
    pr = []
    for i, p in enumerate(plays):
        pr.append({
            "game_id": game_id, "season": season, "season_type": season_type, "seq": i + 1,
            "period": int(_num(_get(p, "period.number"), 0)), "clock": str(_get(p, "clock.displayValue", "0:00")),
            "type_text": str(_get(p, "type.text", "")), "text": str(_get(p, "text", "")),
            "team_id": _int(_get(p, "team.id")),
            "a1": _int(_get(p, "participants.0.athlete.id")), "a2": _int(_get(p, "participants.1.athlete.id")),
            "a3": _int(_get(p, "participants.2.athlete.id")),
            "home_score": int(_num(_get(p, "homeScore"), 0)), "away_score": int(_num(_get(p, "awayScore"), 0)),
            "scoring": bool(_get(p, "scoringPlay", False)), "shooting": bool(_get(p, "shootingPlay", False)),
            "value": int(_num(_get(p, "scoreValue"), 0)), "pts_att": _get(p, "pointsAttempted"),
            "x": _get(p, "coordinate.x"), "y": _get(p, "coordinate.y"),
            "home_id": int(home["id"]), "away_id": int(away["id"]),
        })
    pbp = pd.DataFrame(pr)
    if len(pbp):
        for c in ("team_id", "a1", "a2", "a3"):
            pbp[c] = pbp[c].astype("Int64")
        for c in ("pts_att", "x", "y"):
            pbp[c] = pd.to_numeric(pbp[c], errors="coerce")
        pbp = _finish_pbp(pbp)
    else:
        pbp = pd.DataFrame(columns=PBP_COLS)

    periods = int(pbp["period"].max()) if len(pbp) else len(home.get("linescores") or []) or 4
    meta = {
        "game_id": game_id, "season": season, "season_type": season_type, "game_date": game_date,
        "start_utc": comp.get("date", ""),
        "home_id": int(home["id"]), "away_id": int(away["id"]),
        "home_abbr": _get(home, "team.abbreviation", ""), "away_abbr": _get(away, "team.abbreviation", ""),
        "home_name": _get(home, "team.displayName", ""), "away_name": _get(away, "team.displayName", ""),
        "linescores": {int(c["id"]): [int(_num(_get(ls, "displayValue", ls.get("value") if isinstance(ls, dict) else 0)))
                                     for ls in (c.get("linescores") or [])] for c in competitors.values()},
        "periods": periods,
        "status": _get(comp, "status.type.detail", "Final"),
        "venue": _get(j, "gameInfo.venue.fullName", ""),
        "attendance": int(_num(_get(j, "gameInfo.attendance"), 0)),
        "officials": [o.get("displayName", "") for o in (_get(j, "gameInfo.officials", []) or [])],
        "note": " · ".join(n.get("headline", "") for n in (comp.get("notes") or []) if n.get("headline")),
        "neutral": bool(comp.get("neutralSite", False)),
    }
    return Game(meta=meta, team_box=tb, player_box=pb, pbp=pbp)


def load_game(game_id, refresh: bool = False) -> Game:
    return parse_game(fetch_game_json(game_id, refresh=refresh))


# ---------------------------------------------------------------------------
# 3. a whole season, with individually fetched games merged in
# ---------------------------------------------------------------------------
@dataclass
class Season:
    season: int
    team_box: pd.DataFrame      # every completed game, all 30 teams
    player_box: pd.DataFrame
    pbp: pd.DataFrame
    schedule: pd.DataFrame      # the full league schedule (played and upcoming)

    @property
    def label(self) -> str:
        return config.season_label(self.season)


def _empty(cols):
    return pd.DataFrame(columns=cols)


def _concat(frames: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    """Stack tables, skipping empty ones. An empty table has no column types, and stacking
    it onto real data turns every number column into generic 'object' columns (pandas 3),
    which breaks the maths later. This happens before opening night, when the archive has
    no games yet but a game was fetched straight from ESPN."""
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else _empty(cols)


def load_season(season: int, with_pbp: bool = True, refresh: bool = False) -> Season:
    tb = fetch_archive("team_box", season, refresh)
    pb = fetch_archive("player_box", season, refresh)
    pbp = fetch_archive("pbp", season, refresh) if with_pbp else None
    sched = fetch_archive("schedule", season, refresh)

    tb = _team_box_from_archive(tb) if tb is not None and len(tb) else _empty(TEAM_BOX_COLS)
    pb = _player_box_from_archive(pb) if pb is not None and len(pb) else _empty(PLAYER_BOX_COLS)
    pbp = _pbp_from_archive(pbp) if pbp is not None and len(pbp) else _empty(PBP_COLS)
    sched = _schedule_from_archive(sched) if sched is not None and len(sched) else pd.DataFrame()

    # Games fetched one at a time (e.g. last night's, before the archive catches up)
    # replace their archive rows, so the newest data always wins.
    extra = [parse_game(json.loads(p.read_text(encoding="utf-8")))
             for p in sorted((config.DATA_DIR / "games").glob("*.json"))]
    extra = [g for g in extra if g.meta["season"] == season]
    if extra:
        ids = {g.meta["game_id"] for g in extra}
        tb = _concat([tb[~tb["game_id"].isin(ids)]] + [g.team_box for g in extra], TEAM_BOX_COLS)
        pb = _concat([pb[~pb["game_id"].isin(ids)]] + [g.player_box for g in extra], PLAYER_BOX_COLS)
        if with_pbp:
            pbp = _concat([pbp[~pbp["game_id"].isin(ids)]] + [g.pbp for g in extra], PBP_COLS)

    # NBA teams only (drops All-Star games), and not the NBA Cup final, which doesn't count
    tb = tb[tb["team_id"].isin(NBA_TEAM_IDS) & tb["opp_id"].isin(NBA_TEAM_IDS)]
    if len(sched):
        cup_final = set(sched.loc[sched["note"].str.contains("Cup Championship", case=False), "game_id"])
        tb = tb[~tb["game_id"].isin(cup_final)]
    keep = set(tb["game_id"])
    pb = pb[pb["game_id"].isin(keep)]
    pbp = pbp[pbp["game_id"].isin(keep)] if len(pbp) else pbp
    # true/false columns must stay boolean even when a table is empty (before opening night),
    # otherwise pandas reads df[~df["dnp"]] as a column selection
    tb = tb.astype({"win": bool})
    pb = pb.astype({"dnp": bool, "starter": bool})
    pbp = pbp.astype({"scoring": bool, "shooting": bool})
    tb = tb.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return Season(season, tb, pb.reset_index(drop=True), pbp.reset_index(drop=True), sched)


# ---------------------------------------------------------------------------
# 4. the team's schedule, and finding the latest game
# ---------------------------------------------------------------------------
def team_schedule(season: Season, team: str = None) -> pd.DataFrame:
    """The team's games (played + upcoming) from the archive schedule, plus any games
    only ESPN knows about yet (preseason, NBA Cup games added in December)."""
    team = team or config.TEAM
    s = season.schedule
    rows = s[(s["home_abbr"] == team) | (s["away_abbr"] == team)].copy() if len(s) else pd.DataFrame()
    live = _espn_team_schedule(season.season, team)
    if live is not None and len(live):
        rows = pd.concat([rows, live[~live["game_id"].isin(set(rows.get("game_id", [])))]], ignore_index=True)
    if not len(rows):
        return rows
    rows["home"] = rows["home_abbr"] == team
    rows["opp_abbr"] = rows["away_abbr"].where(rows["home"], rows["home_abbr"])
    return rows.sort_values("start_utc").reset_index(drop=True)


def _espn_team_schedule(season: int, team: str) -> pd.DataFrame | None:
    out = []
    for stype in (PRESEASON, REGULAR, PLAYOFFS):
        try:
            r = _espn_get(ESPN_TEAM_SCHEDULE.format(team=team.lower(), season=season, stype=stype), timeout=20)
            events = r.json().get("events", [])
        except (requests.RequestException, ValueError):
            if stype == PRESEASON:
                continue      # no preseason listed is fine; try the regular season
            return None if not out else pd.DataFrame(out)
        for e in events:
            comp = _get(e, "competitions.0", {})
            cs = comp.get("competitors", [])
            h = next((c for c in cs if c.get("homeAway") == "home"), {})
            a = next((c for c in cs if c.get("homeAway") == "away"), {})
            out.append({
                "game_id": int(e["id"]), "start_utc": pd.to_datetime(e.get("date"), utc=True, errors="coerce"),
                "game_date": _eastern_date(e.get("date", "")), "season_type": stype,
                "home_id": _int(_get(h, "team.id")), "away_id": _int(_get(a, "team.id")),
                "home_abbr": _get(h, "team.abbreviation", ""), "away_abbr": _get(a, "team.abbreviation", ""),
                "venue": _get(comp, "venue.fullName", ""), "neutral": bool(comp.get("neutralSite", False)),
                "note": " · ".join(n.get("headline", "") for n in (comp.get("notes") or [])),
                "completed": bool(_get(comp, "status.type.completed", False)),
            })
    return pd.DataFrame(out) if out else None


def latest_game_id(season: Season, team: str = None) -> int:
    """The most recent game that should be over by now (tip-off more than 2.5 hours ago)."""
    sched = team_schedule(season, team)
    if not len(sched):
        raise GameNotAvailable("No schedule found for this season yet.")
    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=2, minutes=30))
    played = sched[sched["start_utc"] <= cutoff]
    if not len(played):
        first = sched.iloc[0]
        raise GameNotAvailable(f"No games played yet. First game: {first['game_date']} vs {first['opp_abbr']}.")
    return int(played.iloc[-1]["game_id"])
