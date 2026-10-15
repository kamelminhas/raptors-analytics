#!/usr/bin/env python3
"""Raptors analytics: one command after every game.

    python run.py setup                   first-time setup (downloads data, checks everything works)
    python run.py game                    report for the most recent Raptors game
    python run.py game --date 2026-10-21  the Raptors game on a date
    python run.py game --id 401909837     one game, by ESPN game id
    python run.py season                  rebuild the season tracker page
    python run.py demo                    sample reports from last season, to see what you'll get
    python run.py rebuild                 re-render every report this season (after editing templates)

Add --season 2026 to work on another season (seasons are named by the year they end: 2026 = 2025-26).
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

import config

if sys.version_info < (3, 10):
    sys.exit("This project needs Python 3.10 or newer (you have %d.%d)." % sys.version_info[:2])

try:
    import requests
    from raptors import data as D
    from raptors import report as R
except ImportError as e:  # most likely: requirements not installed yet
    sys.exit(f"Missing a package ({e.name}). Run:  python -m pip install -r requirements.txt")


def say(msg: str) -> None:
    print(msg, flush=True)


def open_file(path: Path, args) -> None:
    say(f"\n  → {path}")
    if not getattr(args, "no_open", False):
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception:
            pass


# ---------------------------------------------------------------------------
def cmd_game(args) -> None:
    season = args.season
    if args.id:
        gid = int(args.id)
    else:
        s = D.load_season(season, with_pbp=False)
        if args.date:
            sched = D.team_schedule(s)
            hit = sched[sched["game_date"] == args.date] if len(sched) else sched
            if not len(hit):
                sys.exit(f"No {config.TEAM} game found on {args.date}.")
            gid = int(hit.iloc[0]["game_id"])
        else:
            gid = D.latest_game_id(s)
    say(f"Building the report for game {gid}…")
    try:
        D.fetch_game_json(gid, refresh=args.refresh)
    except D.GameNotAvailable as e:
        say(f"  · {e} Trying the archive instead.")
    t0 = time.time()
    wb = R.load_workbench(season, refresh=args.refresh)
    out = R.build_game_report(wb, gid)
    R.build_season_page(wb)
    R.build_index()
    say(f"  · done in {time.time() - t0:.0f}s")
    open_file(out, args)


def cmd_season(args) -> None:
    wb = R.load_workbench(args.season, refresh=args.refresh)
    out = R.build_season_page(wb)
    R.build_index()
    open_file(out, args)


def cmd_rebuild(args) -> None:
    wb = R.load_workbench(args.season, refresh=args.refresh)
    games = wb.tg[wb.tg["team_abbr"] == config.TEAM].sort_values("game_date")
    say(f"Re-rendering {len(games)} game reports…")
    for i, gid in enumerate(games["game_id"], 1):
        R.build_game_report(wb, int(gid))
        if i % 10 == 0:
            say(f"  · {i}/{len(games)}")
    # a second pass so every report's previous/next links point at existing files
    out = R.build_season_page(wb)
    R.build_index()
    open_file(out, args)


def cmd_demo(args) -> None:
    season = config.CURRENT_SEASON - 1
    say(f"Building sample reports from {config.season_label(season)} (last season)…")
    wb = R.load_workbench(season)
    tt = wb.tg[wb.tg["team_abbr"] == config.TEAM].sort_values("game_date")
    picks = list(tt[tt["season_type"] == D.REGULAR]["game_id"].tail(3)) + list(tt[tt["season_type"] != D.REGULAR]["game_id"])
    for gid in picks:
        try:
            D.fetch_game_json(int(gid))
        except D.GameNotAvailable:
            pass
    wb = R.load_workbench(season)
    for n in range(2):  # twice, so previous/next links find each other
        for gid in picks:
            out = R.build_game_report(wb, int(gid))
    say(f"  · {len(picks)} game reports")
    R.build_season_page(wb)
    R.build_index()
    open_file(out, args)


def cmd_setup(args) -> None:
    say("Setting up…")
    say(f"  · Python {sys.version.split()[0]}: OK")
    cur, prev = config.CURRENT_SEASON, config.CURRENT_SEASON - 1
    for kind in ("schedule", "team_box", "player_box", "pbp"):
        for season in (prev, cur):
            df = D.fetch_archive(kind, season)
            status = f"{len(df):,} rows" if df is not None else "not published yet (normal before opening night)"
            say(f"  · {kind:10s} {config.season_label(season)}: {status}")
    _fonts()
    # can we reach ESPN directly? (same-night reports need it)
    ok, why = D.check_espn()
    say("  · ESPN live data: reachable" if ok else
        f"  · ESPN live data: NOT reachable ({why}), so reports will use the GitHub mirror (can lag by up to a day)")
    wb = R.load_workbench(cur)
    out = R.build_season_page(wb)
    R.build_index()
    say("\nAll set. Next: `python run.py demo` to see sample reports from last season.")
    open_file(out, args)


def _fonts() -> None:
    """Barlow for the PNG charts (the HTML pages load it from Google Fonts)."""
    folder = config.ROOT / "assets" / "fonts"
    folder.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/google/fonts/main/ofl/barlow/"
    got = 0
    for w in ("Regular", "Medium", "SemiBold", "Bold"):
        path = folder / f"Barlow-{w}.ttf"
        if path.exists():
            got += 1
            continue
        try:
            r = requests.get(base + path.name, timeout=30)
            if r.status_code == 200 and r.content[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
                path.write_bytes(r.content)
                got += 1
        except requests.RequestException:
            pass
    say(f"  · chart font (Barlow): {'installed' if got == 4 else 'not downloaded; PNGs will use a fallback font'}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Raptors game-by-game analytics", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--season", type=int, default=config.CURRENT_SEASON, help="season by end year, e.g. 2027 = 2026-27")
    common.add_argument("--refresh", action="store_true", help="re-download data even if the cached copy is recent")
    common.add_argument("--no-open", action="store_true", help="don't open the report in a browser")
    g = sub.add_parser("game", parents=[common], help="report for one game (default: the latest)")
    g.add_argument("--id", help="ESPN game id")
    g.add_argument("--date", help="YYYY-MM-DD")
    g.add_argument("--latest", action="store_true", help="the most recent game (the default)")
    sub.add_parser("season", parents=[common], help="rebuild the season tracker")
    sub.add_parser("rebuild", parents=[common], help="re-render every game report this season")
    sub.add_parser("demo", parents=[common], help="sample reports from last season")
    sub.add_parser("setup", parents=[common], help="first-time setup")
    args = ap.parse_args()
    cmds = {"game": cmd_game, "season": cmd_season, "rebuild": cmd_rebuild, "demo": cmd_demo, "setup": cmd_setup}
    if args.cmd not in cmds:
        ap.print_help()
        return
    try:
        cmds[args.cmd](args)
    except D.GameNotAvailable as e:
        sys.exit(f"\n{e}")
    except requests.ConnectionError:
        sys.exit("\nCouldn't reach the internet. Check your connection and try again.")


if __name__ == "__main__":
    main()
