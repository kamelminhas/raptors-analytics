"""Building the HTML reports: one per game, one season dashboard, and an index."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import jinja2
import markdown as md
import numpy as np
import pandas as pd

import config
from . import charts as C
from . import data as D
from . import insights as I
from . import metrics as M
from . import pbp as P

ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates"),
                         autoescape=jinja2.select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
PERIOD_LABELS = ["Q1", "Q2", "Q3", "Q4", "OT", "2OT", "3OT", "4OT"]


# ---------------------------------------------------------------------------
# everything loaded once, reused for every report in a run
# ---------------------------------------------------------------------------
@dataclass
class Workbench:
    season: D.Season
    prev: D.Season
    tg: pd.DataFrame          # team-games, this season (all teams)
    tg_prev: pd.DataFrame
    pg: pd.DataFrame          # player-games, this season
    pg_prev: pd.DataFrame
    shots: pd.DataFrame       # every shot this season
    shots_prev: pd.DataFrame
    model: dict               # four-factor weights
    names: dict               # athlete id -> name


def load_workbench(season: int, refresh: bool = False) -> Workbench:
    s = D.load_season(season, refresh=refresh)
    prev = D.load_season(season - 1)
    tg = M.team_games(s.team_box, s.player_box, s.pbp)
    tg_prev = M.team_games(prev.team_box, prev.player_box, prev.pbp)
    parts = [x[M.counts_in_standings(x)] for x in (tg_prev, tg) if len(x)]
    both = pd.concat(parts, ignore_index=True)
    names = {**dict(zip(prev.player_box["athlete_id"].astype(int), prev.player_box["name"])),
             **dict(zip(s.player_box["athlete_id"].astype(int), s.player_box["name"]))}
    return Workbench(season=s, prev=prev, tg=tg, tg_prev=tg_prev,
                     pg=M.player_games(s.player_box), pg_prev=M.player_games(prev.player_box),
                     shots=P.shots(s.pbp) if len(s.pbp) else pd.DataFrame(),
                     shots_prev=P.shots(prev.pbp) if len(prev.pbp) else pd.DataFrame(),
                     model=M.fit_four_factors(both), names=names)


# ---------------------------------------------------------------------------
# small formatting helpers
# ---------------------------------------------------------------------------
def long_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%A}, {d:%B} {d.day}, {d.year}"


def short_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


def pct(v, digits=1):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{digits}f}"


def kpi(label, value, base, higher_good, fmt="{:.1f}", note="", base_label=""):
    out = {"label": label, "value": fmt.format(value), "note": note}
    if base is not None and not (isinstance(base, float) and np.isnan(base)):
        d = value - base
        out["delta"] = f"{d:+.1f}"
        out["arrow"] = "▲" if d > 0 else ("▼" if d < 0 else "")
        out["tone"] = "flat" if higher_good is None or abs(d) < 0.05 else ("good" if (d > 0) == higher_good else "bad")
        out["vs"] = f"vs {base_label} {fmt.format(base)}"
    return out


def slug_for(row) -> str:
    return f"{row['game_date']}-{'vs' if row['home_away'] == 'home' else 'at'}-{row['opp_abbr']}"


def season_dir(season: int) -> Path:
    return config.REPORTS_DIR / config.season_label(season)


def _json_for_script(s: str) -> str:
    return s.replace("</", "<\\/")


def _chart_js(fragment: bool, page_dir: Path | None) -> dict:
    """The charting library (Vega), saved as vega-bundle.js in the same folder as the pages
    that use it. Safari only lets a page opened from disk load files from its own folder (and
    below), so one shared copy in a parent folder doesn't work there. Also works offline and
    on GitHub Pages. A single-file page (fragment) gets it inlined instead."""
    import vl_convert as vlc
    if fragment or page_dir is None:
        return {"inline": vlc.javascript_bundle()}
    path = page_dir / "vega-bundle.js"
    if not path.exists():
        page_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(vlc.javascript_bundle(), encoding="utf-8")
    return {"src": "vega-bundle.js"}


# ---------------------------------------------------------------------------
# baselines: "compared to what?"
# ---------------------------------------------------------------------------
def _team_baseline(wb: Workbench, team: str, before: str) -> dict:
    prior = wb.tg[(wb.tg["team_abbr"] == team) & (wb.tg["game_date"] < before) & (wb.tg["season_type"] != D.PRESEASON)]
    label = "season"
    if len(prior) < config.MIN_GAMES_FOR_SEASON_BASELINE:
        prior = wb.tg_prev[(wb.tg_prev["team_abbr"] == team) & M.counts_in_standings(wb.tg_prev)]
        label = config.season_label(wb.prev.season)
    if not len(prior):
        return {}
    t = M.team_season_table(prior.assign(season_type=D.REGULAR)) if len(prior) else None
    r = t.loc[team]
    return {"label": label, "ortg": r["ortg"], "drtg": r["drtg"], "net": r["net"], "pace": r["pace"],
            "efg": r["efg"], "tov_pct": r["tov_pct"], "orb_pct": r["orb_pct"], "ftr": r["ftr"],
            "o_efg": r["o_efg"], "o_tov_pct": r["o_tov_pct"], "o_orb_pct": 1 - r["drb_pct"], "o_ftr": r["o_ftr"],
            "fg3_pct": r["fg3_pct"], "o_fg3_pct": r["o_fg3_pct"],
            "fg3_label": "season" if label == "season" else f"{label}", "n": len(prior)}


def _league_baseline(wb: Workbench, before: str) -> tuple[pd.DataFrame, str]:
    prior = wb.tg[(wb.tg["game_date"] < before) & M.counts_in_standings(wb.tg)]
    if len(prior) >= 300:
        return prior, "this season"
    return wb.tg_prev[M.counts_in_standings(wb.tg_prev)], config.season_label(wb.prev.season)


def _player_baselines(wb: Workbench, ids, before: str) -> dict:
    out = {}
    cur = wb.pg[(wb.pg["game_date"] < before) & wb.pg["athlete_id"].isin(ids) & (wb.pg["season_type"] != D.PRESEASON)]
    old = wb.pg_prev[wb.pg_prev["athlete_id"].isin(ids) & (wb.pg_prev["season_type"] == D.REGULAR)]
    for pid in ids:
        mine = cur[cur["athlete_id"] == pid]
        label = "this season"
        if len(mine) < config.MIN_GAMES_FOR_SEASON_BASELINE:
            mine, label = old[old["athlete_id"] == pid], config.season_label(wb.prev.season)
            if len(mine) < 10:
                continue
        fga, fta = mine["fga"].sum(), mine["fta"].sum()
        out[pid] = {"label": label, "n": len(mine), "gmsc_avg": mine["gmsc"].mean(), "gmsc_sd": mine["gmsc"].std(),
                    "ts_avg": mine["pts"].sum() / (2 * (fga + 0.44 * fta)) if fga + fta > 0 else None,
                    "usg_avg": (mine["usg"] * mine["min"]).sum() / mine["min"].sum(),
                    "pts_avg": mine["pts"].mean(), "min_avg": mine["min"].mean()}
    return out


# ---------------------------------------------------------------------------
# the game report
# ---------------------------------------------------------------------------
def build_game_report(wb: Workbench, game_id: int, fragment: bool = False, out_path: Path | None = None,
                      title: str | None = None) -> Path:
    s, team = wb.season, config.TEAM
    rows = wb.tg[(wb.tg["game_id"] == game_id) & (wb.tg["team_abbr"] == team)]
    if not len(rows):
        raise D.GameNotAvailable(f"Game {game_id} isn't a completed {team} game in {s.label}.")
    row = rows.iloc[0]
    team_id, opp, opp_id, gdate = int(row["team_id"]), row["opp_abbr"], int(row["opp_id"]), row["game_date"]
    home = row["home_away"] == "home"
    gp = s.pbp[s.pbp["game_id"] == game_id]
    meta = _game_meta(s, game_id, gp)
    slug = slug_for(row)

    # --- who, when, where ------------------------------------------------
    tt = wb.tg[(wb.tg["team_abbr"] == team) & (wb.tg["game_date"] <= gdate)]
    reg = tt[M.counts_in_standings(tt)]
    st = int(row["season_type"])
    if st == D.REGULAR:
        game_label = f"Game {len(reg)} of 82"
        record = f"{int(reg['win'].sum())}-{int((~reg['win']).sum())}"
        streak = M.streak(list(reg["win"]))
    elif st == D.PRESEASON:
        game_label, record, streak = "Preseason", "Preseason", ""
    else:
        series = tt[(tt["season_type"] == st) & (tt["opp_abbr"] == opp)]
        game_label = (meta.get("note") or M.season_type_label(st)).replace(" - ", ", ")
        record = f"Series {int(series['win'].sum())}-{int((~series['win']).sum())}"
        streak = ""
    opp_name = meta.get("home_name") if meta.get("home_id") == opp_id else meta.get("away_name")
    periods = int(meta.get("periods") or 4)
    ls_team = meta.get("linescores", {}).get(team_id, [])
    ls_opp = meta.get("linescores", {}).get(opp_id, [])
    linescore = [{"label": PERIOD_LABELS[i] if i < len(PERIOD_LABELS) else f"P{i+1}",
                  "team": ls_team[i] if i < len(ls_team) else "", "opp": ls_opp[i] if i < len(ls_opp) else ""}
                 for i in range(max(len(ls_team), len(ls_opp)))]

    # --- team numbers vs baseline ----------------------------------------
    base = _team_baseline(wb, team, gdate)
    bl = base.get("label", "")
    bl_txt = "season avg" if bl == "season" else f"{bl} avg"
    kpis = [
        kpi("Offensive rating", row["ortg"], base.get("ortg"), True, note="points scored per 100 possessions", base_label=bl_txt),
        kpi("Defensive rating", row["drtg"], base.get("drtg"), False, note="points allowed per 100 possessions", base_label=bl_txt),
        kpi("Net rating", row["net"], base.get("net"), True, fmt="{:+.1f}", note="offense minus defense", base_label=bl_txt),
        kpi("Pace", row["pace"], base.get("pace"), None, note="possessions per 48 minutes", base_label=bl_txt),
    ]
    league, league_label = _league_baseline(wb, gdate)
    contrib = M.factor_contributions(row, wb.model)
    residual = contrib.attrs["residual"]
    lg = {"efg": league["efg"].mean(), "tov_pct": league["tov_pct"].mean(), "orb_pct": league["orb_pct"].mean(),
          "ftr": league["ftr"].mean()}
    factor_rows = []
    for (key, label, a, b), (_, c) in zip(M.FACTORS, contrib.iterrows()):
        f = (lambda v: f"{v:.3f}") if key == "ftr" else (lambda v: f"{100 * v:.1f}%")
        factor_rows.append({"factor": label, "team": f(row[a]), "opp": f(row[b]), "league": f(lg[a]),
                            "team_base": f(base[a]) if base.get(a) is not None else "–",
                            "points": f"{c['points']:+.1f}", "tone": "good" if c["points"] > 0 else "bad",
                            "edge": team if row[a] != row[b] and ((row[a] > row[b]) != (key == "tov")) else opp})

    # --- flow of the game -------------------------------------------------
    fl = P.flow(gp, team_id) if len(gp) else None
    lead = P.lead_stats(fl) if fl is not None else {}
    runs = P.runs(gp, team_id) if len(gp) else []
    clutch = P.clutch(gp, team_id) if len(gp) else None
    run_rows = [{"who": team if r["side"] == "team" else opp, "pts": r["pts"],
                 "span": f"{P.fmt_clock(r['start_period'], r['start_clock'])} to {P.fmt_clock(r['end_period'], r['end_clock'])}"}
                for r in runs[:3] if r["pts"] >= 6]

    # --- players ----------------------------------------------------------
    pg_game = wb.pg[(wb.pg["game_id"] == game_id) & (wb.pg["team_id"] == team_id)]
    pb = _player_baselines(wb, list(pg_game["athlete_id"]), gdate)
    players = []
    for _, p in pg_game.sort_values(["starter", "min"], ascending=[False, False]).iterrows():
        b = pb.get(p["athlete_id"], {})
        z = None
        if b and b.get("gmsc_sd") and b["gmsc_sd"] > 0 and b["n"] >= config.MIN_GAMES_FOR_SEASON_BASELINE:
            z = (p["gmsc"] - b["gmsc_avg"]) / b["gmsc_sd"]
        line = f"{int(p['pts'])} pts ({int(p['fgm'])}-{int(p['fga'])} FG), {int(p['reb'])} reb, {int(p['ast'])} ast"
        players.append({
            "name": p["name"], "short": p["short_name"], "pos": p["position"], "starter": bool(p["starter"]),
            "min": float(p["min"]), "pts": int(p["pts"]), "reb": int(p["reb"]), "ast": int(p["ast"]),
            "stl": int(p["stl"]), "blk": int(p["blk"]), "tov": int(p["tov"]), "pf": int(p["pf"]),
            "oreb": int(p["oreb"]),
            "fg": f"{int(p['fgm'])}-{int(p['fga'])}", "fg3": f"{int(p['fg3m'])}-{int(p['fg3a'])}",
            "ft": f"{int(p['ftm'])}-{int(p['fta'])}", "pm": int(p["pm"]),
            "ts": p["ts"], "usg": p["usg"], "gmsc": p["gmsc"], "line": line,
            "gmsc_avg": b.get("gmsc_avg"), "ts_avg": b.get("ts_avg"), "usg_avg": b.get("usg_avg"), "z": z,
            "base_label": b.get("label", ""),
            "delta": None if not b else p["gmsc"] - b["gmsc_avg"],
        })
    opp_game = wb.pg[(wb.pg["game_id"] == game_id) & (wb.pg["team_id"] == opp_id)]
    opp_players = [{
        "name": p["name"], "pos": p["position"], "starter": bool(p["starter"]), "min": float(p["min"]),
        "pts": int(p["pts"]), "reb": int(p["reb"]), "oreb": int(p["oreb"]), "ast": int(p["ast"]),
        "stl": int(p["stl"]), "blk": int(p["blk"]), "tov": int(p["tov"]),
        "fg": f"{int(p['fgm'])}-{int(p['fga'])}", "fg3": f"{int(p['fg3m'])}-{int(p['fg3a'])}",
        "ft": f"{int(p['ftm'])}-{int(p['fta'])}", "pm": int(p["pm"]), "ts": p["ts"], "gmsc": p["gmsc"],
    } for _, p in opp_game.sort_values(["starter", "min"], ascending=[False, False]).iterrows()]
    dnp_rows = s.player_box[(s.player_box["game_id"] == game_id) & (s.player_box["team_id"] == team_id) & s.player_box["dnp"]]
    dnp = [f"{r['name']}" + (f" ({r['reason'].strip().capitalize()})" if r["reason"].strip() else "") for _, r in dnp_rows.iterrows()]

    # --- shots --------------------------------------------------------------
    sh_game = wb.shots[(wb.shots["game_id"] == game_id) & (wb.shots["team_id"].astype("float") == team_id)] if len(wb.shots) else pd.DataFrame()
    zones = []
    if len(sh_game):
        tonight = P.zone_table(sh_game)
        season_sh = wb.shots[(wb.shots["team_id"].astype("float") == team_id) & wb.shots["game_id"].isin(
            wb.tg[(wb.tg["team_abbr"] == team) & (wb.tg["game_date"] < gdate) & (wb.tg["season_type"] != D.PRESEASON)]["game_id"])]
        s_label = "Season"
        if len(season_sh) < 400:  # fewer than ~5 games: use last season instead
            prev_ids = wb.tg_prev[wb.tg_prev["team_abbr"] == team]["game_id"]
            season_sh = wb.shots_prev[wb.shots_prev["game_id"].isin(prev_ids) & (wb.shots_prev["team_id"].astype("float") == team_id)]
            s_label = config.season_label(wb.prev.season)
        season_z = P.zone_table(season_sh) if len(season_sh) else None
        lg_sh = wb.shots[wb.shots["game_id"].isin(league["game_id"])] if league_label == "this season" else wb.shots_prev
        league_z = P.zone_table(lg_sh) if len(lg_sh) else None
        for z in P.ZONES:
            t = tonight.loc[z]
            zones.append({"zone": z, "fga": int(t["fga"]), "fgm": int(t["fgm"]),
                          "fg": pct(t["fg_pct"], 0) if t["fga"] else "–", "share": pct(t["share"], 0),
                          "s_share": pct(season_z.loc[z, "share"], 0) if season_z is not None else "–",
                          "s_fg": pct(season_z.loc[z, "fg_pct"], 0) if season_z is not None else "–",
                          "l_fg": pct(league_z.loc[z, "fg_pct"], 0) if league_z is not None else "–"})
        zones_label = s_label
    else:
        zones_label = ""

    # --- rest / schedule context --------------------------------------------
    rest = _rest_text(wb, team, opp, gdate)

    # --- what stands out + questions ------------------------------------------
    opp_base = _team_baseline(wb, opp, gdate)
    ctx = {"team": team, "opp": opp, "row": row, "team_baseline": base, "contrib": contrib,
           "players": players, "runs": runs, "clutch": clutch, "rest": rest, "dnp": dnp,
           "opp_fg3_base": opp_base.get("fg3_pct")}
    facts, questions = I.game_insights(ctx)

    # --- your notes ------------------------------------------------------------
    notes_path = config.NOTES_DIR / s.label / f"{slug}.md"
    notes_html = _notes(notes_path, team, opp, gdate, game_id, questions)

    # --- charts: interactive specs + PNGs to post -----------------------------------
    specs, pngs = {}, []
    img = season_dir(s.season) / "img" / slug
    score_line = f"{team} {int(row['pts'])}, {opp} {int(row['opp_pts'])}"
    if fl is not None:
        specs["flow"] = _json_for_script(C.both_themes(C.flow_chart, fl, team, opp))
        pngs.append(C.save_png(C.flow_chart, img / "flow.png", fl, team, opp,
                               title=f"{score_line}: how the lead changed",
                               subtitle=f"{short_date(gdate)} · above zero = {config.TEAM_NAME} ahead"))
    specs["factors"] = _json_for_script(C.both_themes(C.factors_chart, contrib, residual, team, opp))
    pngs.append(C.save_png(C.factors_chart, img / "factors.png", contrib, residual, team, opp,
                           title=f"{score_line}: what decided it",
                           subtitle="Each of the four factors, in points per 100 possessions for Toronto"))
    if len(sh_game):
        specs["shots"] = _json_for_script(C.both_themes(C.shot_chart, sh_game, wb.names))
        pngs.append(C.save_png(C.shot_chart, img / "shots.png", sh_game, wb.names, width=520,
                               title=f"{config.TEAM_NAME} shots {'vs' if home else 'at'} {opp}, {short_date(gdate)}",
                               subtitle=f"{int(sh_game['made'].sum())} of {len(sh_game[sh_game['zone'].isin(P.ZONES)])} made"))

    out = out_path or (season_dir(s.season) / "games" / f"{slug}.html")
    rel_img = Path("..") / "img" / slug if out_path is None else None
    html = ENV.get_template("game.html").render(
        fragment=fragment, cfg=config, season_label=s.label, team=team, opp=opp, opp_name=opp_name,
        home=home, row=row, game_label=game_label, record=record, streak=streak,
        date_long=long_date(gdate), meta=meta, linescore=linescore, periods=periods,
        kpis=kpis, factor_rows=factor_rows, residual=residual, model=wb.model, league_label=league_label,
        base_label=bl_txt, lead=lead, run_rows=run_rows, clutch=clutch, players=players, dnp=dnp,
        opp_players=opp_players,
        zones=zones, zones_label=zones_label, facts=facts, questions=questions,
        notes_html=notes_html, notes_rel=notes_path.relative_to(config.ROOT).as_posix(),
        specs=specs, pngs=[p.name for p in pngs if p], img_rel=rel_img.as_posix() if rel_img else None,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"), game_id=game_id,
        nav=_nav_links(s.season, slug) if out_path is None else {},
        js=_chart_js(fragment, None if fragment else out.parent), title_override=title,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _game_meta(s: D.Season, game_id: int, gp: pd.DataFrame) -> dict:
    """Venue, officials, line score... from the game's ESPN feed; rebuilt from the
    play-by-play and schedule if the feed can't be fetched."""
    try:
        return D.load_game(game_id).meta
    except (D.GameNotAvailable, Exception):
        pass
    meta = {"game_id": game_id, "linescores": {}, "periods": int(gp["period"].max()) if len(gp) else 4}
    sc = s.schedule[s.schedule["game_id"] == game_id] if len(s.schedule) else pd.DataFrame()
    if len(sc):
        r = sc.iloc[0]
        meta.update({"venue": r["venue"], "note": r["note"], "home_id": int(r["home_id"]), "away_id": int(r["away_id"])})
    if len(gp):
        ends = gp.sort_values("seq").groupby("period")[["home_score", "away_score"]].last()
        h = ends["home_score"].diff().fillna(ends["home_score"]).astype(int).tolist()
        a = ends["away_score"].diff().fillna(ends["away_score"]).astype(int).tolist()
        meta["linescores"] = {int(gp["home_id"].iloc[0]): h, int(gp["away_id"].iloc[0]): a}
    return meta


def _rest_text(wb: Workbench, team: str, opp: str, gdate: str) -> str | None:
    def days_since(t):
        prior = wb.tg[(wb.tg["team_abbr"] == t) & (wb.tg["game_date"] < gdate)]
        if not len(prior):
            return None
        return (date.fromisoformat(gdate) - date.fromisoformat(prior["game_date"].max())).days
    a, b = days_since(team), days_since(opp)
    if a is None or b is None:
        return None
    def say(t, d):
        return f"{t} on the second night of a back-to-back" if d == 1 else f"{t} on {d - 1} days' rest"
    if a == 1 or b == 1 or abs(a - b) >= 2:
        return f"Schedule: {say(team, a)}; {say(opp, b)}."
    return None


NOTES_STUB = """<!--
  Your notes for {team} {vs} {opp}, {date}.
  Write in plain markdown. Anything outside these comment blocks shows up in the report's
  "Analyst notes" section when you re-run:   python run.py game --id {game_id}

  Questions the data raised (answer the ones you find interesting, skip the rest):
{questions}
-->

## Headline
<!-- One sentence: what happened, and why. -->


## What I noticed
<!-- Two or three bullets. Each one: a number from the report, then your explanation of it. -->


## What to watch next game
<!-- One thing you expect to continue or change, and why. -->

"""


def _notes(path: Path, team, opp, gdate, game_id, questions) -> str | None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        qs = "\n".join(f"    {i}. {q}" for i, q in enumerate(questions, 1))
        path.write_text(NOTES_STUB.format(team=team, vs="vs/at", opp=opp, date=gdate, game_id=game_id, questions=qs),
                        encoding="utf-8")
        return None
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        return None
    return md.markdown(text, extensions=["extra", "sane_lists"])


def _nav_links(season: int, slug: str) -> dict:
    files = sorted(p.stem for p in (season_dir(season) / "games").glob("*.html"))
    if slug not in files:
        files = sorted(files + [slug])
    i = files.index(slug)
    return {"prev": files[i - 1] if i > 0 else None, "next": files[i + 1] if i + 1 < len(files) else None}


# ---------------------------------------------------------------------------
# the season dashboard
# ---------------------------------------------------------------------------
def build_season_page(wb: Workbench, fragment: bool = False, out_path: Path | None = None) -> Path:
    s, team = wb.season, config.TEAM
    tt = wb.tg[(wb.tg["team_abbr"] == team) & M.counts_in_standings(wb.tg)]
    sched = D.team_schedule(s, team)
    out = out_path or (season_dir(s.season) / "index.html")
    ctx = {"fragment": fragment, "cfg": config, "season_label": s.label, "team": team, "has_games": len(tt) > 0,
           "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "specs": {}, "pngs": [],
           "js": _chart_js(fragment, None if fragment else out.parent)}
    img = season_dir(s.season) / "img" / "season"

    if len(tt):
        ts = M.team_season_table(wb.tg)
        r = ts.loc[team]
        gp = int(r["gp"])
        rank = lambda c: f"{int(r[c + '_rank'])}{_ordinal(int(r[c + '_rank']))}"
        ctx["kpis"] = [
            {"label": "Record", "value": f"{int(r['w'])}-{int(r['l'])}", "note": f"{rank('win_pct')} in the NBA by win %"},
            {"label": "Net rating", "value": f"{r['net']:+.1f}", "note": f"{rank('net')} · points per 100 possessions"},
            {"label": "Offense", "value": f"{r['ortg']:.1f}", "note": f"{rank('ortg')} · points scored per 100"},
            {"label": "Defense", "value": f"{r['drtg']:.1f}", "note": f"{rank('drtg')} · points allowed per 100"},
            {"label": "Pythagorean record", "value": f"{r['pyth_w']:.0f}-{gp - r['pyth_w']:.0f}",
             "note": f"{r['luck']:+.1f} wins vs expected"},
            {"label": "SRS", "value": f"{r['srs']:+.1f}", "note": f"{rank('srs')} · schedule-adjusted margin"},
        ]
        roll = M.rolling_team(tt)
        ctx["specs"]["rolling"] = _json_for_script(C.both_themes(C.rolling_chart, roll))
        ctx["specs"]["league"] = _json_for_script(C.both_themes(C.league_chart, ts, team))
        ctx["pngs"] = [p.name for p in [
            C.save_png(C.rolling_chart, img / "rolling.png", roll, title=f"{config.TEAM_NAME} {s.label}: net rating, game by game",
                       subtitle=f"Grey bars: single games · red line: last 10 games · through game {gp}"),
            C.save_png(C.league_chart, img / "league.png", ts, team, width=560, height=480,
                       title=f"Offense vs defense, every NBA team, {s.label}",
                       subtitle=f"Through {short_date(tt['game_date'].max())} · up and right is better")] if p]
        ctx["factor_table"] = [
            {"side": "Offense", "rows": [(lbl, pct(r[c]) if c != "ftr" else f"{r[c]:.3f}", int(r[c + "_rank"]))
                                         for lbl, c in [("Shooting (eFG%)", "efg"), ("Turnovers (TOV%)", "tov_pct"),
                                                        ("Off. rebounding (ORB%)", "orb_pct"), ("Free throws (FT/FGA)", "ftr")]]},
            {"side": "Defense", "rows": [(lbl, pct(r[c]) if c != "o_ftr" else f"{r[c]:.3f}", int(r[c + "_rank"]))
                                         for lbl, c in [("Opp. shooting (eFG%)", "o_efg"), ("Forced turnovers (TOV%)", "o_tov_pct"),
                                                        ("Def. rebounding (DRB%)", "drb_pct"), ("Opp. free throws (FT/FGA)", "o_ftr")]]},
        ]
        ctx["hca"] = ts.attrs.get("hca", 0.0)
        # players
        pg = wb.pg[(wb.pg["team_abbr"] == team) & (wb.pg["season_type"] == D.REGULAR)]
        summ = M.player_summary(pg).sort_values("min", ascending=False)
        last_ids = tt.sort_values("game_date").tail(10)["game_id"]
        last = M.player_summary(pg[pg["game_id"].isin(last_ids)])
        players = []
        for pid, p in summ.iterrows():
            l10 = last.loc[pid] if pid in last.index else None
            players.append({"name": p["name"], "gp": int(p["gp"]), "gs": int(p["gs"]), "mpg": p["mpg"],
                            "pts": p["pts_pg"], "reb": p["reb_pg"], "ast": p["ast_pg"], "ts": p["ts"], "usg": p["usg"],
                            "gmsc": p["gmsc"], "pm": p["pm_pg"], "fg3": p["fg3_pct"], "fg3a": int(p["fg3a"]),
                            "l10_mpg": None if l10 is None else l10["mpg"], "l10_ts": None if l10 is None else l10["ts"],
                            "l10_gp": 0 if l10 is None else int(l10["gp"])})
        ctx["players"] = players
        # game log
        log = []
        for _, g in tt.sort_values("game_date").iterrows():
            slug = slug_for(g)
            exists = (season_dir(s.season) / "games" / f"{slug}.html").exists() or (out_path is not None)
            log.append({"date": short_date(g["game_date"]), "matchup": ("vs " if g["home_away"] == "home" else "at ") + g["opp_abbr"],
                        "win": bool(g["win"]), "score": f"{int(g['pts'])}-{int(g['opp_pts'])}", "net": g["net"],
                        "href": f"games/{slug}.html" if exists and out_path is None else None})
        ctx["log"] = log[::-1]
        ctx["questions"] = I.season_questions(r, gp)
        ctx["gp"] = gp
        ctx["last_date"] = long_date(tt["game_date"].max())

    # upcoming games + milestones
    if len(sched):
        now = pd.Timestamp.now(tz="UTC")
        reg_sched = sched[sched["season_type"] == D.REGULAR].reset_index(drop=True)
        upcoming = sched[sched["start_utc"] > now].head(8)
        rows_up = []
        all_dates = list(sched["game_date"])
        for _, g in upcoming.iterrows():
            i = all_dates.index(g["game_date"])
            b2b = i > 0 and (date.fromisoformat(g["game_date"]) - date.fromisoformat(all_dates[i - 1])).days == 1
            rows_up.append({"date": short_date(g["game_date"]), "matchup": ("vs " if g["home"] else "at ") + g["opp_abbr"],
                            "b2b": b2b, "note": g.get("note", "") or ""})
        ctx["upcoming"] = rows_up
        marks = []
        for n, label in [(21, "Quarter mark"), (41, "Halfway"), (62, "Three-quarter mark")]:
            if len(reg_sched) >= n:
                g = reg_sched.iloc[n - 1]
                marks.append({"label": label, "game": n, "date": short_date(g["game_date"]),
                              "matchup": ("vs " if g["home"] else "at ") + g["opp_abbr"]})
        ctx["milestones"] = marks
        if len(reg_sched) < 82:
            ctx["sched_note"] = (f"{len(reg_sched)} of 82 games are scheduled so far. The rest are set after NBA Cup "
                                 "group play, which shifts the game numbers after early December.")
        if not len(tt) and len(reg_sched):
            first = reg_sched.iloc[0]
            ctx["opener"] = f"{long_date(first['game_date'])}, {('vs ' if first['home'] else 'at ') + first['opp_abbr']}"

    html = ENV.get_template("season.html").render(**ctx)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _ordinal(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def build_index() -> Path:
    seasons = sorted([p.name for p in config.REPORTS_DIR.glob("20*-*") if (p / "index.html").exists()], reverse=True)
    html = ENV.get_template("index.html").render(cfg=config, seasons=seasons,
                                                 generated=datetime.now().strftime("%Y-%m-%d %H:%M"))
    out = config.REPORTS_DIR / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
