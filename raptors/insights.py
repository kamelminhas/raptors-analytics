"""What stands out, and what to ask about it.

The code can tell you WHAT happened. Your job as the analyst is WHY and SO WHAT.
This module does the first part (flags the numbers that are unusual) and hands you
questions for the second part. The questions are the point: answering them, in your
own words, is what goes in your notes and, eventually, your portfolio.
"""
from __future__ import annotations

import pandas as pd

from . import pbp as P

FACTOR_QUESTIONS = {
    "efg": ("Was the shooting gap about shot quality or shot-making? Compare tonight's zone table with "
            "the season profile: more attempts at the rim and in the corners, or just tough shots going in (or not)?"),
    "tov": ("Were the turnovers forced or unforced? Steals are the forced ones: check how many of the "
            "turnovers were steals, and what the other team did to cause them."),
    "orb": ("Where did the offensive rebounds come from: a size mismatch, a lineup choice, or effort? "
            "Check who grabbed them, and which lineups gave them up."),
    "ftr": ("Why the free-throw gap: aggressive drives (look at points in the paint) or late-game fouling? "
            "Check when in the game the free throws came."),
}


def game_insights(ctx: dict) -> tuple[list[dict], list[str]]:
    """Returns (facts, questions). Each fact: {'kind': ..., 'text': ...}."""
    facts, questions = [], []
    team, opp = ctx["team"], ctx["opp"]
    row, base = ctx["row"], ctx["team_baseline"]

    # 1. the decisive factor
    con = ctx["contrib"]
    top = con.reindex(con["points"].abs().sort_values(ascending=False).index).iloc[0]
    helped = top["points"] > 0
    facts.append({"kind": "factor", "text": (
        f"The biggest factor was {top['factor'].split(' (')[0].lower()} "
        f"({_fmt_factor(top['key'], top['team'])} for {team} vs {_fmt_factor(top['key'], top['opp'])}), "
        f"worth about {top['points']:+.1f} points per 100 possessions {'for' if helped else 'against'} {team}.")})
    questions.append(FACTOR_QUESTIONS[top["key"]])

    # 2. three-point shooting vs normal (ours and theirs)
    if row["fg3a"] >= 15 and base.get("fg3_pct"):
        d = row["fg3_pct"] - base["fg3_pct"]
        if abs(d) >= 0.08:
            facts.append({"kind": "shooting", "text": (
                f"{team} shot {int(row['fg3m'])}/{int(row['fg3a'])} from three ({row['fg3_pct']:.0%}), "
                f"{'well above' if d > 0 else 'well below'} their {base['fg3_label']} {base['fg3_pct']:.1%}.")})
    if row["o_fg3a"] >= 15 and ctx.get("opp_fg3_base"):
        d = row["o_fg3_pct"] - ctx["opp_fg3_base"]
        if abs(d) >= 0.08:
            facts.append({"kind": "shooting", "text": (
                f"{opp} shot {int(row['o_fg3m'])}/{int(row['o_fg3a'])} from three ({row['o_fg3_pct']:.0%}), "
                f"vs {ctx['opp_fg3_base']:.1%} normally.")})
            questions.append(
                f"{opp}'s three-point shooting was {'unusually good' if d > 0 else 'unusually bad'}. Research suggests "
                "a defense controls how many threes it allows and how open they are far more than whether they go in. "
                "Was this the defense, or luck? What would you want to see on film to decide?")

    # 3. individual games that stood out
    for p in ctx["players"]:
        if p.get("z") is not None and abs(p["z"]) >= 1.5 and p["min"] >= 12:
            direction = "above" if p["z"] > 0 else "below"
            facts.append({"kind": "player", "text": (
                f"{p['name']}: Game Score {p['gmsc']:.1f}, {abs(p['z']):.1f} standard deviations {direction} "
                f"his usual ({p['gmsc_avg']:.1f}). {p['line']}.")})
    outliers = [p for p in ctx["players"] if p.get("z") is not None and abs(p["z"]) >= 1.5 and p["min"] >= 12]
    if outliers:
        p = max(outliers, key=lambda q: abs(q["z"]))
        usg = f"usage {p['usg']:.0f}% vs {p['usg_avg']:.0f}% normally" if p.get("usg_avg") else f"usage {p['usg']:.0f}%"
        ts = f"true shooting {p['ts']:.0%} vs {p['ts_avg']:.0%}" if p.get("ts_avg") and p.get("ts") == p.get("ts") else ""
        questions.append(
            f"{p['name']} was far from his normal. What changed: minutes, role, or shooting luck? "
            f"({usg}{'; ' + ts if ts else ''}.) Is any of it likely to carry over to the next game?")

    # 4. the run that swung it
    runs = ctx["runs"]
    if runs and runs[0]["pts"] >= 8:
        r = runs[0]
        who = team if r["side"] == "team" else opp
        facts.append({"kind": "run", "text": (
            f"Biggest run: {who} scored {r['pts']} straight, from {P.fmt_clock(r['start_period'], r['start_clock'])} "
            f"to {P.fmt_clock(r['end_period'], r['end_clock'])}.")})
        questions.append(
            f"What happened during {who}'s {r['pts']}-0 run ({P.fmt_clock(r['start_period'], r['start_clock'])} to "
            f"{P.fmt_clock(r['end_period'], r['end_clock'])})? Look for a substitution, a timeout, a switch in coverage, or a turnover streak.")

    # 5. clutch
    cl = ctx.get("clutch")
    if cl:
        facts.append({"kind": "clutch", "text": (
            f"Clutch time (last 5 minutes, within 5 points): {team} {cl['pts_for']}, {opp} {cl['pts_against']}.")})

    # 6. context: rest, back-to-backs
    rest = ctx.get("rest")
    if rest:
        facts.append({"kind": "context", "text": rest})

    # 7. who didn't play
    if ctx.get("dnp"):
        facts.append({"kind": "context", "text": "Didn't play: " + "; ".join(ctx["dnp"]) + "."})

    questions.append("If you could post one chart from this game, which one tells the story best? "
                     "Write the one-sentence caption you'd put under it.")
    return facts, questions


def _fmt_factor(key: str, v: float) -> str:
    return f"{v:.3f}" if key == "ftr" else f"{v:.1%}"


def season_questions(ts_row: pd.Series, gp: int) -> list[str]:
    q = []
    if gp >= 10 and abs(ts_row["luck"]) >= 2:
        q.append(f"The Raptors have {ts_row['luck']:+.1f} more wins than their points scored and allowed predict "
                 "(Pythagorean record). What explains the gap: close-game results, blowouts, or randomness? "
                 "Teams usually drift back toward their Pythagorean record. How much would you expect them to?")
    q.append("Which of the four factors is this team's identity on offense, and on defense? Check the league ranks. "
             "Has that changed since last season, and why (roster, scheme)?")
    q.append("Pick one player whose last 10 games differ from his season numbers. Is it role (minutes, usage) "
             "or efficiency? How many more games would you need before you believed it?")
    return q
