"""Play-by-play: shot locations, the flow of the game, runs, clutch time.

ESPN shot coordinates are in feet on a half court: x runs sideline to sideline (0-50,
rim at x = 25) and y runs from the baseline end toward half court. Fitting the stated
shot distances in every 2025-26 play ("makes 15-foot jumper") puts the rim centre at
y = 0.84, accurate to about a third of a foot, so the court below is drawn around that.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import game_length_seconds

RIM_X, RIM_Y = 25.0, 0.84
BASELINE_Y = RIM_Y - 5.25             # rim centre sits 5'3" in from the baseline
FT_LINE_Y = RIM_Y + 13.75             # free-throw line is 15 ft from the backboard
CORNER_Y = RIM_Y + math.sqrt(23.75 ** 2 - 22 ** 2)   # where the 3-pt arc meets the corners
HALF_COURT_Y = BASELINE_Y + 47

ZONES = ["Restricted area", "Paint (non-RA)", "Mid-range", "Corner 3", "Above-break 3"]


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------
def shots(pbp: pd.DataFrame) -> pd.DataFrame:
    """Every field-goal attempt, with distance and zone."""
    fga = pbp[pbp["shooting"].astype(bool) & ~pbp["type_text"].str.contains("Free Throw", case=False)].copy()
    for c in ("x", "y", "pts_att"):
        fga[c] = pd.to_numeric(fga[c], errors="coerce")
    fga["made"] = fga["scoring"].astype(bool)
    fga["three"] = fga["pts_att"] == 3
    stated = pd.to_numeric(fga["text"].str.extract(r"(\d+)-foot")[0], errors="coerce")
    from_xy = np.hypot(fga["x"] - RIM_X, fga["y"] - RIM_Y)
    fga["dist"] = stated.fillna(from_xy.round())
    fga["zone"] = [_zone(t, d, x, y) for t, d, x, y in zip(fga["three"], fga["dist"], fga["x"], fga["y"])]
    # End-of-quarter heaves are logged in the play-by-play but left out of official box
    # scores, so they're tagged and excluded from shot profiles (FGA then matches the box).
    fga.loc[fga["text"].str.contains("heave", case=False), "zone"] = "Heave"
    fga["blocked"] = fga["text"].str.contains("blocks", case=False)
    return fga


def _zone(three: bool, dist: float, x: float, y: float) -> str:
    has_xy = not (pd.isna(x) or pd.isna(y))
    if three:
        if has_xy and y > HALF_COURT_Y:
            return "Heave"
        if has_xy:
            return "Corner 3" if y <= CORNER_Y else "Above-break 3"
        return "Above-break 3"
    if pd.notna(dist) and dist <= 4:
        return "Restricted area"
    if has_xy:
        in_paint = abs(x - RIM_X) <= 8 and y <= FT_LINE_Y
        return "Paint (non-RA)" if in_paint else "Mid-range"
    return "Paint (non-RA)" if pd.notna(dist) and dist <= 14 else "Mid-range"


def zone_table(sh: pd.DataFrame) -> pd.DataFrame:
    """Attempts, share of attempts, and FG% by zone."""
    sh = sh[sh["zone"].isin(ZONES)]
    g = sh.groupby("zone").agg(fga=("made", "size"), fgm=("made", "sum"))
    g = g.reindex(ZONES).fillna(0)
    g["share"] = g["fga"] / max(g["fga"].sum(), 1)
    g["fg_pct"] = g["fgm"] / g["fga"].replace(0, np.nan)
    g["pts_per_shot"] = g["fg_pct"] * np.where(g.index.str.contains("3"), 3, 2)
    return g


def court_lines() -> pd.DataFrame:
    """Half-court markings as polylines (feet, same frame as the shot coordinates)."""
    segs = []

    def add(name, pts):
        segs.extend({"seg": name, "i": i, "x": float(px), "y": float(py)} for i, (px, py) in enumerate(pts))

    def arc(cx, cy, r, a0, a1, n=60):
        return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]

    add("boundary", [(0, BASELINE_Y), (50, BASELINE_Y), (50, HALF_COURT_Y), (0, HALF_COURT_Y), (0, BASELINE_Y)])
    add("lane", [(17, BASELINE_Y), (17, FT_LINE_Y), (33, FT_LINE_Y), (33, BASELINE_Y)])
    add("ft_circle", arc(RIM_X, FT_LINE_Y, 6, 0, 180))
    theta = math.degrees(math.acos(22 / 23.75))
    # the arc runs from the left corner (angle 180-theta) over the top to the right corner (theta)
    add("three",[(3, BASELINE_Y), (3, CORNER_Y)] + arc(RIM_X, RIM_Y, 23.75, 180 - theta, theta, 90) + [(47, CORNER_Y), (47, BASELINE_Y)])
    add("restricted", [(21, RIM_Y - 1.25)] + arc(RIM_X, RIM_Y, 4, 180, 0, 40) + [(29, RIM_Y - 1.25)])
    add("backboard", [(22, RIM_Y - 1.25), (28, RIM_Y - 1.25)])
    add("rim", arc(RIM_X, RIM_Y, 0.75, 0, 360, 30))
    add("center_circle", arc(RIM_X, HALF_COURT_Y, 6, 180, 360))
    return pd.DataFrame(segs)


# ---------------------------------------------------------------------------
# game flow
# ---------------------------------------------------------------------------
def flow(game_pbp: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """The score margin (team minus opponent) after every scoring play."""
    p = game_pbp.sort_values("seq")
    home = int(p["home_id"].iloc[0]) == team_id
    sc = p[p["scoring"]].copy()
    sc["team_score"] = sc["home_score"] if home else sc["away_score"]
    sc["opp_score"] = sc["away_score"] if home else sc["home_score"]
    sc["margin"] = sc["team_score"] - sc["opp_score"]
    periods = int(p["period"].max())
    start = pd.DataFrame([{"t": 0.0, "team_score": 0, "opp_score": 0, "margin": 0, "text": "Tip-off", "period": 1, "clock": "12:00"}])
    end_row = sc.tail(1)[["team_score", "opp_score", "margin"]].to_dict("records")
    end = pd.DataFrame([{**(end_row[0] if end_row else {"team_score": 0, "opp_score": 0, "margin": 0}),
                         "t": float(game_length_seconds(periods)), "text": "Final", "period": periods, "clock": "0:00"}])
    out = pd.concat([start, sc[["t", "team_score", "opp_score", "margin", "text", "period", "clock"]], end], ignore_index=True)
    out["minute"] = out["t"] / 60
    return out


def lead_stats(fl: pd.DataFrame) -> dict:
    """Largest leads, lead changes, ties."""
    m = fl["margin"].values
    sign = np.sign(m)
    nz = sign[sign != 0]
    lead_changes = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    ties = int(((m[1:] == 0) & (m[:-1] != 0)).sum()) if len(m) > 1 else 0
    ahead = fl["t"].diff().shift(-1).fillna(0)
    total = max(fl["t"].iloc[-1], 1)
    return {"max_lead": int(max(m.max(), 0)), "max_deficit": int(max(-m.min(), 0)),
            "lead_changes": lead_changes, "ties": ties,
            "time_leading": float(ahead[fl["margin"] > 0].sum() / total),
            "time_trailing": float(ahead[fl["margin"] < 0].sum() / total)}


def runs(game_pbp: pd.DataFrame, team_id: int) -> list[dict]:
    """Unanswered scoring runs, biggest first."""
    p = game_pbp.sort_values("seq")
    sc = p[p["scoring"] & (p["value"] > 0)]
    out, cur = [], None
    for _, r in sc.iterrows():
        side = "team" if int(r["team_id"]) == team_id else "opp"
        if cur and cur["side"] == side:
            cur["pts"] += int(r["value"])
            cur["end_t"], cur["end_period"], cur["end_clock"] = r["t"], int(r["period"]), r["clock"]
        else:
            if cur:
                out.append(cur)
            cur = {"side": side, "pts": int(r["value"]), "start_t": r["t"], "start_period": int(r["period"]),
                   "start_clock": r["clock"], "end_t": r["t"], "end_period": int(r["period"]), "end_clock": r["clock"]}
    if cur:
        out.append(cur)
    return sorted(out, key=lambda d: -d["pts"])


def clutch(game_pbp: pd.DataFrame, team_id: int) -> dict | None:
    """NBA definition: last five minutes of the 4th quarter or overtime, score within five.
    Returns points for/against scored while the game was in that state (None if it never was)."""
    p = game_pbp.sort_values("seq").copy()
    home = int(p["home_id"].iloc[0]) == team_id
    p["margin_before"] = ((p["home_score"] - p["away_score"]) * (1 if home else -1)).shift(1).fillna(0)
    left = [_period_left(per, t) for per, t in zip(p["period"], p["t"])]
    p["late"] = (p["period"] >= 4) & (pd.Series(left, index=p.index) <= 300)
    c = p[p["late"] & (p["margin_before"].abs() <= 5)]
    if not len(c):
        return None
    sc = c[c["scoring"]]
    mine = sc["team_id"].astype("float") == team_id
    return {"pts_for": int(sc.loc[mine, "value"].sum()), "pts_against": int(sc.loc[~mine, "value"].sum()),
            "scorers": sc[mine].groupby("a1")["value"].sum().sort_values(ascending=False).to_dict()}


def _period_left(period: int, t: float) -> float:
    if period <= 4:
        return period * 720 - t
    return 2880 + (period - 4) * 300 - t


def fmt_clock(period: int, clock: str) -> str:
    q = f"Q{period}" if period <= 4 else ("OT" if period == 5 else f"{period - 4}OT")
    return f"{q} {clock}"
