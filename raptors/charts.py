"""Charts (Altair / Vega-Lite).

Each chart is built once per theme (light + dark) for the HTML reports, where it's
interactive (hover for details), and once more as a PNG for posting.

Colour rules used everywhere:
  * The Raptors are the only coloured series (red); opponents and context are grey.
  * "Helped" vs "hurt" uses blue vs orange (never red, which means Raptors here).
  * The palettes were checked for colour-blind separation and contrast in both themes.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

import config
from . import pbp as P

alt.data_transformers.disable_max_rows()

FONT = "Barlow, 'Helvetica Neue', Arial, sans-serif"
THEMES = {
    "light": dict(surface="#ffffff", ink="#141417", ink2="#55545c", muted="#85848f", grid="#ebeaef",
                  axis="#c9c8d0", accent="#CE1141", context="#8b8a95", up="#2a78d6", down="#eb6834",
                  wash=0.10, court="#d6d5dd"),
    "dark": dict(surface="#1b1b1f", ink="#f3f3f5", ink2="#c5c4cc", muted="#8f8e97", grid="#2a2a30",
                 axis="#3b3b42", accent="#ee3d5a", context="#8e8d98", up="#3987e5", down="#d95926",
                 wash=0.16, court="#3a3a42"),
}
MISS_SHAPE = "M-0.8,-0.8L0.8,0.8M-0.8,0.8L0.8,-0.8"   # an "x"


def vl_config(th: dict) -> dict:
    return {
        "background": th["surface"],
        "padding": {"left": 4, "right": 12, "top": 8, "bottom": 4},
        "view": {"stroke": None},
        "font": FONT,
        "axis": {"labelColor": th["muted"], "titleColor": th["ink2"], "gridColor": th["grid"],
                 "domainColor": th["axis"], "tickColor": th["axis"], "labelFontSize": 11,
                 "titleFontSize": 11, "titleFontWeight": 500, "gridWidth": 1, "domainWidth": 1,
                 "labelFont": FONT, "titleFont": FONT, "titlePadding": 8},
        "legend": {"labelColor": th["ink2"], "titleColor": th["ink2"], "labelFont": FONT,
                   "titleFont": FONT, "labelFontSize": 12, "symbolStrokeWidth": 2},
        "title": {"color": th["ink"], "subtitleColor": th["ink2"], "font": FONT, "subtitleFont": FONT,
                  "anchor": "start", "fontSize": 18, "fontWeight": 600, "subtitleFontSize": 12,
                  "subtitlePadding": 6, "offset": 14},
        "text": {"color": th["ink2"], "font": FONT, "fontSize": 11},
    }


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def on_fill(hex_color: str, dark: str = "#141417", light: str = "#ffffff") -> str:
    """Text colour for a label sitting on a coloured fill: whichever contrasts more."""
    L = _luminance(hex_color)
    c_dark = (L + 0.05) / (_luminance(dark) + 0.05)
    c_light = (_luminance(light) + 0.05) / (L + 0.05)
    return dark if c_dark >= c_light else light


def signed(expr: str = "datum.value") -> str:
    return f"({expr} > 0 ? '+' : '') + format({expr}, 'd')"


# ---------------------------------------------------------------------------
# game charts
# ---------------------------------------------------------------------------
def flow_chart(fl: pd.DataFrame, team: str, opp: str, th: dict, width="container", height=260):
    """Score margin through the game (above zero = Raptors ahead)."""
    d = fl.copy()
    d["pos"] = d["margin"].clip(lower=0)
    d["neg"] = d["margin"].clip(upper=0)
    d["when"] = [P.fmt_clock(p, c) if txt not in ("Tip-off", "Final") else txt
                 for p, c, txt in zip(d["period"], d["clock"], d["text"])]
    d["score"] = [f"{team} {a} – {b} {opp}" for a, b in zip(d["team_score"], d["opp_score"])]
    d["lead"] = [f"{team} +{m}" if m > 0 else (f"{opp} +{-m}" if m < 0 else "Tied") for m in d["margin"]]
    end = float(d["minute"].max())
    lim = max(10, int(np.ceil(d["margin"].abs().max() / 5.0) * 5))
    x = alt.X("minute:Q", scale=alt.Scale(domain=[0, end], nice=False), title=None,
              axis=alt.Axis(values=_mid_quarters(end), labelExpr=_quarter_label_expr(), ticks=False,
                            grid=False, domain=False, labelPadding=6))
    y = alt.Y("margin:Q", scale=alt.Scale(domain=[-lim, lim]), title=None,
              axis=alt.Axis(labelExpr=signed(), tickCount=5, domain=False, ticks=False))
    base = alt.Chart(d).encode(x=x)
    # stack=None matters: two scoring plays at the same second (an and-one) would otherwise stack
    ys = alt.Scale(domain=[-lim, lim])
    pos = base.mark_area(interpolate="step-after", color=th["accent"], opacity=th["wash"]).encode(
        y=alt.Y("pos:Q", stack=None, scale=ys))
    neg = base.mark_area(interpolate="step-after", color=th["context"], opacity=th["wash"] + 0.04).encode(
        y=alt.Y("neg:Q", stack=None, scale=ys))
    line = base.mark_line(interpolate="step-after", color=th["ink2"], strokeWidth=2, strokeJoin="round").encode(y=y)
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=th["axis"], strokeWidth=1).encode(y="z:Q")
    bounds = [b for b in [12, 24, 36, 48, 53, 58, 63] if b < end]
    qrule = alt.Chart(pd.DataFrame({"m": bounds})).mark_rule(color=th["grid"], strokeWidth=1).encode(x="m:Q")
    side = pd.DataFrame({"minute": [0.4, 0.4], "margin": [lim * 0.86, -lim * 0.86],
                         "label": [f"{team} ahead", f"{opp} ahead"]})
    side_lbl = alt.Chart(side).mark_text(align="left", fontSize=11, fontWeight=500, color=th["muted"]).encode(
        x="minute:Q", y="margin:Q", text="label:N")
    # direct labels: biggest lead each way + final
    marks = []
    if d["margin"].max() > 0:
        i = d["margin"].idxmax(); marks.append((d.loc[i, "minute"], d.loc[i, "margin"], f"+{d.loc[i, 'margin']}", "bottom"))
    if d["margin"].min() < 0:
        i = d["margin"].idxmin(); marks.append((d.loc[i, "minute"], d.loc[i, "margin"], f"−{-d.loc[i, 'margin']}", "top"))
    ann = pd.DataFrame(marks, columns=["minute", "margin", "label", "vpos"]) if marks else pd.DataFrame(columns=["minute", "margin", "label", "vpos"])
    dots = alt.Chart(ann).mark_point(filled=True, size=64, color=th["ink2"], stroke=th["surface"], strokeWidth=2).encode(
        x="minute:Q", y="margin:Q")
    up = alt.Chart(ann[ann.vpos == "bottom"]).mark_text(dy=-12, fontSize=12, fontWeight=600, color=th["ink"]).encode(
        x="minute:Q", y="margin:Q", text="label:N")
    down = alt.Chart(ann[ann.vpos == "top"]).mark_text(dy=14, fontSize=12, fontWeight=600, color=th["ink"]).encode(
        x="minute:Q", y="margin:Q", text="label:N")
    # hover: nearest scoring play
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["minute"], empty=False, clear="pointerout")
    catcher = base.mark_rule(opacity=0, strokeWidth=12).encode(
        tooltip=[alt.Tooltip("when:N", title="Time"), alt.Tooltip("score:N", title="Score"),
                 alt.Tooltip("lead:N", title="Lead"), alt.Tooltip("text:N", title="Play")]).add_params(hover)
    rule = base.mark_rule(color=th["muted"], strokeWidth=1).encode().transform_filter(hover)
    point = base.mark_point(filled=True, size=70, color=th["ink"], stroke=th["surface"], strokeWidth=2).encode(y="margin:Q").transform_filter(hover)
    layers = [pos, neg, zero, qrule, line, side_lbl, dots, up, down, catcher, rule, point]
    return alt.layer(*layers).properties(width=width, height=height)


def _mid_quarters(end_minute: float) -> list[float]:
    mids = [6, 18, 30, 42]
    ot = 48
    while ot < end_minute - 0.1:
        mids.append(ot + 2.5)
        ot += 5
    return mids


def _quarter_label_expr() -> str:
    return ("datum.value < 12 ? 'Q1' : datum.value < 24 ? 'Q2' : datum.value < 36 ? 'Q3' : "
            "datum.value < 48 ? 'Q4' : datum.value < 53 ? 'OT' : datum.value < 58 ? '2OT' : '3OT'")


def factors_chart(contrib: pd.DataFrame, residual: float, team: str, opp: str, th: dict, width="container"):
    """How many points per 100 possessions each factor was worth to the Raptors."""
    d = contrib[["key", "factor", "points", "team", "opp"]].copy()
    short = {"efg": "Shooting", "tov": "Turnovers", "orb": "Offensive rebounds", "ftr": "Free throws"}
    fmt = {"efg": "{:.1%}", "tov": "{:.1%}", "orb": "{:.1%}", "ftr": "{:.3f}"}
    d["detail"] = [f"{f}: {team} {fmt[k].format(a)} · {opp} {fmt[k].format(b)}" for k, f, a, b in zip(d.key, d.factor, d.team, d.opp)]
    d["factor"] = d["key"].map(short)
    d = pd.concat([d, pd.DataFrame([{"factor": "Everything else", "points": residual, "detail": "What the four factors don't explain"}])],
                  ignore_index=True)
    order = list(d["factor"])
    d["label"] = [("+" if v >= 0 else "−") + f"{abs(v):.1f}" for v in d["points"]]
    d["kind"] = ["other" if f == "Everything else" else ("up" if v >= 0 else "down") for f, v in zip(d.factor, d.points)]
    lim = max(6.0, float(d["points"].abs().max()) * 1.3)
    x = alt.X("points:Q", scale=alt.Scale(domain=[-lim, lim]), title="Points per 100 possessions",
              axis=alt.Axis(labelExpr=signed(), tickCount=7, grid=True, domain=False, ticks=False))
    y = alt.Y("factor:N", sort=order, title=None, axis=alt.Axis(domain=False, ticks=False, labelFontSize=12,
                                                               labelColor=th["ink2"], labelLimit=150, labelPadding=8))
    tip = [alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("detail:N", title="Values"),
           alt.Tooltip("label:N", title="Worth")]
    color = alt.Color("kind:N", scale=alt.Scale(domain=["up", "down", "other"], range=[th["up"], th["down"], th["context"]]), legend=None)
    base = alt.Chart(d).encode(y=y, tooltip=tip)
    posb = base.transform_filter("datum.points >= 0").mark_bar(size=18, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(x=x, color=color)
    negb = base.transform_filter("datum.points < 0").mark_bar(size=18, cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4).encode(x=x, color=color)
    # Value labels sit just past the bar end; long bars carry them inside instead, so a label
    # never runs into the axis labels on a narrow screen. Inside text is white or ink, whichever
    # contrasts more with that bar's fill.
    d["inside"] = d["points"].abs() >= 0.5 * lim
    d["ink_inside"] = d["kind"].map({"up": on_fill(th["up"]), "down": on_fill(th["down"]), "other": on_fill(th["context"])})
    # (its own channel with raw colour values: sharing the bars' colour scale would paint the text bar-coloured)
    ink_on = alt.Fill("ink_inside:N", scale=None, legend=None)
    lbl = alt.Chart(d).encode(y=y, x="points:Q", text="label:N", tooltip=tip)
    out_pos = lbl.transform_filter("datum.points >= 0 && !datum.inside").mark_text(align="left", dx=6, fontSize=12, fontWeight=600, color=th["ink"])
    out_neg = lbl.transform_filter("datum.points < 0 && !datum.inside").mark_text(align="right", dx=-6, fontSize=12, fontWeight=600, color=th["ink"])
    in_pos = lbl.transform_filter("datum.points >= 0 && datum.inside").mark_text(align="right", dx=-7, fontSize=12, fontWeight=700).encode(fill=ink_on)
    in_neg = lbl.transform_filter("datum.points < 0 && datum.inside").mark_text(align="left", dx=7, fontSize=12, fontWeight=700).encode(fill=ink_on)
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=th["axis"], strokeWidth=1).encode(x="z:Q")
    return alt.layer(posb, negb, zero, out_pos, out_neg, in_pos, in_neg).properties(width=width, height=alt.Step(40))


def shot_chart(sh: pd.DataFrame, names: dict, th: dict, width=500):
    """Every shot on a half court: filled dot = make, x = miss."""
    top = P.BASELINE_Y + 36
    d = sh[sh["zone"].isin(P.ZONES) & sh["x"].notna() & (sh["y"] <= top)].copy()
    d["Result"] = np.where(d["made"], "Made", "Missed")
    d["player"] = d["a1"].map(lambda a: names.get(int(a), "") if pd.notna(a) else "")
    d["when"] = [P.fmt_clock(p, c) for p, c in zip(d["period"], d["clock"])]
    d["dist_ft"] = d["dist"].map(lambda v: f"{int(v)} ft" if pd.notna(v) else "")
    court = P.court_lines()
    height = int(width * (top - P.BASELINE_Y) / 50)
    xs = alt.Scale(domain=[0, 50], nice=False, zero=False)
    ys = alt.Scale(domain=[P.BASELINE_Y, top], nice=False, zero=False)
    lines = alt.Chart(court).mark_line(color=th["court"], strokeWidth=1.2, clip=True).encode(
        x=alt.X("x:Q", scale=xs, axis=None), y=alt.Y("y:Q", scale=ys, axis=None), detail="seg:N", order="i:Q")
    pts = alt.Chart(d).mark_point(size=58, strokeWidth=1.6, opacity=0.95, clip=True).encode(
        x=alt.X("x:Q", scale=xs, axis=None), y=alt.Y("y:Q", scale=ys, axis=None),
        shape=alt.Shape("Result:N", scale=alt.Scale(domain=["Made", "Missed"], range=["circle", MISS_SHAPE]),
                        legend=alt.Legend(title=None, orient="top-left", direction="horizontal", symbolSize=90,
                                          fillColor=th["surface"], padding=6)),
        fill=alt.Fill("Result:N", scale=alt.Scale(domain=["Made", "Missed"], range=[th["accent"], "transparent"]), legend=None),
        stroke=alt.Stroke("Result:N", scale=alt.Scale(domain=["Made", "Missed"], range=[th["surface"], th["context"]]), legend=None),
        tooltip=[alt.Tooltip("player:N", title="Player"), alt.Tooltip("Result:N"), alt.Tooltip("zone:N", title="Zone"),
                 alt.Tooltip("dist_ft:N", title="Distance"), alt.Tooltip("when:N", title="When"),
                 alt.Tooltip("text:N", title="Play")])
    # draw misses under makes
    return alt.layer(lines, pts.transform_filter("datum.Result == 'Missed'"),
                     pts.transform_filter("datum.Result == 'Made'")).resolve_scale(shape="shared").properties(width=width, height=height)


# ---------------------------------------------------------------------------
# season charts
# ---------------------------------------------------------------------------
def rolling_chart(g: pd.DataFrame, th: dict, window: int = 10, width="container", height=240):
    """Net rating game by game (grey) and over the last `window` games (red)."""
    d = g[["game_no", "game_date", "opp_abbr", "home_away", "win", "pts", "opp_pts", "net", "roll_net"]].copy()
    d["matchup"] = [("vs " if h == "home" else "at ") + o for h, o in zip(d.home_away, d.opp_abbr)]
    d["result"] = [f"{'W' if w else 'L'} {a}-{b}" for w, a, b in zip(d.win, d.pts, d.opp_pts)]
    d["net_s"] = d["net"].map(lambda v: f"{v:+.1f}")
    d["roll_s"] = d["roll_net"].map(lambda v: f"{v:+.1f}")
    lim = max(20, int(np.ceil(d["net"].abs().max() / 10.0) * 10))
    n = int(d["game_no"].max())
    x = alt.X("game_no:Q", title="Game", scale=alt.Scale(domain=[0.5, max(n, 10) + 0.5], nice=False),
              axis=alt.Axis(tickMinStep=1, grid=False, domain=False, labelFlush=True))
    y = alt.Y("net:Q", title="Net rating (points per 100 possessions)", scale=alt.Scale(domain=[-lim, lim]),
              axis=alt.Axis(labelExpr=signed(), tickCount=5, domain=False, ticks=False))
    tip = [alt.Tooltip("game_no:Q", title="Game"), alt.Tooltip("game_date:N", title="Date"),
           alt.Tooltip("matchup:N", title="Opponent"), alt.Tooltip("result:N", title="Result"),
           alt.Tooltip("net_s:N", title="Net rating"), alt.Tooltip("roll_s:N", title=f"Last {window}")]
    series = alt.Color("series:N", scale=alt.Scale(domain=["Single game", f"Last {window} games"],
                                                   range=[th["context"], th["accent"]]),
                       legend=alt.Legend(title=None, orient="top-left", direction="horizontal", symbolType="stroke"))
    bars = alt.Chart(d).transform_calculate(series="'Single game'").mark_bar(
        size=max(2, min(8, int(560 / max(n, 1)) - 2)), opacity=0.55, cornerRadius=1).encode(x=x, y=y, color=series, tooltip=tip)
    line = alt.Chart(d).transform_calculate(series=f"'Last {window} games'").mark_line(
        strokeWidth=2.5, interpolate="monotone").encode(x="game_no:Q", y="roll_net:Q", color=series, tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=th["axis"]).encode(y="z:Q")
    last = d.tail(1).assign(lbl=lambda t: t["roll_net"].map(lambda v: f"{v:+.1f}"))
    end_dot = alt.Chart(last).mark_point(filled=True, size=70, color=th["accent"], stroke=th["surface"], strokeWidth=2).encode(
        x="game_no:Q", y="roll_net:Q")
    end_lbl = alt.Chart(last).mark_text(align="left", dx=8, fontSize=12, fontWeight=600, color=th["ink"]).encode(
        x="game_no:Q", y="roll_net:Q", text="lbl:N")
    return alt.layer(zero, bars, line, end_dot, end_lbl).properties(width=width, height=height)


def league_chart(ts: pd.DataFrame, team: str, th: dict, width=520, height=460):
    """Every team's offense (x) vs defense (y, flipped so up = better)."""
    d = ts.reset_index()[["team_abbr", "w", "l", "ortg", "drtg", "net", "ortg_rank", "drtg_rank", "net_rank"]].copy()
    d["is_team"] = d["team_abbr"] == team
    d["record"] = [f"{w}-{l}" for w, l in zip(d.w, d.l)]
    d["o"] = [f"{v:.1f} (#{r})" for v, r in zip(d.ortg, d.ortg_rank)]
    d["dd"] = [f"{v:.1f} (#{r})" for v, r in zip(d.drtg, d.drtg_rank)]
    d["n"] = [f"{v:+.1f} (#{r})" for v, r in zip(d.net, d.net_rank)]
    pad = 1.5
    xs = alt.Scale(domain=[float(d.ortg.min() - pad), float(d.ortg.max() + pad)], nice=False, zero=False)
    ys = alt.Scale(domain=[float(d.drtg.max() + pad), float(d.drtg.min() - pad)], nice=False, zero=False)
    x = alt.X("ortg:Q", scale=xs, title="Offensive rating (points scored per 100 possessions) →", axis=alt.Axis(domain=False, ticks=False))
    y = alt.Y("drtg:Q", scale=ys, title="Defensive rating (allowed per 100, flipped) →", axis=alt.Axis(domain=False, ticks=False))
    tip = [alt.Tooltip("team_abbr:N", title="Team"), alt.Tooltip("record:N", title="Record"),
           alt.Tooltip("o:N", title="Offense"), alt.Tooltip("dd:N", title="Defense"), alt.Tooltip("n:N", title="Net")]
    avg_o, avg_d = float(np.average(d.ortg)), float(np.average(d.drtg))
    vrule = alt.Chart(pd.DataFrame({"v": [avg_o]})).mark_rule(color=th["axis"]).encode(x=alt.X("v:Q", scale=xs))
    hrule = alt.Chart(pd.DataFrame({"v": [avg_d]})).mark_rule(color=th["axis"]).encode(y=alt.Y("v:Q", scale=ys))
    quad = pd.DataFrame({"ortg": [float(d.ortg.max() + pad * 0.8)], "drtg": [float(d.drtg.min() - pad * 0.7)], "t": ["Good offense, good defense"]})
    quad_lbl = alt.Chart(quad).mark_text(align="right", fontSize=11, color=th["muted"]).encode(x=alt.X("ortg:Q", scale=xs), y=alt.Y("drtg:Q", scale=ys), text="t:N")
    others = alt.Chart(d[~d.is_team]).mark_point(filled=True, size=60, color=th["context"], stroke=th["surface"], strokeWidth=2, opacity=0.9).encode(x=x, y=y, tooltip=tip)
    me = alt.Chart(d[d.is_team]).mark_point(filled=True, size=150, color=th["accent"], stroke=th["surface"], strokeWidth=2).encode(x=x, y=y, tooltip=tip)
    # labels: placed one by one wherever they don't collide with a dot or an earlier label
    lab = _place_labels(d, team, xs.domain, ys.domain, width, height)
    lx = alt.X("lx:Q", scale=xs)
    ly = alt.Y("ly:Q", scale=ys)
    layers = [vrule, hrule, quad_lbl, others, me]
    for align in ("left", "center", "right"):
        sub = lab[(lab["align"] == align) & ~lab["is_team"]]
        if len(sub):
            layers.append(alt.Chart(sub).mark_text(align=align, baseline="middle", fontSize=10, color=th["muted"]).encode(x=lx, y=ly, text="team_abbr:N"))
    mine = lab[lab["is_team"]]
    layers.append(alt.Chart(mine).mark_text(align=mine["align"].iloc[0] if len(mine) else "center", baseline="middle",
                                            fontSize=13, fontWeight=700, color=th["ink"]).encode(x=lx, y=ly, text="team_abbr:N"))
    return alt.layer(*layers).properties(width=width, height=height)


def _place_labels(d: pd.DataFrame, team: str, xdom, ydom, width: int, height: int) -> pd.DataFrame:
    """Greedy label placement for a scatter: for each point (the highlighted team first), try
    positions around the dot and keep the first whose box hits no dot and no placed label."""
    x0, x1 = xdom
    y_top, y_bot = ydom[1], ydom[0]           # the y scale is flipped: small values at the top
    px = lambda v: (v - x0) / (x1 - x0) * width
    py = lambda v: (v - y_top) / (y_bot - y_top) * height
    dots = [(px(a), py(b)) for a, b in zip(d.ortg, d.drtg)]
    boxes = [(cx - 5, cy - 5, cx + 5, cy + 5) for cx, cy in dots]
    boxes.append((width - 190, 0, width, 16))  # the corner caption
    cands = [(0, -12, "center"), (8, 0, "left"), (-8, 0, "right"), (0, 13, "center"),
             (7, -9, "left"), (-7, -9, "right"), (7, 9, "left"), (-7, 9, "right")]
    order = sorted(range(len(d)), key=lambda i: (d.iloc[i]["team_abbr"] != team, i))
    out = []
    for i in order:
        r = d.iloc[i]
        big = r["team_abbr"] == team
        w, h = (8.6 if big else 6.4) * len(r["team_abbr"]), (15 if big else 11)
        cx, cy = dots[i]
        best = None
        for dx, dy, align in (cands if not big else [(0, -15, "center")] + cands):
            ax, ay = cx + dx, cy + dy
            left = ax - w / 2 if align == "center" else (ax if align == "left" else ax - w)
            box = (left, ay - h / 2, left + w, ay + h / 2)
            inside = box[0] >= 0 and box[2] <= width and box[1] >= 0 and box[3] <= height
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                      for j, b in enumerate(boxes) if j != i)
            if inside and not hit:
                best = (ax, ay, align, box)
                break
        if best is None:  # nowhere free: fall back to above the dot
            ax, ay, align = cx, cy - 12, "center"
            best = (ax, ay, align, (ax - w / 2, ay - h / 2, ax + w / 2, ay + h / 2))
        ax, ay, align, box = best
        boxes.append(box)
        out.append({"team_abbr": r["team_abbr"], "is_team": big, "align": align,
                    "lx": x0 + ax / width * (x1 - x0), "ly": y_top + ay / height * (y_bot - y_top)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def both_themes(builder, *args, **kwargs) -> str:
    """JSON with a light and a dark spec, for the HTML report's theme switcher."""
    out = {}
    for name, th in THEMES.items():
        ch = builder(*args, th=th, **kwargs).configure(**vl_config(th))
        out[name] = ch.to_dict()
    return json.dumps(out, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


_fonts_registered = False


def save_png(builder, path: Path, *args, title: str = "", subtitle: str = "", width: int = 720, **kwargs) -> Path | None:
    """Export a chart as a 2x PNG with a title and credit line, ready to post."""
    global _fonts_registered
    try:
        import vl_convert as vlc
    except ImportError:
        return None
    fonts = config.ROOT / "assets" / "fonts"
    if not _fonts_registered and fonts.exists():
        vlc.register_font_directory(str(fonts))
        _fonts_registered = True
    th = THEMES["light"]
    kwargs.setdefault("width", width)
    ch = builder(*args, th=th, **kwargs)
    credit = f"{config.CREDIT} · {config.DATA_CREDIT}" if config.CREDIT else config.DATA_CREDIT
    sub = [s for s in [subtitle, credit] if s]
    cfg = vl_config(th)
    cfg["padding"] = {"left": 16, "right": 24, "top": 16, "bottom": 12}
    ch = ch.properties(title=alt.TitleParams(title, subtitle=sub)).configure(**cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        png = vlc.vegalite_to_png(ch.to_dict(), scale=2)
    except Exception as e:  # never let a PNG failure break the report
        print(f"  · couldn't export {path.name}: {e}")
        return None
    path.write_bytes(png)
    return path
