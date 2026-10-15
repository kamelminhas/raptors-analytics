"""Turning box scores into analysis-grade numbers.

Everything here is standard public basketball analytics (Dean Oliver's four factors,
Basketball-Reference's possession estimate, Hollinger's Game Score), written out
explicitly so you can explain every formula in an interview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PLAY_IN, PLAYOFFS, REGULAR


# ---------------------------------------------------------------------------
# team, one game at a time
# ---------------------------------------------------------------------------
def team_games(team_box: pd.DataFrame, player_box: pd.DataFrame | None = None,
               pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team per game, with the opponent's box score alongside (o_*)
    and every rate stat computed."""
    tb = team_box.copy()
    opp_cols = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
                "ast", "stl", "blk", "tov", "pf", "pts_paint", "pts_fb"]
    opp = tb[["game_id", "team_id"] + opp_cols].rename(columns={c: f"o_{c}" for c in opp_cols})
    opp = opp.rename(columns={"team_id": "opp_id"})
    df = tb.merge(opp, on=["game_id", "opp_id"], how="inner")

    # Periods played (for pace): from the play-by-play if we have it, else from player minutes
    periods = pd.Series(np.nan, index=df.index)
    if pbp is not None and len(pbp):
        per = pbp.groupby("game_id")["period"].max()
        periods = pd.to_numeric(df["game_id"].map(per), errors="coerce")
    if player_box is not None and len(player_box):
        mins = player_box[~player_box["dnp"]].groupby(["game_id", "team_id"])["min"].sum()
        est = ((mins - 240) / 25).round().clip(lower=0) + 4
        by_min = pd.Series([est.get((g, t), np.nan) for g, t in zip(df["game_id"], df["team_id"])], index=df.index)
        periods = periods.fillna(by_min)
    df["periods"] = periods.fillna(4).clip(lower=4).astype(int)
    df["minutes"] = 48 + 5 * (df["periods"] - 4)

    # Possessions: Basketball-Reference's estimate, averaged over both teams so that
    # both sides of a game get the same count (one team's ORtg = the other's DRtg).
    def poss_est(fga, fta, oreb, opp_dreb, fgm, tov):
        orb_share = oreb / (oreb + opp_dreb).replace(0, np.nan)
        return fga + 0.4 * fta - 1.07 * orb_share * (fga - fgm) + tov
    own = poss_est(df.fga, df.fta, df.oreb, df.o_dreb, df.fgm, df.tov)
    other = poss_est(df.o_fga, df.o_fta, df.o_oreb, df.dreb, df.o_fgm, df.o_tov)
    df["poss"] = 0.5 * (own + other)
    df["pace"] = 48 * df["poss"] / df["minutes"]
    df["ortg"] = 100 * df["pts"] / df["poss"]
    df["drtg"] = 100 * df["opp_pts"] / df["poss"]
    df["net"] = df["ortg"] - df["drtg"]
    df["margin"] = df["pts"] - df["opp_pts"]

    # The four factors (offense), then the same four for the opponent (= our defense)
    df["efg"] = (df.fgm + 0.5 * df.fg3m) / df.fga
    df["tov_pct"] = df.tov / (df.fga + 0.44 * df.fta + df.tov)
    df["orb_pct"] = df.oreb / (df.oreb + df.o_dreb)
    df["ftr"] = df.ftm / df.fga
    df["o_efg"] = (df.o_fgm + 0.5 * df.o_fg3m) / df.o_fga
    df["o_tov_pct"] = df.o_tov / (df.o_fga + 0.44 * df.o_fta + df.o_tov)
    df["o_orb_pct"] = df.o_oreb / (df.o_oreb + df.dreb)
    df["o_ftr"] = df.o_ftm / df.o_fga
    df["drb_pct"] = 1 - df["o_orb_pct"]

    # Shooting detail
    df["ts"] = df.pts / (2 * (df.fga + 0.44 * df.fta))
    df["fg3_pct"] = df.fg3m / df.fg3a.replace(0, np.nan)
    df["fg3a_rate"] = df.fg3a / df.fga
    df["o_fg3_pct"] = df.o_fg3m / df.o_fg3a.replace(0, np.nan)
    df["ft_pct"] = df.ftm / df.fta.replace(0, np.nan)
    df["ast_rate"] = df.ast / df.fgm
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


FACTORS = [  # (key, label, team column, opponent column, higher is better for the team?)
    ("efg", "Shooting (eFG%)", "efg", "o_efg"),
    ("tov", "Ball security (TOV%)", "tov_pct", "o_tov_pct"),
    ("orb", "Offensive rebounding (ORB%)", "orb_pct", "o_orb_pct"),
    ("ftr", "Getting to the line (FT/FGA)", "ftr", "o_ftr"),
]


def factor_diffs(tg: pd.DataFrame) -> pd.DataFrame:
    """Team minus opponent on each factor, in percentage points."""
    return pd.DataFrame({key: 100 * (tg[a] - tg[b]) for key, _, a, b in FACTORS}, index=tg.index)


def fit_four_factors(tg: pd.DataFrame) -> dict:
    """How many points (per 100 possessions) is each factor worth?

    Ordinary least squares across every team-game in the sample:
        net rating = b0 + b1*eFG diff + b2*TOV diff + b3*ORB diff + b4*FT-rate diff
    The fitted b's turn a game's factor gaps into points, so each game's net rating can be
    split into "shooting was worth +6, turnovers -2, ..." with a small residual left over.
    """
    X = factor_diffs(tg).astype(float)
    y = tg["net"].astype(float)
    ok = X.notna().all(axis=1) & y.notna()
    A = np.column_stack([np.ones(ok.sum()), X[ok].values])
    coef, *_ = np.linalg.lstsq(A, y[ok].values, rcond=None)
    pred = A @ coef
    r2 = 1 - ((y[ok].values - pred) ** 2).sum() / ((y[ok].values - y[ok].mean()) ** 2).sum()
    return {"intercept": coef[0], "coef": dict(zip(X.columns, coef[1:])), "r2": r2, "n": int(ok.sum())}


def factor_contributions(row: pd.Series, model: dict) -> pd.DataFrame:
    """Split one game's net rating into the four factors (+ what they don't explain)."""
    out = []
    for key, label, a, b in FACTORS:
        diff = 100 * (row[a] - row[b])
        out.append({"key": key, "factor": label, "team": row[a], "opp": row[b], "diff": diff,
                    "points": model["coef"][key] * diff})
    df = pd.DataFrame(out)
    explained = model["intercept"] + df["points"].sum()
    df.attrs["residual"] = row["net"] - explained
    return df


# ---------------------------------------------------------------------------
# players, one game at a time
# ---------------------------------------------------------------------------
def player_games(player_box: pd.DataFrame) -> pd.DataFrame:
    """Per player per game: true shooting, usage, Game Score."""
    pb = player_box[~player_box["dnp"]].copy()
    team = pb.groupby(["game_id", "team_id"])[["min", "fga", "fta", "tov"]].sum().add_prefix("t_")
    pb = pb.join(team, on=["game_id", "team_id"])
    pb["ts"] = pb.pts / (2 * (pb.fga + 0.44 * pb.fta)).replace(0, np.nan)
    pb["efg"] = (pb.fgm + 0.5 * pb.fg3m) / pb.fga.replace(0, np.nan)
    # Usage: share of team plays a player "used" (shot, trip to the line, turnover) while on court
    pb["usg"] = 100 * (pb.fga + 0.44 * pb.fta + pb.tov) * (pb.t_min / 5) / \
        (pb["min"] * (pb.t_fga + 0.44 * pb.t_fta + pb.t_tov)).replace(0, np.nan)
    # Hollinger's Game Score: one number for a box-score line, scaled like points
    pb["gmsc"] = (pb.pts + 0.4 * pb.fgm - 0.7 * pb.fga - 0.4 * (pb.fta - pb.ftm) + 0.7 * pb.oreb
                  + 0.3 * pb.dreb + pb.stl + 0.7 * pb.ast + 0.7 * pb.blk - 0.4 * pb.pf - pb.tov)
    return pb


def player_summary(pg: pd.DataFrame) -> pd.DataFrame:
    """Season-style totals and averages per player from a set of player-games."""
    if not len(pg):
        return pd.DataFrame()
    agg = pg.groupby("athlete_id").agg(
        name=("name", "last"), short_name=("short_name", "last"), team_abbr=("team_abbr", "last"),
        gp=("game_id", "nunique"), gs=("starter", "sum"), min=("min", "sum"), pts=("pts", "sum"),
        reb=("reb", "sum"), ast=("ast", "sum"), stl=("stl", "sum"), blk=("blk", "sum"), tov=("tov", "sum"),
        fgm=("fgm", "sum"), fga=("fga", "sum"), fg3m=("fg3m", "sum"), fg3a=("fg3a", "sum"),
        ftm=("ftm", "sum"), fta=("fta", "sum"), pm=("pm", "sum"),
        gmsc=("gmsc", "mean"), gmsc_sd=("gmsc", "std"),
        usg_num=("usg", lambda s: (s * pg.loc[s.index, "min"]).sum()),
    )
    agg["mpg"] = agg["min"] / agg["gp"]
    for c in ("pts", "reb", "ast", "stl", "blk", "tov", "pm"):
        agg[f"{c}_pg"] = agg[c] / agg["gp"]
    agg["ts"] = agg.pts / (2 * (agg.fga + 0.44 * agg.fta)).replace(0, np.nan)
    agg["fg3_pct"] = agg.fg3m / agg.fg3a.replace(0, np.nan)
    agg["usg"] = agg["usg_num"] / agg["min"].replace(0, np.nan)
    agg["pts36"] = 36 * agg.pts / agg["min"].replace(0, np.nan)
    return agg.drop(columns="usg_num")


# ---------------------------------------------------------------------------
# a season: league table, Pythagorean record, SRS
# ---------------------------------------------------------------------------
def counts_in_standings(tg: pd.DataFrame) -> pd.Series:
    return tg["season_type"] == REGULAR


def team_season_table(tg: pd.DataFrame, pythag_exp: float = 14.0) -> pd.DataFrame:
    """All 30 teams, regular season to date, with league ranks."""
    reg = tg[counts_in_standings(tg)]
    if not len(reg):
        return pd.DataFrame()
    s = ["pts", "opp_pts", "poss", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
         "tov", "o_fgm", "o_fga", "o_fg3m", "o_fg3a", "o_ftm", "o_fta", "o_oreb", "o_dreb", "o_tov", "minutes"]
    t = reg.groupby("team_abbr").agg(gp=("game_id", "count"), w=("win", "sum"), **{c: (c, "sum") for c in s})
    t["l"] = t.gp - t.w
    t["win_pct"] = t.w / t.gp
    t["ortg"] = 100 * t.pts / t.poss
    t["drtg"] = 100 * t.opp_pts / t.poss
    t["net"] = t.ortg - t.drtg
    t["pace"] = 48 * t.poss / t.minutes
    t["mov"] = (t.pts - t.opp_pts) / t.gp
    t["efg"] = (t.fgm + 0.5 * t.fg3m) / t.fga
    t["tov_pct"] = t.tov / (t.fga + 0.44 * t.fta + t.tov)
    t["orb_pct"] = t.oreb / (t.oreb + t.o_dreb)
    t["ftr"] = t.ftm / t.fga
    t["o_efg"] = (t.o_fgm + 0.5 * t.o_fg3m) / t.o_fga
    t["o_tov_pct"] = t.o_tov / (t.o_fga + 0.44 * t.o_fta + t.o_tov)
    t["drb_pct"] = t.dreb / (t.dreb + t.o_oreb)
    t["o_ftr"] = t.o_ftm / t.o_fga
    t["fg3a_rate"] = t.fg3a / t.fga
    t["fg3_pct"] = t.fg3m / t.fg3a
    t["o_fg3_pct"] = t.o_fg3m / t.o_fg3a
    # Pythagorean expectation: points scored/allowed predict wins better than wins do
    pf, pa = t.pts ** pythag_exp, t.opp_pts ** pythag_exp
    t["pyth_pct"] = pf / (pf + pa)
    t["pyth_w"] = t.pyth_pct * t.gp
    t["luck"] = t.w - t.pyth_w
    srs_ratings, hca = srs(reg)
    t["srs"] = srs_ratings.reindex(t.index)
    t.attrs["hca"] = hca
    # ranks: 1 = best
    for col, best_high in [("ortg", True), ("drtg", False), ("net", True), ("pace", True), ("srs", True),
                           ("efg", True), ("tov_pct", False), ("orb_pct", True), ("ftr", True),
                           ("o_efg", False), ("o_tov_pct", True), ("drb_pct", True), ("o_ftr", False),
                           ("fg3a_rate", True), ("win_pct", True)]:
        t[f"{col}_rank"] = t[col].rank(ascending=not best_high, method="min").astype(int)
    return t


def srs(reg_games: pd.DataFrame, neutral_ids: set | None = None) -> tuple[pd.Series, float]:
    """Simple Rating System: every game says  margin = rating_home - rating_away + HCA.

    Stack all games into A @ r = margins and solve by least squares. The ratings are only
    defined up to a constant, and numpy's least-squares solution picks the one with
    ratings summing to zero, so SRS reads as 'points per game better than an average team,
    adjusted for who you played'. (This is the normal-equations problem from linear algebra.)
    """
    home = reg_games[reg_games["home_away"] == "home"]
    if not len(home):
        return pd.Series(dtype=float), 0.0
    teams = sorted(set(reg_games["team_abbr"]) | set(reg_games["opp_abbr"]))
    idx = {t: i for i, t in enumerate(teams)}
    A = np.zeros((len(home), len(teams) + 1))
    for row, (h, a, gid) in enumerate(zip(home["team_abbr"], home["opp_abbr"], home["game_id"])):
        A[row, idx[h]] = 1
        A[row, idx[a]] = -1
        A[row, -1] = 0 if (neutral_ids and gid in neutral_ids) else 1
    b = (home["pts"] - home["opp_pts"]).values.astype(float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pd.Series(x[:-1], index=teams), float(x[-1])


def rolling_team(tg_team: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """A team's games in order, with record-to-date and rolling (possession-weighted) ratings."""
    g = tg_team.sort_values(["game_date", "game_id"]).reset_index(drop=True).copy()
    g["game_no"] = np.arange(1, len(g) + 1)
    g["w_to_date"] = g["win"].cumsum()
    g["l_to_date"] = g["game_no"] - g["w_to_date"]
    r_pts = g["pts"].rolling(window, min_periods=1).sum()
    r_opp = g["opp_pts"].rolling(window, min_periods=1).sum()
    r_poss = g["poss"].rolling(window, min_periods=1).sum()
    g["roll_ortg"] = 100 * r_pts / r_poss
    g["roll_drtg"] = 100 * r_opp / r_poss
    g["roll_net"] = g["roll_ortg"] - g["roll_drtg"]
    g["roll_n"] = np.minimum(g["game_no"], window)
    return g


def streak(results: list[bool]) -> str:
    """[True, True, False, False] -> 'L2'"""
    if not results:
        return ""
    last, n = results[-1], 0
    for r in reversed(results):
        if r != last:
            break
        n += 1
    return f"{'W' if last else 'L'}{n}"


def season_type_label(st: int) -> str:
    return {REGULAR: "Regular season", PLAYOFFS: "Playoffs", PLAY_IN: "Play-in"}.get(st, "Preseason")
