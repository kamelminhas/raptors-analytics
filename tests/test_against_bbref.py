"""Checks the pipeline against Basketball-Reference's published numbers for 2025-26.

Run with:  python -m pip install pytest  then  python -m pytest -q
(downloads last season's data the first time)
"""
import pytest

from raptors import data as D
from raptors import metrics as M


@pytest.fixture(scope="module")
def season():
    s = D.load_season(2026)
    return s, M.team_games(s.team_box, s.player_box, s.pbp)


def test_game7_four_factors(season):
    """Game 7, TOR at CLE, 2026-05-03 (basketball-reference.com/boxscores/202605030CLE.html)."""
    _, tg = season
    g = tg[tg["game_id"] == 401869384].set_index("team_abbr")
    expected = {  # team: (eFG%, TOV%, ORB%, ORtg)
        "TOR": (48.3, 12.5, 14.9, 102.8),
        "CLE": (51.2, 14.4, 43.5, 114.8),
    }
    for team, (efg, tov, orb, ortg) in expected.items():
        r = g.loc[team]
        assert round(100 * r["efg"], 1) == efg
        assert round(100 * r["tov_pct"], 1) == tov
        assert round(100 * r["orb_pct"], 1) == orb
        assert round(r["ortg"], 1) == ortg


def test_raptors_season_summary(season):
    """basketball-reference.com/teams/TOR/2026.html"""
    _, tg = season
    t = M.team_season_table(tg).loc["TOR"]
    assert (int(t["w"]), int(t["l"])) == (46, 36)
    assert round(t["pyth_w"]) == 48
    assert abs(t["srs"] - 2.75) < 0.05
    assert abs(t["pace"] - 98.4) < 0.1
    assert abs(t["ortg"] - 115.9) < 0.1
    assert abs(t["drtg"] - 113.0) < 0.1
    assert int(t["ortg_rank"]) == 15 and int(t["drtg_rank"]) == 5 and int(t["srs_rank"]) == 11


def test_shots_match_box_score(season):
    """Field-goal attempts rebuilt from the play-by-play should match the official box score."""
    from raptors import pbp as P
    s, _ = season
    sh = P.shots(s.pbp)
    sh = sh[sh["zone"] != "Heave"]
    reg = s.team_box[s.team_box["season_type"] == D.REGULAR].set_index(["game_id", "team_id"])
    fga = sh.groupby(["game_id", "team_id"]).size()
    joined = reg[["fga"]].join(fga.rename("pbp_fga")).dropna()
    assert (joined["pbp_fga"] == joined["fga"]).mean() > 0.95
