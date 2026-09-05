"""Board-order apply: league VORP queue without touching O-line teams."""

from __future__ import annotations

from src.draft_assistant.checklist_apply_board_order import apply_board_order, league_points


def _proj(
    pid: str,
    name: str,
    position: str,
    *,
    rush_yd: float = 0,
    rush_td: float = 0,
    rec_yd: float = 0,
    rec_td: float = 0,
    pass_yd: float = 0,
    pass_td: float = 0,
    ints: float = 0,
) -> dict:
    return {
        "player_gsis_id": pid,
        "full_name": name,
        "position": position,
        "proj_rushing_yards": rush_yd,
        "proj_rushing_tds": rush_td,
        "proj_receiving_yards": rec_yd,
        "proj_receiving_tds": rec_td,
        "proj_passing_yards": pass_yd,
        "proj_passing_tds": pass_td,
        "proj_interceptions": ints,
        "proj_receptions": 0,
    }


def test_league_points_match_screenshot_bijan_shape():
    # Rough Bijan-shaped components → ~248.8 league pts in the live board.
    row = _proj(
        "bijan",
        "Bijan Robinson",
        "RB",
        rush_yd=1212.77,
        rush_td=6.45,
        rec_yd=695.62,
        rec_td=3.21,
    )
    assert round(league_points(row), 1) == 248.8


def test_apply_board_order_sets_vorp_ranks_and_preserves_teams():
    checklist = {
        "meta": {
            "rank_source": "adp",
            "market_as_of": {"scoring": "half-ppr", "teams": 12},
            "ol_included": False,
        },
        "teams": [{"abbr": "ATL", "ol_unit_rank": 7}],
        "players": [
            {
                "player_id": "puka",
                "name": "Puka Nacua",
                "position": "WR",
                "adp": 2.8,
                "rank_tier": "adp",
                "pos_market_rank": 1,
                "checks": {"offense_top16": True},
            },
            {
                "player_id": "bijan",
                "name": "Bijan Robinson",
                "position": "RB",
                "adp": 2.3,
                "rank_tier": "adp",
                "pos_market_rank": 2,
                "checks": {"offense_top16": True},
            },
            {
                "player_id": "chase",
                "name": "Ja'Marr Chase",
                "position": "WR",
                "adp": 1.5,
                "rank_tier": "adp",
                "pos_market_rank": 2,
                "checks": {"offense_top16": False},
            },
        ],
    }
    # Pad replacements so VORP baselines exist; Bijan/Puka beat Chase on VORP.
    projections = [
        _proj("bijan", "Bijan Robinson", "RB", rush_yd=1200, rush_td=6, rec_yd=700, rec_td=3),
        _proj("puka", "Puka Nacua", "WR", rec_yd=1500, rec_td=9),
        _proj("chase", "Ja'Marr Chase", "WR", rec_yd=1200, rec_td=7),
    ]
    for i in range(1, 43):
        projections.append(_proj(f"wr{i}", f"WR {i}", "WR", rec_yd=100 - i))
    for i in range(1, 31):
        projections.append(_proj(f"rb{i}", f"RB {i}", "RB", rush_yd=100 - i))
    for i in range(1, 13):
        projections.append(_proj(f"qb{i}", f"QB {i}", "QB", pass_yd=3000 - i * 10, pass_td=20))
        projections.append(_proj(f"te{i}", f"TE {i}", "TE", rec_yd=500 - i * 10))

    out = apply_board_order(checklist, projections, board_as_of="2026-09-04")
    assert out["meta"]["rank_source"] == "league_vorp"
    assert out["meta"]["market_as_of"]["scoring"] == "half-ppr"
    assert out["teams"] == [{"abbr": "ATL", "ol_unit_rank": 7}]

    by_id = {p["player_id"]: p for p in out["players"]}
    assert by_id["bijan"]["overall_rank"] == 1
    assert by_id["puka"]["overall_rank"] == 2
    assert by_id["chase"]["overall_rank"] > by_id["puka"]["overall_rank"]
    assert "vorp" not in by_id["bijan"]
    assert by_id["bijan"]["checks"]["offense_top16"] is True
