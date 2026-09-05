"""Unit tests for draft checklist prepare + API service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.draft_assistant.checklist_prepare import (
    apply_ol_unit_ranks,
    criteria_for_meta,
    load_sealed_ol_unit_ranks,
    rank_descending,
)


def test_rank_descending_higher_is_better():
    ranks = rank_descending({"DAL": 7000.0, "SEA": 6000.0, "CIN": 5000.0})
    assert ranks["DAL"] == 1
    assert ranks["SEA"] == 2
    assert ranks["CIN"] == 3


def test_sos_and_ol_criteria_omitted_when_gate_fails():
    criteria = criteria_for_meta(ol_included=False, sos_included=False)
    assert "sos_rank" not in criteria["WR"]
    assert "ol_rank" not in criteria["RB"]
    assert "offense_pts_rank" in criteria["QB"]
    assert "offense_yds_rank" in criteria["QB"]


def test_sealed_ol_unit_ranks_load_and_apply(tmp_path: Path):
    ranks_path = tmp_path / "ol_unit_ranks_2026.json"
    ranks_path.write_text(
        json.dumps(
            {
                "season": 2026,
                "unit_ranks": {"DEN": 1, "PHI": 2, "DET": 17, "KC": 23},
            }
        ),
        encoding="utf-8",
    )
    ranks = load_sealed_ol_unit_ranks(ranks_path)
    assert ranks["DEN"] == 1
    assert ranks["DET"] == 17

    payload = {
        "meta": {
            "top_n": 16,
            "ol_included": False,
            "ol_source": "missing",
            "criteria_labels": {"offense_pts_rank": "OFFENSE PTS RANK"},
        },
        "criteria_by_position": {
            "QB": ["pass_att_rank", "offense_pts_rank", "sos_rank"],
            "RB": ["tgt_rank", "offense_pts_rank", "sos_rank"],
            "WR": ["tgt_rank", "offense_pts_rank", "sos_rank"],
        },
        "teams": [
            {"abbr": "DEN", "name": "Denver Broncos", "offense_rank": 12},
            {"abbr": "DET", "name": "Detroit Lions", "offense_rank": 5},
            {"abbr": "KC", "name": "Kansas City Chiefs", "offense_rank": 8},
        ],
        "players": [
            {
                "player_id": "qb-den",
                "name": "Bo Nix",
                "position": "QB",
                "team": "DEN",
                "ranks": {"offense_pts_rank": 12},
                "checks": {},
            },
            {
                "player_id": "rb-det",
                "name": "Jahmyr Gibbs",
                "position": "RB",
                "team": "DET",
                "ranks": {"offense_pts_rank": 5},
                "checks": {},
            },
            {
                "player_id": "qb-kc",
                "name": "Patrick Mahomes",
                "position": "QB",
                "team": "KC",
                "ranks": {"offense_pts_rank": 8},
                "checks": {},
            },
            {
                "player_id": "wr-den",
                "name": "Courtland Sutton",
                "position": "WR",
                "team": "DEN",
                "ranks": {"offense_pts_rank": 12},
                "checks": {},
            },
        ],
    }
    apply_ol_unit_ranks(payload, ranks, ol_source="sealed:test.json")
    assert payload["meta"]["ol_included"] is True
    assert payload["meta"]["ol_source"] == "sealed:test.json"
    assert "ol_rank" in payload["meta"]["criteria_labels"]
    assert "ol_rank" in payload["criteria_by_position"]["QB"]
    assert "ol_rank" in payload["criteria_by_position"]["RB"]
    assert "ol_rank" not in payload["criteria_by_position"]["WR"]
    by_abbr = {t["abbr"]: t for t in payload["teams"]}
    assert by_abbr["DEN"]["ol_unit_rank"] == 1
    assert by_abbr["DET"]["ol_unit_rank"] == 17
    by_id = {p["player_id"]: p for p in payload["players"]}
    assert by_id["qb-den"]["ranks"]["ol_rank"] == 1
    assert by_id["rb-det"]["ranks"]["ol_rank"] == 17
    assert by_id["qb-kc"]["ranks"]["ol_rank"] == 23
    assert "ol_rank" not in by_id["wr-den"]["ranks"]


def test_committed_ol_unit_ranks_cover_all_32_teams():
    path = Path("draft_assistant/data/ol_unit_ranks_2026.json")
    if not path.is_file():
        pytest.skip("sealed OL ranks not present")
    from src.team_stats.prepare import TEAM_META

    ranks = load_sealed_ol_unit_ranks(path)
    expected = {meta["abbr"] for meta in TEAM_META}
    assert set(ranks) == expected
    assert sorted(ranks.values()) == list(range(1, 33))


def test_committed_checklist_json_loads():
    path = Path("draft_assistant/data/draft_checklist_2026.json")
    if not path.is_file():
        pytest.skip("checklist artifact not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["meta"]["rank_source"] == "market_avg"
    assert payload["meta"]["market_as_of"]["scoring"] == "ppr"
    assert payload["meta"]["market_as_of"]["teams"] == 12
    assert "vorp" not in (payload["players"][0] or {})
    assert payload["players"][0]["rank_tier"] == "market_avg"
    assert payload["meta"]["player_count"] >= 700
    assert len(payload["players"]) == payload["meta"]["player_count"]
    assert "ranks" in payload["players"][0]
    assert "pass_att_rank" in payload["criteria_by_position"]["QB"]
    assert "tgt_rank" in payload["criteria_by_position"]["WR"]


def test_committed_checklist_has_vegas_ranks_for_stars():
    path = Path("draft_assistant/data/draft_checklist_2026.json")
    if not path.is_file():
        pytest.skip("checklist artifact not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chase = next(p for p in payload["players"] if p["name"] == "Ja'Marr Chase")
    assert chase["position"] == "WR"
    assert chase["pos_market_rank"] == 1
    assert chase["ranks"]["tgt_rank"] is not None
    assert chase["ranks"]["qb_rank"] is not None
    assert chase["ranks"]["offense_pts_rank"] is not None
    assert chase["ranks"]["offense_yds_rank"] is not None
    assert chase["ranks"]["sos_rank"] is not None

    allen = next(p for p in payload["players"] if p["name"] == "Josh Allen")
    assert allen["ranks"]["pass_att_rank"] is not None
    assert allen["ranks"]["ol_rank"] is not None


def test_market_average_skips_missing_sources():
    from src.draft_assistant.market_adp import average_market_value

    assert average_market_value(
        {"adp_espn": 10.0, "adp_ffc": None, "adp_mfl": 14.0, "ecr": 12.0}
    ) == pytest.approx(12.0)
    assert average_market_value(
        {"adp_espn": None, "adp_ffc": None, "adp_mfl": None, "ecr": None}
    ) is None


def test_committed_checklist_ol_ranks_match_sealed_board():
    checklist_path = Path("draft_assistant/data/draft_checklist_2026.json")
    ranks_path = Path("draft_assistant/data/ol_unit_ranks_2026.json")
    if not checklist_path.is_file() or not ranks_path.is_file():
        pytest.skip("checklist or sealed OL ranks not present")

    payload = json.loads(checklist_path.read_text(encoding="utf-8"))
    ranks = load_sealed_ol_unit_ranks(ranks_path)
    assert payload["meta"]["ol_included"] is True
    assert str(payload["meta"]["ol_source"]).startswith("sealed:")
    assert "ol_rank" in payload["criteria_by_position"]["QB"]
    assert "ol_rank" in payload["criteria_by_position"]["RB"]
    assert "ol_rank" not in payload["criteria_by_position"]["WR"]
    assert "ol_rank" not in payload["criteria_by_position"]["TE"]

    by_abbr = {t["abbr"]: t for t in payload["teams"]}
    for abbr, rank in ranks.items():
        assert by_abbr[abbr]["ol_unit_rank"] == rank

    for player in payload["players"]:
        pos = player["position"]
        if pos not in ("QB", "RB"):
            assert "ol_rank" not in (player.get("ranks") or {})
            continue
        team = player.get("team")
        assert player["ranks"]["ol_rank"] == ranks.get(str(team))


