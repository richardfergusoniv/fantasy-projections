"""Unit tests for draft checklist prepare + API service."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.draft_assistant.checklist_prepare import (
    assign_rank_tiers,
    criteria_for_meta,
    db_has_tables,
    rank_descending,
    volume_flags_for_players,
)


def test_volume_leader_within_2026_group_from_history():
    players = [
        {
            "player_id": "wr1",
            "display_name": "A",
            "position": "WR",
            "team": "SEA",
            "history": [{"season": 2025, "targets": 120, "fantasy_pts_season": 200}],
        },
        {
            "player_id": "wr2",
            "display_name": "B",
            "position": "WR",
            "team": "SEA",
            "history": [{"season": 2025, "targets": 80, "fantasy_pts_season": 150}],
        },
        {
            "player_id": "rb1",
            "display_name": "C",
            "position": "RB",
            "team": "SEA",
            "history": [
                {
                    "season": 2025,
                    "targets": 40,
                    "rushing_yards": 900,
                    "fantasy_pts_season": 180,
                }
            ],
        },
        {
            "player_id": "te1",
            "display_name": "D",
            "position": "TE",
            "team": "SEA",
            "history": [{"season": 2025, "targets": 70, "fantasy_pts_season": 120}],
        },
        {
            "player_id": "te2",
            "display_name": "E",
            "position": "TE",
            "team": "SEA",
            "history": [{"season": 2025, "targets": 55, "fantasy_pts_season": 90}],
        },
        {
            "player_id": "te3",
            "display_name": "F",
            "position": "TE",
            "team": "SEA",
            "history": [{"season": 2025, "targets": 10, "fantasy_pts_season": 30}],
        },
        {
            "player_id": "qb1",
            "display_name": "G",
            "position": "QB",
            "team": "SEA",
            "history": [
                {
                    "season": 2025,
                    "attempts": 500,
                    "rushing_yards": 200,
                    "fantasy_pts_season": 300,
                }
            ],
        },
    ]
    flags = volume_flags_for_players(players, top_n=16)
    assert flags["wr1"]["target_leader_in_group"] is True
    assert flags["wr2"]["target_leader_in_group"] is False
    assert flags["rb1"]["target_leader_in_group"] is False
    assert flags["rb1"]["rush_vol_leader_in_group"] is True
    assert flags["te1"]["te_top2_targets_in_group"] is True
    assert flags["te2"]["te_top2_targets_in_group"] is True
    assert flags["te3"]["te_top2_targets_in_group"] is False
    assert flags["qb1"]["pass_att_top16"] is True
    assert flags["wr1"]["qb_top16"] is True


def test_rank_tiers_adp_ecr_prior_and_unranked_break():
    players = [
        {"player_id": "1", "display_name": "ADP Guy", "position": "WR", "team": "X", "history": []},
        {
            "player_id": "2",
            "display_name": "ECR Guy",
            "position": "WR",
            "team": "X",
            "history": [{"season": 2025, "fantasy_pts_season": 50}],
        },
        {
            "player_id": "3",
            "display_name": "Prior Guy",
            "position": "WR",
            "team": "X",
            "history": [{"season": 2025, "fantasy_pts_season": 180}],
        },
        {
            "player_id": "4",
            "display_name": "None Guy",
            "position": "WR",
            "team": "X",
            "history": [],
        },
    ]
    comparison = {
        "1": {"adp": 12.0, "ecr": 10.0},
        "2": {"ecr": 40.0},
    }
    rows = assign_rank_tiers(players, comparison)
    by_id = {row["player_id"]: row for row in rows}
    assert by_id["1"]["rank_tier"] == "adp"
    assert by_id["2"]["rank_tier"] == "ecr"
    assert by_id["3"]["rank_tier"] == "prior_pts"
    assert by_id["4"]["rank_tier"] == "none"
    assert by_id["1"]["pos_market_rank"] == 1
    assert by_id["2"]["pos_market_rank"] == 2
    assert by_id["3"]["pos_market_rank"] == 3
    assert by_id["3"]["unranked_break"] is True
    assert by_id["4"]["unranked_break"] is False


def test_sos_criteria_omitted_when_gate_fails():
    criteria = criteria_for_meta(ol_included=False, sos_included=False)
    assert "sos_top16" not in criteria["WR"]
    assert "ol_top16" not in criteria["RB"]
    assert "offense_top16" in criteria["QB"]


def test_sealed_ol_unit_ranks_load_and_apply(tmp_path: Path):
    from src.draft_assistant.checklist_prepare import (
        apply_ol_unit_ranks,
        load_sealed_ol_unit_ranks,
    )

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
            "criteria_labels": {"offense_top16": "TOP 16 OFFENSE"},
        },
        "criteria_by_position": {
            "QB": ["pass_att_top16", "offense_top16", "sos_top16"],
            "RB": ["target_leader_in_group", "offense_top16", "sos_top16"],
            "WR": ["target_leader_in_group", "offense_top16", "sos_top16"],
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
                "checks": {"offense_top16": True},
            },
            {
                "player_id": "rb-det",
                "name": "Jahmyr Gibbs",
                "position": "RB",
                "team": "DET",
                "checks": {"offense_top16": True},
            },
            {
                "player_id": "qb-kc",
                "name": "Patrick Mahomes",
                "position": "QB",
                "team": "KC",
                "checks": {"offense_top16": True},
            },
            {
                "player_id": "wr-den",
                "name": "Courtland Sutton",
                "position": "WR",
                "team": "DEN",
                "checks": {"offense_top16": True},
            },
        ],
    }
    apply_ol_unit_ranks(payload, ranks, ol_source="sealed:test.json")
    assert payload["meta"]["ol_included"] is True
    assert payload["meta"]["ol_source"] == "sealed:test.json"
    assert "ol_top16" in payload["meta"]["criteria_labels"]
    assert payload["criteria_by_position"]["QB"] == [
        "pass_att_top16",
        "offense_top16",
        "ol_top16",
        "sos_top16",
    ]
    assert "ol_top16" not in payload["criteria_by_position"]["WR"]
    by_abbr = {t["abbr"]: t for t in payload["teams"]}
    assert by_abbr["DEN"]["ol_unit_rank"] == 1
    assert by_abbr["DET"]["ol_unit_rank"] == 17
    by_id = {p["player_id"]: p for p in payload["players"]}
    assert by_id["qb-den"]["checks"]["ol_top16"] is True
    assert by_id["rb-det"]["checks"]["ol_top16"] is False
    assert by_id["qb-kc"]["checks"]["ol_top16"] is False
    assert "ol_top16" not in by_id["wr-den"]["checks"]


def test_committed_ol_unit_ranks_cover_all_32_teams():
    path = Path("draft_assistant/data/ol_unit_ranks_2026.json")
    if not path.is_file():
        pytest.skip("sealed OL ranks not present")
    from src.draft_assistant.checklist_prepare import load_sealed_ol_unit_ranks
    from src.team_stats.prepare import TEAM_META

    ranks = load_sealed_ol_unit_ranks(path)
    expected = {meta["abbr"] for meta in TEAM_META}
    assert set(ranks) == expected
    assert sorted(ranks.values()) == list(range(1, 33))
    # Screenshot top-16 board used for checklist alignment.
    top16 = {abbr for abbr, rank in ranks.items() if rank <= 16}
    assert top16 == {
        "DEN",
        "PHI",
        "CHI",
        "BUF",
        "LA",
        "SF",
        "TB",
        "SEA",
        "BAL",
        "LAC",
        "MIN",
        "ATL",
        "DAL",
        "NE",
        "NYJ",
        "ARI",
    }


def test_offense_rank_uses_team_totals_not_player_sum():
    # rank_descending is what offense ranks use; higher yards => rank 1
    ranks = rank_descending({"DAL": 7000.0, "SEA": 6000.0, "CIN": 5000.0})
    assert ranks["DAL"] == 1
    assert ranks["SEA"] == 2
    assert ranks["CIN"] == 3


def test_committed_checklist_json_loads():
    path = Path("draft_assistant/data/draft_checklist_2026.json")
    if not path.is_file():
        pytest.skip("checklist artifact not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["meta"]["market_as_of"]["scoring"] == "half-ppr"
    assert payload["meta"]["market_as_of"]["teams"] == 12
    assert "vorp" not in (payload["players"][0] or {})
    assert payload["players"][0]["rank_tier"] in {"adp", "ecr", "prior_pts", "none"}


def test_committed_checklist_ol_matches_sealed_screenshot_board():
    checklist_path = Path("draft_assistant/data/draft_checklist_2026.json")
    ranks_path = Path("draft_assistant/data/ol_unit_ranks_2026.json")
    if not checklist_path.is_file() or not ranks_path.is_file():
        pytest.skip("checklist or sealed OL ranks not present")
    from src.draft_assistant.checklist_prepare import load_sealed_ol_unit_ranks

    payload = json.loads(checklist_path.read_text(encoding="utf-8"))
    ranks = load_sealed_ol_unit_ranks(ranks_path)
    assert payload["meta"]["ol_included"] is True
    assert payload["meta"]["ol_source"].startswith("sealed:")
    assert "ol_top16" in payload["criteria_by_position"]["QB"]
    assert "ol_top16" in payload["criteria_by_position"]["RB"]
    assert "ol_top16" not in payload["criteria_by_position"]["WR"]
    assert "ol_top16" not in payload["criteria_by_position"]["TE"]

    by_abbr = {t["abbr"]: t for t in payload["teams"]}
    for abbr, rank in ranks.items():
        assert by_abbr[abbr]["ol_unit_rank"] == rank

    for player in payload["players"]:
        pos = player["position"]
        if pos not in ("QB", "RB"):
            assert "ol_top16" not in (player.get("checks") or {})
            continue
        team = player.get("team")
        expected = bool(team) and (ranks.get(str(team)) or 99) <= 16
        assert player["checks"]["ol_top16"] is expected


def _sqlite_with(path: Path, *tables: str) -> Path:
    conn = sqlite3.connect(path)
    try:
        for table in tables:
            conn.execute(f"create table {table} (a int)")
        conn.commit()
    finally:
        conn.close()
    return path


def test_db_guard_rejects_empty_or_partial_projections_db(tmp_path: Path):
    """A present-but-unusable projections.db must fall through to nflverse.

    projections.db ships as a zero-byte placeholder in some deploy targets and
    sqlite3.connect opens it happily, so file existence alone is not a usable
    signal.
    """
    required = ("pbp", "schedules")

    missing = tmp_path / "absent.db"
    assert db_has_tables(missing, required) is False

    empty = tmp_path / "zero.db"
    empty.touch()
    assert db_has_tables(empty, required) is False

    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a sqlite database")
    assert db_has_tables(junk, required) is False

    partial = _sqlite_with(tmp_path / "partial.db", "pbp")
    assert db_has_tables(partial, required) is False


def test_db_guard_accepts_a_populated_projections_db(tmp_path: Path):
    full = _sqlite_with(tmp_path / "full.db", "pbp", "schedules")
    assert db_has_tables(full, ("pbp", "schedules")) is True


def test_checklist_is_not_written_into_the_sealed_release():
    """The frozen namespace stays byte-stable; the checklist lives beside it.

    Anything added under releases/<namespace>/ needs a manifest entry with a
    digest, or verify_browser_surfaces cannot integrity-check it.
    """
    sealed = Path("draft_assistant/data/releases/v2_baseline_20260830")
    if not sealed.is_dir():
        pytest.skip("release bundle not present")
    assert not (sealed / "draft_checklist_2026.json").exists()

    manifest_path = sealed / "release_bundle_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("release manifest not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {entry.get("role") for entry in manifest.get("artifacts") or []}
    assert "draft_checklist" not in roles
