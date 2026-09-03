"""Unit tests for draft checklist prepare + API service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.draft_assistant.checklist_prepare import (
    assign_rank_tiers,
    criteria_for_meta,
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
