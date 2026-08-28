"""Smoke tests for rankings comparison export."""

from __future__ import annotations

from src.draft_assistant.compare_prepare import (
    _join_keys,
    _norm_name,
    rebase_comparison_payload,
)


def test_norm_name_strips_suffixes():
    assert _norm_name("Ja'Marr Chase") == "jamarr chase"
    assert _norm_name("Kenneth Walker III") == "kenneth walker"


def test_join_keys_team_alias():
    n, p, t = _join_keys("A Player", "WR", "JAC")
    assert n == "a player"
    assert p == "WR"
    assert t == "JAX"


def test_rebase_preserves_market_snapshot_and_updates_our_side():
    comparison = {
        "meta": {
            "season": 2026,
            "generated_at": "old",
            "ecr": {"scrape_date": "2026-08-21"},
            "adp": {"end_date": "2026-08-27"},
        },
        "players": [
            {
                "player_id": "wr1",
                "our_rank": 36,
                "fantasy_pts_season": 154.1,
                "ecr": 8.84,
                "adp": 12.5,
                "ecr_sd": 2.32,
                "adp_stdev": 2.3,
            }
        ],
    }
    board = {
        "meta": {
            "season": 2026,
            "generated_at": "new-board",
            "model_id": "accuracy_first_ensemble",
            "source_file": "accuracy.csv",
        },
        "players": [
            {
                "player_id": "wr1",
                "display_name": "CeeDee Lamb",
                "position": "WR",
                "team": "DAL",
                "overall_rank": 16,
                "pos_rank": 6,
                "fantasy_pts": 11.48,
                "fantasy_pts_season": 195.14,
                "vorp": 81.5,
            }
        ],
    }

    result = rebase_comparison_payload(board, comparison)
    row = result["players"][0]
    assert row["our_rank"] == 16
    assert row["our_pos_rank"] == 6
    assert row["fantasy_pts_season"] == 195.14
    assert row["ecr"] == 8.84
    assert row["adp"] == 12.5
    assert row["ecr_sd"] == 2.32
    assert row["delta_ecr"] == 7.16
    assert row["delta_adp"] == 3.5
    assert result["meta"]["ecr"]["scrape_date"] == "2026-08-21"
    assert result["meta"]["adp"]["end_date"] == "2026-08-27"
    assert result["meta"]["board_model_id"] == "accuracy_first_ensemble"
    assert result["meta"]["market_snapshot_preserved"] is True
