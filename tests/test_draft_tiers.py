"""Tests for draft tier assignment and JSON export."""

import json

import pandas as pd
import pytest

from src.draft_assistant.prepare import build_player_records, export_draft_data
from src.draft_assistant.tiers import assign_tiers, add_tier_columns


def test_assign_tiers_splits_on_large_gap():
    pts = pd.Series([20.0, 19.5, 18.0, 17.9], index=[0, 1, 2, 3])
    tiers = assign_tiers(pts, gap=1.0, pct_gap=None)
    assert tiers.tolist() == [1, 1, 2, 2]


def test_assign_tiers_splits_on_pct_gap():
    pts = pd.Series([10.0, 9.5, 9.0], index=[0, 1, 2])
    tiers = assign_tiers(pts, gap=99.0, pct_gap=0.04)
    assert tiers.tolist() == [1, 2, 3]


def test_add_tier_columns_per_position():
    df = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "position": ["RB", "RB", "WR", "WR"],
            "fantasy_pts": [20.0, 19.0, 15.0, 14.5],
        }
    )
    out = add_tier_columns(df)
    assert "overall_tier" in out.columns
    assert "pos_tier" in out.columns
    rb = out[out.position == "RB"].sort_values("pos_rank")
    assert rb.iloc[0]["pos_tier"] == 1


def test_add_tier_columns_flex_pool_excludes_qb():
    df = pd.DataFrame(
        {
            "player_id": ["qb", "rb1", "rb2", "wr1"],
            "position": ["QB", "RB", "RB", "WR"],
            "fantasy_pts": [25.0, 18.0, 17.0, 16.0],
        }
    )
    out = add_tier_columns(df)
    qb = out[out.player_id == "qb"].iloc[0]
    rb1 = out[out.player_id == "rb1"].iloc[0]
    wr1 = out[out.player_id == "wr1"].iloc[0]

    assert pd.isna(qb.flex_rank)
    assert rb1.flex_rank == 1
    assert wr1.flex_rank == 3
    assert rb1.flex_tier == 1


def test_build_player_records_replaces_nan_with_null():
    df = pd.DataFrame(
        {
            "player_id": ["x"],
            "display_name": ["Test Player"],
            "position": ["WR"],
            "team": ["TST"],
            "fantasy_pts": [10.0],
            "fantasy_pts_low": [float("nan")],
            "fantasy_pts_high": [12.0],
            "fantasy_pts_season": [150.0],
            "projected_games": [15.0],
            "source": ["test"],
            "low_confidence": [float("nan")],
            "role": [float("nan")],
            "depth_chart_status": ["starter"],
            "overall_rank": [1],
            "overall_tier": [1],
            "pos_rank": [1],
            "pos_tier": [1],
        }
    )
    rec = build_player_records(df)[0]
    assert rec["role"] is None
    assert rec["fantasy_pts_low"] is None
    assert rec["low_confidence"] is False
    json.dumps(rec)


def test_export_draft_data_writes_strict_json(tmp_path, monkeypatch):
    csv_dir = tmp_path / "output"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "player_id": ["a"],
            "display_name": ["Alpha"],
            "position": ["QB"],
            "team": ["TST"],
            "fantasy_pts": [20.0],
            "fantasy_pts_low": [float("nan")],
            "fantasy_pts_high": [25.0],
            "fantasy_pts_season": [300.0],
            "projected_games": [15.0],
            "source": ["test"],
            "low_confidence": [False],
            "role": [float("nan")],
            "depth_chart_status": ["starter"],
        }
    ).to_csv(csv_dir / "fantasy_points_2099.csv", index=False)

    draft_dir = tmp_path / "draft_assistant" / "data"
    monkeypatch.setattr("src.draft_assistant.prepare.OUTPUT_DIR", str(csv_dir))
    monkeypatch.setattr("src.draft_assistant.prepare.DRAFT_DATA_DIR", str(draft_dir))

    out_path = export_draft_data(2099)
    raw = open(out_path, encoding="utf-8").read()
    assert "NaN" not in raw
    json.loads(raw)
