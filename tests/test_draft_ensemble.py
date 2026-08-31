"""Tests for draft-assistant v1/v2 ensemble post-process."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.draft_assistant.prepare import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    apply_ensemble_points,
    load_ensemble_weights,
    resolve_ensemble_weights_path,
)


def test_shipped_ensemble_weights_load():
    weights = load_ensemble_weights(DEFAULT_ENSEMBLE_WEIGHTS)
    assert weights is not None
    assert abs(weights["QB"]["v1_pred"] + weights["QB"]["v2_pred"] - 1.0) < 1e-9
    assert weights["RB"]["v1_pred"] == 1.0
    assert weights["WR"]["v2_pred"] == 0.0


def test_apply_ensemble_points_blends_and_reports(tmp_path: Path):
    v1 = pd.DataFrame(
        [
            {
                "player_id": "a",
                "position": "QB",
                "fantasy_pts_season": 100.0,
                "fantasy_pts": 10.0,
                "projected_games": 10.0,
            },
            {
                "player_id": "b",
                "position": "RB",
                "fantasy_pts_season": 200.0,
                "fantasy_pts": 20.0,
                "projected_games": 10.0,
            },
        ]
    )
    v2_path = tmp_path / "v2.csv"
    pd.DataFrame(
        [
            {"player_id": "a", "position": "QB", "fantasy_pts_season": 200.0},
            {"player_id": "b", "position": "RB", "fantasy_pts_season": 50.0},
        ]
    ).to_csv(v2_path, index=False)
    weights = {
        "QB": {"v1_pred": 0.4, "v2_pred": 0.6},
        "RB": {"v1_pred": 1.0, "v2_pred": 0.0},
    }
    out, applied = apply_ensemble_points(v1, weights, v2_points_path=str(v2_path))
    assert applied
    qb = out[out["player_id"] == "a"].iloc[0]
    rb = out[out["player_id"] == "b"].iloc[0]
    assert abs(float(qb["fantasy_pts_season"]) - (0.4 * 100 + 0.6 * 200)) < 1e-6
    assert abs(float(rb["fantasy_pts_season"]) - 200.0) < 1e-6


def test_apply_ensemble_points_missing_v2_is_noop(tmp_path: Path):
    v1 = pd.DataFrame(
        [
            {
                "player_id": "a",
                "position": "QB",
                "fantasy_pts_season": 100.0,
                "fantasy_pts": 10.0,
                "projected_games": 10.0,
            }
        ]
    )
    out, applied = apply_ensemble_points(
        v1, {"QB": {"v1_pred": 0.5, "v2_pred": 0.5}}, v2_points_path=str(tmp_path / "missing.csv")
    )
    assert not applied
    assert float(out.iloc[0]["fantasy_pts_season"]) == 100.0


def test_resolve_ensemble_defaults_on(tmp_path: Path):
    v2 = tmp_path / "fantasy_points_2099.csv"
    v2.write_text("player_id,position,fantasy_pts_season\na,QB,1\n", encoding="utf-8")
    path = resolve_ensemble_weights_path(
        season=2099,
        ensemble_weights_path=None,
        use_ensemble=True,
        ensemble_v2_points_path=str(v2),
    )
    assert path == DEFAULT_ENSEMBLE_WEIGHTS
    assert (
        resolve_ensemble_weights_path(
            season=2099,
            ensemble_weights_path=None,
            use_ensemble=False,
            ensemble_v2_points_path=str(v2),
        )
        is None
    )
