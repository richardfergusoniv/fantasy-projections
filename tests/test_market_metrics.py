"""Unit tests for draft-edge metrics and ensemble helpers."""

from __future__ import annotations

import pandas as pd

from src.projection.market_metrics import (
    apply_blend,
    draft_edge_proxy,
    fit_nonnegative_blend_weights,
    matched_market_frame,
    norm_name,
)


def test_norm_name():
    assert norm_name("Ja'Marr Chase") == "jamarr chase"
    assert norm_name("Kenneth Walker III") == "kenneth walker"


def test_matched_market_frame_reranks():
    board = pd.DataFrame(
        [
            {"player_id": "a", "display_name": "A", "position": "WR", "model_points": 200, "actual_points": 180},
            {"player_id": "b", "display_name": "B", "position": "RB", "model_points": 150, "actual_points": 160},
            {"player_id": "c", "display_name": "C", "position": "QB", "model_points": 100, "actual_points": 90},
        ]
    )
    consensus = pd.DataFrame(
        [
            {"player_id": "a", "display_name": "A", "position": "WR", "adp": 5.0},
            {"player_id": "b", "display_name": "B", "position": "RB", "adp": 1.0},
            {"player_id": "c", "display_name": "C", "position": "QB", "adp": 10.0},
        ]
    )
    matched = matched_market_frame(board, consensus, max_market_rank=None)
    assert len(matched) == 3
    assert set(matched["mkt"]) == {1, 2, 3}
    assert set(matched["our"]) == {1, 2, 3}
    # B has best ADP -> mkt 1; A has most model points -> our 1
    b = matched[matched["player_id"] == "b"].iloc[0]
    assert int(b["mkt"]) == 1
    a = matched[matched["player_id"] == "a"].iloc[0]
    assert int(a["our"]) == 1


def test_draft_edge_proxy_detects_signal():
    # Model ranks A above market; A actually scores more
    matched = pd.DataFrame(
        {
            "our": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "mkt": [5, 6, 1, 2, 3, 4, 7, 8, 9, 10],
            "d": [-4, -4, 2, 2, 2, 2, 0, 0, 0, 0],
            "actual_points": [200, 190, 80, 70, 60, 50, 40, 30, 20, 10],
            "actual_rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    edge = draft_edge_proxy(matched)
    assert edge["has_actuals"]
    assert edge["n"] == 10
    assert edge["edge_corr_residual_vs_actual_points"] < 0


def test_blend_weights_nonnegative():
    rows = []
    for i in range(20):
        rows.append(
            {
                "position": "WR",
                "actual_points": 100 + i,
                "v1_pred": 90 + i * 0.5,
                "v2_pred": 100 + i,
            }
        )
        rows.append(
            {
                "position": "RB",
                "actual_points": 80 + i,
                "v1_pred": 80 + i,
                "v2_pred": 60 + i * 0.2,
            }
        )
    frame = pd.DataFrame(rows)
    weights = fit_nonnegative_blend_weights(frame)
    assert abs(weights["WR"]["v1_pred"] + weights["WR"]["v2_pred"] - 1.0) < 1e-9
    assert weights["WR"]["v2_pred"] >= weights["WR"]["v1_pred"]  # v2 closer
    blended = apply_blend(frame, weights)
    assert "blend_pred" in blended.columns
