"""Tests for Shadow 0A evaluation."""
from __future__ import annotations

import pandas as pd

from src.projection.shadow.evaluate_0a import evaluate_shadow_0a_on_long_board


def test_evaluate_shadow_0a_on_synthetic_board():
    rows = []
    for pid, share in [("w1", 0.2), ("w2", 0.15), ("r1", 0.05)]:
        rows.append(
            {
                "player_id": pid,
                "position": "WR" if pid.startswith("w") else "RB",
                "team": "AAA",
                "stat": "targets",
                "pred_pg": share * 5,
                "projected_games": 17,
                "target_share": share,
                "team_pass_attempts_pg_pred": 35.0,
                "adp": 10,
            }
        )
    board = pd.DataFrame(rows)
    metrics = evaluate_shadow_0a_on_long_board(board)
    assert metrics["n_shadow_players"] >= 2
    assert metrics["reconciliation_burden"] >= 0.0
