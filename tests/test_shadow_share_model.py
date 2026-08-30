"""Tests for shadow 0A compositional target shares."""
from __future__ import annotations

import pandas as pd

from src.projection.shadow.share_model import (
    OTHER_TARGET_BUCKET,
    allocate_target_shares,
    validate_share_simplex,
)


def test_shares_sum_to_one_with_explicit_other_bucket():
    room = pd.DataFrame(
        {
            "player_id": ["w1", "w2", "t1"],
            "position": ["WR", "WR", "TE"],
            "team": ["AAA", "AAA", "AAA"],
            "target_share": [0.2, 0.15, 0.1],
        }
    )
    allocated = allocate_target_shares(room, other_share=0.05)
    player_sum = allocated["target_share_pred"].sum()
    other = float(allocated[OTHER_TARGET_BUCKET].iloc[0])
    assert abs(player_sum + other - 1.0) < 1e-6


def test_validate_share_simplex_passes():
    allocated = allocate_target_shares(
        pd.DataFrame(
            {
                "player_id": ["w1", "r1"],
                "team": ["AAA", "AAA"],
                "target_share": [0.3, 0.2],
            }
        ),
        other_share=0.1,
    )
    results = validate_share_simplex(allocated)
    assert all(results.values())
