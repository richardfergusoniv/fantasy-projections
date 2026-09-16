"""Same-week outcomes must never enter Role 2 prediction frames."""
from __future__ import annotations

import pandas as pd
import pytest

from src.projection.weekly.draws.feature_outcome_split import SAME_WEEK_OUTCOME_DENYLIST
from src.projection.weekly_eval.errors import (
    OutcomeFeatureLeakageError,
    Role3BlendForbiddenError,
)
from src.projection.weekly_eval.leakage import (
    assert_no_role3_blend,
    assert_prediction_frame_has_no_outcomes,
)


def test_smuggled_same_week_targets_fail_closed():
    frame = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": 2026,
                "week": 1,
                "market": "rec_yards",
                "model_mean": 80.0,
                "targets": 11,
            }
        ]
    )
    with pytest.raises(OutcomeFeatureLeakageError, match="targets"):
        assert_prediction_frame_has_no_outcomes(frame.columns)


def test_smuggled_team_attempts_fail_closed():
    with pytest.raises(OutcomeFeatureLeakageError, match="team_attempts"):
        assert_prediction_frame_has_no_outcomes(
            ["player_id", "season", "week", "model_mean", "team_attempts"]
        )


def test_lagged_roll_columns_are_allowed():
    assert_prediction_frame_has_no_outcomes(
        ["player_id", "season", "week", "model_mean", "targets_l3", "team_attempts_l3"]
    )


def test_clean_shadow_board_columns_are_allowed():
    assert_prediction_frame_has_no_outcomes(
        [
            "player_id",
            "player_name",
            "season",
            "week",
            "market",
            "model_mean",
            "model_std",
            "model_p_over",
            "position",
            "team",
        ]
    )


def test_denylist_covers_known_same_week_aggregates():
    for name in ("team_attempts", "team_air_yards", "fantasy_points", "attempts"):
        assert name in SAME_WEEK_OUTCOME_DENYLIST
        with pytest.raises(OutcomeFeatureLeakageError):
            assert_prediction_frame_has_no_outcomes(["player_id", name])


def test_smuggled_actual_label_column_fails_closed():
    with pytest.raises(OutcomeFeatureLeakageError, match="actual"):
        assert_prediction_frame_has_no_outcomes(
            ["player_id", "season", "week", "market", "model_mean", "actual"]
        )


def test_role3_blend_is_forbidden():
    with pytest.raises(Role3BlendForbiddenError, match="Role 3"):
        assert_no_role3_blend({"model": 0.7, "vegas": 0.3})
    assert_no_role3_blend(None)
    assert_no_role3_blend({})
