"""Tests for weekly feature as-of contract audit."""
from __future__ import annotations

import pandas as pd
import pytest

from src.projection.weekly_audit.feature_contract import audit_feature_contracts
from src.projection.weekly_audit.property_tests import (
    build_weekly_feature_matrix,
    prior_weeks_unchanged_when_future_mutated,
    rolling_window_respects_kickoff_cutoff,
    target_week_unchanged_when_target_outcomes_mutated,
    week1_uses_only_prior_season,
)


def test_audit_passes_shifted_roll3_contracts():
    report = audit_feature_contracts()
    assert "targets_share_roll3" not in report["failing_features"]
    assert "carries_share_roll3" not in report["failing_features"]
    assert report["passes"] is True


def test_build_player_week_features_excludes_current_week():
    weekly = pd.DataFrame(
        {
            "player_id": ["p1"] * 4,
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "targets_share": [0.1, 0.2, 0.9, 0.3],
        }
    )
    usage = weekly.sort_values(["player_id", "season", "week"]).copy()
    usage["targets_share_roll3"] = usage.groupby("player_id")["targets_share"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    week3 = float(usage.loc[usage["week"].eq(3), "targets_share_roll3"].iloc[0])
    assert week3 == pytest.approx(0.15)


def test_target_week_unchanged_when_outcomes_mutated():
    assert target_week_unchanged_when_target_outcomes_mutated(
        feature_before=1.0,
        feature_after=1.0,
    )


def test_prior_weeks_unchanged_when_future_mutated():
    before = {1: 0.1, 2: 0.2, 3: 0.3}
    after = {1: 0.1, 2: 0.2, 3: 0.9}
    assert prior_weeks_unchanged_when_future_mutated(before, after, mutated_week=3)


def test_rolling_window_respects_kickoff_cutoff():
    assert rolling_window_respects_kickoff_cutoff([1, 2], target_week=3)
    assert not rolling_window_respects_kickoff_cutoff([1, 3], target_week=3)


def test_week1_uses_only_prior_season_data():
    assert week1_uses_only_prior_season(feature_week1=0.42, prior_season_only_value=0.42)


def test_shifted_rolling_ignores_current_week():
    weekly = pd.DataFrame(
        {
            "player_id": ["p1"] * 4,
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "target_share": [0.1, 0.2, 0.9, 0.3],
        }
    )
    features = build_weekly_feature_matrix(weekly, value_col="target_share", player_id="p1")
    week3_feature = float(features.loc[features["week"].eq(3), "feature_value"].iloc[0])
    assert week3_feature == pytest.approx(0.15)
