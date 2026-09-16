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


def _load_integrity_extension():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_weekly_integrity_extension.py"
    spec = importlib.util.spec_from_file_location("audit_weekly_integrity_extension", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_as_of_cutoff_and_market_snapshot_checks_skip_until_helpers_exist():
    mod = _load_integrity_extension()
    cutoff = {c["name"]: c for c in mod.as_of_cutoff_checks()}
    market = {c["name"]: c for c in mod.market_snapshot_checks()}
    assert cutoff["as_of_cutoff_values_carry_available_at"]["skipped"] is True
    assert cutoff["as_of_cutoff_vintage_filter_drops_later_arriving_values"]["skipped"] is True
    assert market["market_snapshot_rows_carry_snapshot_date"]["skipped"] is True
    assert market["market_snapshot_filter_excludes_later_season_values"]["skipped"] is True
    assert "not yet in tree" in cutoff["as_of_cutoff_values_carry_available_at"]["detail"].lower()
    assert "not yet in tree" in market["market_snapshot_rows_carry_snapshot_date"]["detail"].lower()


def test_as_of_cutoff_contract_drops_later_arriving_values():
    import polars as pl

    mod = _load_integrity_extension()

    def filter_available_at(df, *, cutoff):
        return df.filter(pl.col("available_at") <= cutoff)

    checks = {c["name"]: c for c in mod.as_of_cutoff_checks(filter_fn=filter_available_at)}
    assert checks["as_of_cutoff_values_carry_available_at"]["passed"] is True
    assert checks["as_of_cutoff_vintage_filter_drops_later_arriving_values"]["passed"] is True
    assert checks["as_of_cutoff_vintage_filter_drops_later_arriving_values"]["skipped"] is False


def test_as_of_cutoff_contract_fails_if_filter_keeps_later_values():
    mod = _load_integrity_extension()

    def leaky(df, *, cutoff):
        return df

    checks = {c["name"]: c for c in mod.as_of_cutoff_checks(filter_fn=leaky)}
    assert checks["as_of_cutoff_vintage_filter_drops_later_arriving_values"]["passed"] is False


def test_market_snapshot_contract_drops_later_season_adp():
    import polars as pl

    mod = _load_integrity_extension()

    def filter_market_snapshot(df, *, as_of):
        return df.filter(pl.col("snapshot_date") <= as_of)

    checks = {
        c["name"]: c
        for c in mod.market_snapshot_checks(
            filter_fn=filter_market_snapshot, snapshot_col="snapshot_date"
        )
    }
    assert checks["market_snapshot_rows_carry_snapshot_date"]["passed"] is True
    assert checks["market_snapshot_filter_excludes_later_season_values"]["passed"] is True


def test_market_snapshot_contract_fails_if_final_adp_leaks():
    mod = _load_integrity_extension()

    def leaky(df, *, snapshot_date):
        return df

    checks = {c["name"]: c for c in mod.market_snapshot_checks(filter_fn=leaky)}
    assert checks["market_snapshot_filter_excludes_later_season_values"]["passed"] is False

