"""Shared quote primitives must hold at both grains, not just the season one.

`src/projection/market_quotes` was extracted from the season consensus, so it
inherited that module's bugs and handed them to the weekly props pipeline.
These pin the fixes at the shared layer, where both grains read them.
"""

from __future__ import annotations

from src.projection.market_quotes import (
    SEASON_QUOTE_POLICY,
    conflicts_with_projection,
    is_count_market,
    one_sided_longshot,
)
from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY

WEEKLY = DEFAULT_WEEKLY_POLICY.quote


def test_even_money_one_sided_quote_is_not_a_longshot():
    # +100 is even money — a main number, not an alt rung. Books that publish
    # only an over price would otherwise lose their main line at both grains.
    assert one_sided_longshot(100, None) is False
    assert one_sided_longshot(None, 100) is False
    assert one_sided_longshot(100, None, threshold=WEEKLY.one_sided_longshot_threshold) is False
    assert one_sided_longshot(125, None) is True
    assert one_sided_longshot(None, 400, threshold=WEEKLY.one_sided_longshot_threshold) is True


def test_conflict_gate_never_fires_downward_at_either_grain():
    # A book under a model projection is disagreement, not a juice ladder.
    # Dropping those biases consensus upward.
    assert conflicts_with_projection(400.0, 900.0) is False
    assert conflicts_with_projection(20.0, 60.0, policy=WEEKLY, market="rec_yards") is False
    assert conflicts_with_projection(1.0, 6.0, policy=WEEKLY, market="rec_tds") is False
    # Upward juice is still caught.
    assert conflicts_with_projection(1499.5, 364.0) is True


def test_count_branch_follows_the_market_not_the_magnitude():
    # Season: an elite quarterback at 27 projected passing TDs is still a count
    # market. Under the magnitude rule it got the yardage gate, where a 34.5
    # rung clears `yard_upward_abs` (80) trivially and was kept.
    assert conflicts_with_projection(34.5, 27.0, market="pass_tds") is True
    assert conflicts_with_projection(34.5, 27.0) is False  # magnitude fallback

    # Weekly: the same hole in reverse. `count_scale_ceiling` is 10.0, so an
    # elite receiver projected for 12 catches fell into the yardage branch.
    # There a 15-catch line clears neither `yard_conflict_rel_hard` (0.45) nor
    # `yard_upward_abs` (25), so it survived; the count branch rejects it.
    assert conflicts_with_projection(15.0, 12.0, policy=WEEKLY, market="receptions") is True
    assert conflicts_with_projection(15.0, 12.0, policy=WEEKLY) is False  # magnitude fallback

    # And a low-magnitude yardage line must not get count thresholds: a weekly
    # 9-yard projection against an 11-yard line is noise, not a juice ladder.
    assert conflicts_with_projection(11.0, 9.0, policy=WEEKLY, market="rush_yards") is False
    assert conflicts_with_projection(11.0, 9.0, policy=WEEKLY) is True  # magnitude fallback


def test_count_market_classification_is_by_canonical_name():
    assert is_count_market("rec_tds") is True
    assert is_count_market("receptions") is True
    assert is_count_market("rec_yards") is False
    assert is_count_market(None) is None
    assert is_count_market("") is None


def test_season_policy_still_carries_the_calibrated_thresholds():
    # The season grain's Pierce calibration must survive the extraction.
    assert SEASON_QUOTE_POLICY.count_conflict_rel == 0.15
    assert SEASON_QUOTE_POLICY.yard_upward_abs == 80.0
    # Pierce: Caesars-only 7.5 rec TDs vs a RotoWire 6.0 projection.
    assert conflicts_with_projection(7.5, 6.0, market="rec_tds") is True
    # FanDuel 925.5 vs 872 is a fair number, not juice.
    assert conflicts_with_projection(925.5, 872.0, market="rec_yards") is False
