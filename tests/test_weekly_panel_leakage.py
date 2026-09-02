"""Poisoned-future leakage tests on the real weekly-v2 player-week panel."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

PANEL_PATH = Path("data/processed/player_week_panel.parquet")


@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel not built")
def test_poisoned_future_fantasy_points_do_not_change_prior_rolling_features():
    """Mutating week-10 outcomes must not change week-9 lagged rolling features."""
    from src.projection.weekly.features.rolling import add_rolling_means

    panel = pl.read_parquet(PANEL_PATH)
    candidates = (
        panel.filter((pl.col("season") == 2024) & (pl.col("week") == 10))
        .select("gsis_id")
        .unique()
        .head(1)
    )
    if candidates.is_empty():
        pytest.skip("no week-10 rows in 2024 panel")
    player = candidates.item(0, 0)
    sub = panel.filter((pl.col("gsis_id") == player) & (pl.col("season") == 2024)).sort("week")
    assert sub.filter(pl.col("week") == 10).height == 1

    clean = add_rolling_means(sub, ["fantasy_points"], windows=(3,))
    for poison_week in range(4, 15):
        poisoned = sub.with_columns(
            pl.when(pl.col("week") >= poison_week)
            .then(pl.lit(999.0))
            .otherwise(pl.col("fantasy_points"))
            .alias("fantasy_points")
        )
        dirty = add_rolling_means(poisoned, ["fantasy_points"], windows=(3,))
        prior_week = poison_week
        effect_week = poison_week + 1
        if (
            clean.filter(pl.col("week") == prior_week).is_empty()
            or clean.filter(pl.col("week") == effect_week).is_empty()
        ):
            continue
        clean_prior = clean.filter(pl.col("week") == prior_week)["fantasy_points_l3"].item()
        dirty_prior = dirty.filter(pl.col("week") == prior_week)["fantasy_points_l3"].item()
        clean_effect = clean.filter(pl.col("week") == effect_week)["fantasy_points_l3"].item()
        dirty_effect = dirty.filter(pl.col("week") == effect_week)["fantasy_points_l3"].item()
        if clean_prior == pytest.approx(dirty_prior) and clean_effect != dirty_effect:
            return
    pytest.fail("no poison week demonstrated lagged rolling isolation")


@pytest.mark.skipif(not PANEL_PATH.exists(), reason="panel not built")
def test_filter_as_of_excludes_target_season_rows():
    from src.projection.weekly.features.leakage import filter_as_of

    panel = pl.read_parquet(PANEL_PATH)
    as_of = filter_as_of(panel, season=2024, week=5)
    assert as_of.filter(pl.col("season") > 2024).is_empty()
    assert as_of.filter((pl.col("season") == 2024) & (pl.col("week") >= 5)).is_empty()
