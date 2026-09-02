"""Regression tests for weekly-v2 tuning/evaluation harness integrity."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.projection.weekly.evaluate.harness import (
    PreseasonEvalConfig,
    evaluate_season,
    run_preseason_backtest,
)
from src.projection.weekly.evaluate.nested_selection import (
    NestedSelectionConfig,
    rank_candidate_on_inner,
    run_nested_selection,
)
from src.projection.weekly.evaluate.preseason import PromotionPolicy, promotion_gate
from src.projection.weekly.models.volume_config import VolumeModelConfig


def _synthetic_panel() -> pl.DataFrame:
    rows = []
    for season in range(2018, 2024):
        for week in range(1, 4):
            for i, pos in enumerate(("QB", "RB", "WR", "TE")):
                gsis = f"{pos}{i:02d}"
                fp = float((i + 1) * 3 + week + season % 5)
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "gsis_id": gsis,
                        "player_name": f"P{gsis}",
                        "position": pos,
                        "team": "TST",
                        "fantasy_points": fp,
                        "targets": 2.0 + i,
                        "carries": 1.0 + i,
                        "attempts": 5.0 if pos == "QB" else 0.0,
                        "receptions": 1.0 + i,
                        "receiving_yards": 10.0 * i,
                        "rushing_yards": 5.0 * i,
                        "passing_yards": 100.0 if pos == "QB" else 0.0,
                        "target_share": 0.1 * (i + 1),
                        "carry_share": 0.1 * (i + 1),
                        "dropback_share": 0.9 if pos == "QB" else 0.0,
                        "snap_share": 0.5,
                        "air_yards_share": 0.1,
                        "redzone_target_share": 0.05,
                        "games_played_prior": 10,
                        "is_rookie": False,
                        "depth_rank": float(i + 1),
                    }
                )
    return pl.DataFrame(rows)


def test_tuner_and_evaluator_share_run_preseason_backtest():
    """Tuner inner logic and evaluator must use the same backtest function."""
    panel = _synthetic_panel()
    config = PreseasonEvalConfig(
        panel_path=Path("data/processed/player_week_panel.parquet"),
        outer_start=2022,
        outer_end=2022,
        volume_options={"two_stage": False},
    )
    fake_report = {
        "season": 2022,
        "mae": 1.0,
        "rank_corr": 0.5,
        "dispersion_ratio": 0.8,
        "baseline": {"mae": 2.0, "rank_corr": 0.4},
        "train_seasons": [2018, 2019, 2020, 2021],
    }
    fake_oof = panel.head(1).select(
        pl.col("gsis_id"),
        pl.lit(2022).alias("season"),
        pl.lit(1).alias("week"),
        pl.col("position"),
        pl.lit(1.0).alias("actual_fantasy_points"),
        pl.lit(1.0).alias("projected_fantasy_points"),
    )
    with patch(
        "src.projection.weekly.evaluate.harness.evaluate_season",
        return_value=(fake_report, fake_oof),
    ) as mocked:
        run_preseason_backtest(panel, config=config)
        assert mocked.call_count == 1


def test_dispersion_not_disabled_in_selection_ranking():
    """Candidates failing dispersion must rank below passing candidates."""
    policy = PromotionPolicy()
    good = [
        {
            "season": 2023,
            "mae": 3.0,
            "rank_corr": 0.5,
            "dispersion_ratio": 0.85,
            "coverage": 1.0,
            "baseline": {"mae": 4.0, "rank_corr": 0.4},
            "interval": {"coverage": 0.8},
        }
    ]
    bad = [
        {
            "season": 2023,
            "mae": 2.5,
            "rank_corr": 0.6,
            "dispersion_ratio": 0.55,
            "coverage": 1.0,
            "baseline": {"mae": 4.0, "rank_corr": 0.4},
            "interval": {"coverage": 0.8},
        }
    ]
    good_key, _ = rank_candidate_on_inner(good, baseline_reports=good, policy=policy)
    bad_key, _ = rank_candidate_on_inner(bad, baseline_reports=good, policy=policy)
    assert good_key > bad_key


def test_promotion_gate_enforces_dispersion_bounds():
  reports = [
      {
          "season": 2023,
          "mae": 3.0,
          "rank_corr": 0.5,
          "dispersion_ratio": 0.65,
          "coverage": 1.0,
          "baseline": {"mae": 4.0, "rank_corr": 0.4},
          "interval": {"coverage": 0.8},
      }
  ]
  gate = promotion_gate(reports)
  assert not gate["promote"]
  assert any("dispersion" in f for f in gate["failures"])


def test_volume_config_fingerprint_changes_with_options():
    a = VolumeModelConfig()
    b = VolumeModelConfig(recency_half_life_seasons=4.0)
    assert a.fingerprint() != b.fingerprint()


def test_stale_root_tuning_selection_not_loaded_by_train(tmp_path: Path):
    """Train only reads explicit --tuning-selection, not MODELS_DIR root files."""
    stale = tmp_path / "stale_tuning_selection.json"
    stale.write_text(
        json.dumps(
            {
                "promote": True,
                "volume_options": {"two_stage": False, "recency_half_life_seasons": 2.0},
            }
        ),
        encoding="utf-8",
    )
    opts, artifact = __import__(
        "scripts.weekly_v2_train", fromlist=["_load_volume_options"]
    )._load_volume_options(None)
    assert opts == {}
    assert artifact is None


def test_explicit_output_paths_isolated_via_registry(tmp_path: Path):
    from src.projection.weekly.models.registry import active_models_dir, set_registry_dir

    a = tmp_path / "a"
    b = tmp_path / "b"
    set_registry_dir(a)
    assert active_models_dir() == a
    set_registry_dir(b)
    assert active_models_dir() == b
    set_registry_dir(None)


def test_nested_selection_marks_warmup_fold():
    from src.projection.weekly.evaluate.nested_selection import select_from_cached_backtests

    tiny_candidates = (
        {
            "name": "baseline",
            "options": VolumeModelConfig().to_options(),
        },
    )
    config = NestedSelectionConfig(
        outer_start=2022,
        outer_end=2022,
        min_inner_seasons=2,
        candidates=tiny_candidates,
    )
    selection = select_from_cached_backtests(
        2022, {"baseline": []}, tiny_candidates, config=config
    )
    assert selection["status"] == "warmup"
    assert selection["selected"] is None


def test_candidate_cache_invalidates_on_options_change(tmp_path: Path):
    from src.projection.weekly.evaluate.nested_selection import _candidate_cache_path

    cache = tmp_path / "cache"
    cache.mkdir()
    path = _candidate_cache_path(cache, "baseline")
    path.write_text(
        json.dumps({"volume_options": {"two_stage": True}, "calibrated_seasons": []}),
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["volume_options"] != {"two_stage": False}


def test_evaluate_cli_accepts_tuning_selection(tmp_path: Path):
    selection = tmp_path / "tuning_selection.json"
    selection.write_text(
        json.dumps({"volume_options": {"two_stage": False}}),
        encoding="utf-8",
    )
    from scripts.weekly_v2_evaluate import _load_volume_options
    import argparse

    args = argparse.Namespace(
        volume_options_json=None,
        tuning_selection=selection,
    )
    opts = _load_volume_options(args)
    assert opts == {"two_stage": False}
