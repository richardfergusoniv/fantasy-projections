"""Regression tests for event cohort + evaluation-integrity repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.projection.weekly.draws.cohort_panel import build_complete_roster_cohort, sha256_frame_full
from src.projection.weekly.draws.contracts_v2 import (
    derive_event_labels,
    event_denominator_mask,
    observed_active_from_roster_status,
)
from src.projection.weekly.draws.event_baselines import fit_training_baselines
from src.projection.weekly.draws.event_models import (
    evaluate_event_predictions,
    fit_event_models,
)
from src.projection.weekly.draws.feature_outcome_split import (
    assert_no_outcome_columns,
    split_prediction_outcome_frames,
)
from src.projection.weekly.draws.prediction_inputs import build_team_game_input_from_predictions
from src.projection.weekly.draws.readiness import default_no_go_report
from src.projection.weekly.draws.game_engine import generate_game_draws
from src.projection.weekly.draws.game_engine import ScheduledGameInput


def _mini_cohort_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gsis_id": ["qb1", "rb1", "wr1", "wr2", "te1", "wr3"],
            "season": [2024] * 6,
            "week": [2] * 6,
            "team": ["AAA"] * 6,
            "position": ["QB", "RB", "WR", "WR", "TE", "WR"],
            "game_id": ["g1"] * 6,
            "opponent": ["BBB"] * 6,
            "has_scheduled_game": [True] * 6,
            "roster_status": ["ACT", "ACT", "INA", "ACT", "ACT", "DEV"],
            "offense_snaps": [65.0, 30.0, 0.0, 45.0, 20.0, None],
            "targets": [0.0, 2.0, 0.0, 8.0, 3.0, None],
            "carries": [3.0, 12.0, 0.0, 0.0, 0.0, None],
            "attempts": [32.0, 0.0, 0.0, 0.0, 0.0, None],
            "fantasy_points": [18.0, 10.0, 0.0, 12.0, 4.0, None],
            "play_prob": [1.0, 0.9, 0.1, 1.0, 1.0, 0.0],
            "is_out": [False, False, True, False, False, False],
            "depth_rank": [1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
        }
    )


def test_active_label_independent_of_play_prob():
    df = _mini_cohort_table()
    labeled = derive_event_labels(df)
    # Poison play_prob — labels must not change.
    poisoned = df.with_columns(pl.lit(0.01).alias("play_prob"))
    labeled2 = derive_event_labels(poisoned)
    assert labeled["active_label"].to_list() == labeled2["active_label"].to_list()
    assert observed_active_from_roster_status("ACT") is True
    assert observed_active_from_roster_status("INA") is False


def test_event_denominators_exact_counts():
    df = derive_event_labels(_mini_cohort_table())
    active_d = df.filter(event_denominator_mask("active_label", df)).height
    part_d = df.filter(event_denominator_mask("participated_label", df)).height
    pos_d = df.filter(event_denominator_mask("positive_usage_label", df)).height
    assert active_d == df.filter(pl.col("active_label").is_not_null()).height
    assert part_d == df.filter(pl.col("active_label") == True).height  # noqa: E712
    assert pos_d == df.filter(pl.col("participated_label") == True).height  # noqa: E712


def test_test_fold_cannot_define_baseline():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.3, 0.7, 0.6, 0.4, 0.8])
    with pytest.raises(ValueError, match="baseline"):
        evaluate_event_predictions(y, p)
    train_rate = 0.2
    metrics = evaluate_event_predictions(y, p, baseline_rate=train_rate)
    assert metrics["brier_baseline"] == pytest.approx(
        float(np.mean((np.full(5, train_rate) - y) ** 2))
    )


def test_balanced_weight_not_default():
    panel = derive_event_labels(_mini_cohort_table())
    bundle = fit_event_models(panel, min_positive=1, positions=("QB", "RB", "WR", "TE"))
    assert bundle.config.get("class_weight") is None


def test_inference_rejects_same_week_actuals():
    row = {
        "gsis_id": "wr1",
        "position": "WR",
        "team": "AAA",
        "p_participates": 0.8,
        "p_positive_usage": 0.7,
        "targets": 9.0,
        "fantasy_points": 12.0,
    }
    with pytest.raises(ValueError, match="forbidden"):
        build_team_game_input_from_predictions([row], team="AAA", opponent="BBB")


def test_fitted_event_probs_change_draw_partition():
    low = {
        "gsis_id": "wr1",
        "position": "WR",
        "team": "AAA",
        "p_active": 0.95,
        "p_participates": 0.2,
        "p_positive_usage": 0.2,
        "pred_target_share": 0.2,
        "pred_carry_share": 0.0,
        "pred_mean_pass_attempts": 34.0,
        "pred_mean_rush_attempts": 26.0,
    }
    high = {**low, "p_participates": 0.95, "p_positive_usage": 0.95}
    t_low = build_team_game_input_from_predictions([low], team="AAA", opponent="BBB")
    t_high = build_team_game_input_from_predictions([high], team="AAA", opponent="BBB")
    away = build_team_game_input_from_predictions(
        [
            {
                "gsis_id": "qb2",
                "position": "QB",
                "team": "BBB",
                "p_active": 1.0,
                "p_participates": 0.99,
                "p_positive_usage": 0.99,
                "pred_target_share": 0.0,
                "pred_carry_share": 0.1,
                "pred_mean_pass_attempts": 32.0,
                "pred_mean_rush_attempts": 24.0,
            }
        ],
        team="BBB",
        opponent="AAA",
    )
    g_low = generate_game_draws(
        ScheduledGameInput("g", 2024, 1, home=t_low, away=away), draw_count=200, seed=1
    )
    g_high = generate_game_draws(
        ScheduledGameInput("g", 2024, 1, home=t_high, away=away), draw_count=200, seed=1
    )
    wr_low = next(p for p in g_low["teams"][0]["players"] if p["player_id"] == "wr1")
    wr_high = next(p for p in g_high["teams"][0]["players"] if p["player_id"] == "wr1")
    z_low = sum(1 for d in wr_low["draws"] if d.get("targets", 0) == 0) / len(wr_low["draws"])
    z_high = sum(1 for d in wr_high["draws"] if d.get("targets", 0) == 0) / len(wr_high["draws"])
    assert z_low > z_high


def test_feature_outcome_split_poison_safe():
    df = derive_event_labels(_mini_cohort_table())
    feats, outcomes, _ = split_prediction_outcome_frames(df)
    assert "fantasy_points" in outcomes.columns
    assert "fantasy_points" not in feats.columns
    assert_no_outcome_columns(feats.columns)


def test_conservation_gate_no_arbitrary_bypass():
    report = default_no_go_report(point_dispersion_passes=False)
    report.per_draw_conservation.passed = True
    report.per_draw_conservation.evidence_hash = "abc"
    report.event_probability_calibration.passed = True
    report.event_probability_calibration.evidence_hash = "abc"
    report.joint_draw_proper_scores.passed = True
    report.joint_draw_proper_scores.evidence_hash = "abc"
    report.artifact_integrity.passed = True
    report.artifact_integrity.evidence_hash = "abc"
    report.recompute_decisions(point_dispersion_passes=False)
    assert report.joint_draw_classification == "GO"
    assert report.auto_publish_allowed is False


def test_training_baseline_independent_of_test_prevalence():
    train = pl.DataFrame(
        {
            "season": [2022] * 100,
            "position": ["WR"] * 100,
            "has_scheduled_game": [True] * 100,
            "active_label": [True] * 80 + [False] * 20,
            "participated_label": [True] * 60 + [False] * 20 + [None] * 20,
            "positive_usage_label": [True] * 40 + [False] * 20 + [None] * 40,
            "depth_rank": [1.0] * 100,
            "play_prob": [0.9] * 100,
            "is_out": [False] * 100,
        }
    )
    test = pl.DataFrame(
        {
            "season": [2023] * 10,
            "position": ["WR"] * 10,
            "has_scheduled_game": [True] * 10,
            "active_label": [True] * 10,
            "participated_label": [True] * 10,
            "positive_usage_label": [True] * 10,
            "depth_rank": [1.0] * 10,
            "play_prob": [0.9] * 10,
            "is_out": [False] * 10,
        }
    )
    bundle = fit_training_baselines(train)
    base_p = bundle.predict("active_label", "WR", test)  # type: ignore[arg-type]
    assert np.all(base_p > 0.5) and np.all(base_p < 0.95)
    assert not np.allclose(base_p, 1.0)


def test_full_frame_content_hash_changes_with_rows():
    a = pl.DataFrame({"x": [1, 2]})
    b = pl.DataFrame({"x": [1, 3]})
    assert sha256_frame_full(a) != sha256_frame_full(b)


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("data/processed/player_week_panel.parquet").is_file(),
    reason="external research panel unavailable in clean clone",
)
def test_roster_row_without_boxscore_survives_cohort():
    panel = pl.read_parquet("data/processed/player_week_panel.parquet")
    cohort = build_complete_roster_cohort(panel, seasons=[2024])
    # Cohort should be broader than panel skill rows with stats.
    panel_skill = panel.filter(
        (pl.col("season") == 2024) & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
    )
    assert cohort.height >= panel_skill.height
