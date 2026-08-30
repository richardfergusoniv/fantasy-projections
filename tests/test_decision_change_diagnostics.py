"""Tests for decision-change diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.evaluation.decision_change_diagnostics import (
    build_decision_change_event_rows,
    build_player_stability_diagnostic_rows,
    classify_decision_event,
    load_decision_diagnostic_config,
    verify_draw_id_uniqueness,
    verify_nested_prefix_provenance,
)
from src.projection.evaluation.draw_stability import (
    ACTIVE_DECISION_METRIC_KEYS,
    count_decision_changes,
    enumerate_decision_changes,
    qualifies,
    resolve_decision_threshold_registry,
    signed_distance_from_threshold,
)


def test_qualifies_ge_and_le():
    assert qualifies(0.51, 0.5, "ge") is True
    assert qualifies(0.49, 0.5, "ge") is False
    assert qualifies(11.0, 12.0, "le") is True
    assert qualifies(13.0, 12.0, "le") is False


def test_signed_distance_positive_when_qualifies():
    assert signed_distance_from_threshold(0.55, 0.5, "ge") == pytest.approx(0.05)
    assert signed_distance_from_threshold(0.45, 0.5, "ge") == pytest.approx(-0.05)
    assert signed_distance_from_threshold(11.0, 12.0, "le") == pytest.approx(1.0)
    assert signed_distance_from_threshold(13.0, 12.0, "le") == pytest.approx(-1.0)


def test_enumerate_decision_changes_returns_player_rows():
    candidate = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "p_finish_top12": [0.49, 0.60],
            "p_finish_top24": [0.8, 0.8],
            "p_vorp_positive": [0.4, 0.4],
        }
    )
    reference = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "p_finish_top12": [0.51, 0.40],
            "p_finish_top24": [0.8, 0.8],
            "p_vorp_positive": [0.4, 0.4],
        }
    )
    events = enumerate_decision_changes(candidate, reference)
    assert len(events) == 2
    assert {event["player_id"] for event in events} == {"a", "b"}
    assert all(event["metric"] == "p_finish_top12" for event in events)


def test_count_decision_changes_matches_enumeration():
    candidate = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "p_finish_top12": [0.49, 0.60],
            "p_finish_top24": [0.8, 0.8],
            "p_vorp_positive": [0.4, 0.4],
        }
    )
    reference = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "p_finish_top12": [0.51, 0.40],
            "p_finish_top24": [0.8, 0.8],
            "p_vorp_positive": [0.4, 0.4],
        }
    )
    counts = count_decision_changes(candidate, reference)
    events = enumerate_decision_changes(candidate, reference)
    assert counts["total"] == len(events)
    assert counts["by_metric"]["p_finish_top12"] == 2


def test_active_decision_metrics_exclude_rank():
    registry = resolve_decision_threshold_registry()
    assert "expected_pos_rank_top12" not in registry
    assert set(registry) == set(ACTIVE_DECISION_METRIC_KEYS)


def test_classify_reference_instability_on_crossing_disagree():
    config = load_decision_diagnostic_config()
    row = {
        "metric_kind": "probability",
        "reference_10k_vs_20k_crossing_disagrees": True,
        "reference_10k_to_20k_abs_diff": 0.001,
        "distance_from_threshold_10k": 0.1,
        "change_survives_at_20k": True,
    }
    assert classify_decision_event(row, config=config) == "reference_instability"


def test_classify_boundary_noise_when_not_surviving():
    config = load_decision_diagnostic_config()
    row = {
        "metric_kind": "probability",
        "reference_10k_vs_20k_crossing_disagrees": False,
        "reference_10k_to_20k_abs_diff": 0.001,
        "distance_from_threshold_10k": 0.01,
        "change_survives_at_20k": False,
    }
    assert classify_decision_event(row, config=config) == "boundary_noise"


def test_classify_material_when_survives_at_20k():
    config = load_decision_diagnostic_config()
    row = {
        "metric_kind": "probability",
        "reference_10k_vs_20k_crossing_disagrees": False,
        "reference_10k_to_20k_abs_diff": 0.001,
        "distance_from_threshold_10k": 0.1,
        "change_survives_at_20k": True,
    }
    assert classify_decision_event(row, config=config) == "material"


def test_change_survives_at_20k_uses_qualification_not_distance():
    config = load_decision_diagnostic_config()
    threshold_registry = resolve_decision_threshold_registry()
    player_meta = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "Player One",
                "position": "WR",
                "adp": 5.0,
                "selected_fantasy_point_forecast": 200.0,
                "is_core_adp_player": True,
            }
        ]
    )
    candidate_metrics = pd.DataFrame(
        {
            "player_id": ["p1"],
            "p_finish_top12": [0.49],
            "p_finish_top24": [0.8],
            "p_vorp_positive": [0.4],
            "p10": [180.0],
            "p50": [200.0],
            "p90": [220.0],
            "sim_vorp_p50": [10.0],
            "p_finish_top6": [0.1],
            "p_finish_top48": [0.9],
        }
    )
    reference_10k = pd.DataFrame(
        {
            "player_id": ["p1"],
            "p_finish_top12": [0.51],
            "p_finish_top24": [0.8],
            "p_vorp_positive": [0.4],
            "p10": [180.0],
            "p50": [200.0],
            "p90": [220.0],
            "sim_vorp_p50": [10.0],
            "p_finish_top6": [0.1],
            "p_finish_top48": [0.9],
        }
    )
    reference_20k = pd.DataFrame(
        {
            "player_id": ["p1"],
            "p_finish_top12": [0.52],
            "p_finish_top24": [0.8],
            "p_vorp_positive": [0.4],
            "p10": [180.0],
            "p50": [200.0],
            "p90": [220.0],
            "sim_vorp_p50": [10.0],
            "p_finish_top6": [0.1],
            "p_finish_top48": [0.9],
        }
    )
    rows = build_decision_change_event_rows(
        season=2026,
        selected_board_hash="abc",
        simulation_seed=123,
        candidate_draw_count=1000,
        primary_reference_draws=10000,
        diagnostic_reference_draws=20000,
        candidate_metrics=candidate_metrics,
        reference_10k_metrics=reference_10k,
        reference_20k_metrics=reference_20k,
        player_meta=player_meta,
        threshold_registry=threshold_registry,
        config=config,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["change_survives_at_20k"] is True
    assert row["reference_10k_vs_20k_crossing_disagrees"] is False
    assert row["review_required"] is True


def test_player_tail_instability_flag_independent_of_events():
    config = load_decision_diagnostic_config()
    player_meta = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "position": "WR",
                "adp": 50.0,
                "selected_fantasy_point_forecast": 200.0,
                "is_core_adp_player": False,
            }
        ]
    )
    candidate_metrics = pd.DataFrame(
        {
            "player_id": ["p1"],
            "p10": [170.0],
            "p50": [200.0],
            "p90": [240.0],
            "p_finish_top6": [0.1],
            "p_finish_top12": [0.51],
            "p_finish_top24": [0.8],
            "p_finish_top36": [0.9],
            "p_finish_top48": [0.95],
            "p_vorp_positive": [0.6],
            "sim_vorp_p50": [10.0],
            "expected_pos_rank": [8.0],
        }
    )
    reference_10k = pd.DataFrame(
        {
            "player_id": ["p1"],
            "p10": [180.0],
            "p50": [200.1],
            "p90": [220.0],
            "p_finish_top6": [0.1],
            "p_finish_top12": [0.51],
            "p_finish_top24": [0.8],
            "p_finish_top36": [0.9],
            "p_finish_top48": [0.95],
            "p_vorp_positive": [0.6],
            "sim_vorp_p50": [10.0],
            "expected_pos_rank": [8.0],
        }
    )
    rows = build_player_stability_diagnostic_rows(
        candidate_draw_count=5000,
        candidate_metrics=candidate_metrics,
        reference_10k_metrics=reference_10k,
        player_meta=player_meta,
        config=config,
    )
    assert rows[0]["tail_instability_flag"] is True


def _make_draws(n_draws: int, player_id: str = "p1", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(200, 20, size=n_draws)
    return pd.DataFrame(
        {
            "player_id": [player_id] * n_draws,
            "position": ["WR"] * n_draws,
            "team": ["AAA"] * n_draws,
            "draw": list(range(n_draws)),
            "fantasy_pts_season": values,
        }
    )


def test_verify_draw_id_uniqueness_passes_for_complete_prefix():
    draws = _make_draws(100)
    result = verify_draw_id_uniqueness(draws, max_draw_id=100)
    assert result["passes"] is True
    assert result["rows_per_draw"] == 1


def test_verify_draw_id_uniqueness_allows_multiple_players_per_draw():
    draws = pd.concat([_make_draws(10, player_id="p1"), _make_draws(10, player_id="p2")])
    result = verify_draw_id_uniqueness(draws, max_draw_id=10)
    assert result["passes"] is True
    assert result["rows_per_draw"] == 2


def test_verify_nested_prefix_provenance_fails_on_hash_mismatch():
    primary = _make_draws(100, seed=42)
    diagnostic = _make_draws(100, seed=99)
    result = verify_nested_prefix_provenance(
        season=2026,
        primary_reference_draws=100,
        diagnostic_reference_draws=200,
        primary_raw=primary,
        diagnostic_raw=diagnostic,
        contract_hashes={"selected_board_hash": "abc"},
        simulation_seed=123,
        checkpoint_meta={"seed": 123},
        selected_board_model_id="accuracy_first_ensemble",
        canonical_projection_run_id="run1",
    )
    assert result["passes"] is False
    assert "prefix_hash_mismatch" in result["failures"]
