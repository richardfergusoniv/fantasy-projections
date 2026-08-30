"""Tests for nested-prefix draw-count stability evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projection.evaluation.draw_stability import (
    candidate_passes_gate,
    classify_candidate_vs_reference_event,
    compare_metric_distributions,
    count_decision_changes,
    enumerate_decision_changes,
    evaluate_production_decision_events,
    filter_draw_prefix,
    passes_numerical_stability,
    prefix_is_nested_subset,
    qualifies,
    select_smallest_passing_draw_count,
    signed_distance_from_threshold,
    summarize_candidate_metrics,
    verify_checkpoint_draws,
    verify_contract_match,
    _build_player_meta_for_stability,
)


def _make_draws(n_draws: int, player_id: str = "p1") -> pd.DataFrame:
    rng = np.random.default_rng(42)
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


def test_filter_draw_prefix_uses_half_open_interval():
    draws = _make_draws(10)
    prefix = filter_draw_prefix(draws, 3)
    assert sorted(prefix["draw"].unique().tolist()) == [0, 1, 2]


def test_prefix_is_nested_subset_byte_identical():
    full = _make_draws(100)
    assert prefix_is_nested_subset(filter_draw_prefix(full, 25), full, max_draw_id=25)


def test_candidate_summaries_use_only_allowed_prefix():
    draws = _make_draws(200)
    board = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "position": "WR",
                "team": "AAA",
                "fantasy_pts_season": 200.0,
                "projected_games": 17,
            }
        ]
    )
    small = summarize_candidate_metrics(draws, board=board, max_draws=50)
    large = summarize_candidate_metrics(draws, board=board, max_draws=200)
    # Prefix summaries should differ when MC noise matters; both should exist.
    assert not small.empty
    assert not large.empty
    assert "p50" in small.columns


def test_compare_metric_distributions_median_p95_gate():
    candidate = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "p50": [100.0, 101.0, 102.0],
            "p_finish_top12": [0.49, 0.50, 0.51],
        }
    )
    reference = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "p50": [100.1, 100.9, 102.2],
            "p_finish_top12": [0.50, 0.50, 0.52],
        }
    )
    result = compare_metric_distributions(
        candidate,
        reference,
        metrics=("p50", "p_finish_top12"),
        tolerances={"p50_abs": 0.25, "probability_abs": 0.015},
    )
    assert result["per_metric"]["p50"]["median_passes"] is True
    assert result["per_metric"]["p50"]["p95_passes"] is True


def test_count_decision_changes_threshold_crossings():
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
    changes = count_decision_changes(
        candidate,
        reference,
        thresholds={"p_finish_top12": 0.5},
    )
    assert changes["by_metric"]["p_finish_top12"] == 2
    events = enumerate_decision_changes(candidate, reference, thresholds={"p_finish_top12": 0.5})
    assert len(events) == 2


def test_qualifies_and_signed_distance_helpers():
    assert qualifies(0.6, 0.5, "ge")
    assert signed_distance_from_threshold(0.6, 0.5, "ge") == pytest.approx(0.1)


def test_select_smallest_passing_draw_count():
    rows = [
        {"draw_count": 1000, "passes_gate": False},
        {"draw_count": 2000, "passes_gate": True},
        {"draw_count": 5000, "passes_gate": True},
        {"draw_count": 10000, "passes_gate": True},
    ]
    assert select_smallest_passing_draw_count(rows, reference_draws=10000) == 2000
    assert select_smallest_passing_draw_count(rows[:1], reference_draws=10000) is None


def test_verify_contract_match_detects_hash_drift():
    ref = {"selected_board_hash": "abc", "replacement_contract_hash": "def"}
    cand = {"selected_board_hash": "abc", "replacement_contract_hash": "xyz"}
    mismatches = verify_contract_match(cand, ref)
    assert any("replacement_contract_hash" in m for m in mismatches)


def test_candidate_passes_gate_requires_median_and_p95():
    comparison = {
        "per_metric": {
            "p50": {"median_passes": True, "p95_passes": False},
        }
    }
    assert candidate_passes_gate(
        comparison,
        decision_changes={"total": 0},
    ) is False
    assert passes_numerical_stability(comparison) is False


def test_production_v20k_gate_blocks_material_events():
    comparison = {
        "per_metric": {
            "p50": {"median_passes": True, "p95_passes": True},
        }
    }
    production_decision = {
        "material_decision_events": 1,
        "core_adp_decision_events": 0,
    }
    assert candidate_passes_gate(
        comparison,
        decision_changes={"total": 1},
        gate_mode="production_v20k",
        production_decision=production_decision,
    ) is False


def test_production_v20k_gate_blocks_core_adp_events():
    comparison = {
        "per_metric": {
            "p50": {"median_passes": True, "p95_passes": True},
        }
    }
    production_decision = {
        "material_decision_events": 0,
        "core_adp_decision_events": 1,
    }
    assert candidate_passes_gate(
        comparison,
        decision_changes={"total": 1},
        gate_mode="production_v20k",
        production_decision=production_decision,
    ) is False


def test_classify_boundary_vs_material_event():
    from src.projection.evaluation.decision_change_diagnostics import load_decision_diagnostic_config

    config = load_decision_diagnostic_config()
    boundary = {
        "metric_kind": "probability",
        "distance_from_threshold_reference": 0.01,
    }
    material = {
        "metric_kind": "probability",
        "distance_from_threshold_reference": 0.05,
    }
    assert classify_candidate_vs_reference_event(boundary, config=config) == "boundary_noise"
    assert classify_candidate_vs_reference_event(material, config=config) == "material"


def test_evaluate_production_decision_events_counts_core_adp():
    from src.projection.evaluation.draw_stability import resolve_decision_threshold_registry
    from src.projection.evaluation.decision_change_diagnostics import load_decision_diagnostic_config

    candidate = pd.DataFrame(
        {
            "player_id": ["core1"],
            "p_finish_top12": [0.49],
            "p_finish_top24": [0.8],
            "p_vorp_positive": [0.4],
        }
    )
    reference = pd.DataFrame(
        {
            "player_id": ["core1"],
            "p_finish_top12": [0.51],
            "p_finish_top24": [0.8],
            "p_vorp_positive": [0.4],
        }
    )
    player_meta = pd.DataFrame(
        {
            "player_id": ["core1"],
            "position": ["WR"],
            "adp": [10.0],
            "is_core_adp_player": [True],
        }
    )
    result = evaluate_production_decision_events(
        candidate,
        reference,
        threshold_registry=resolve_decision_threshold_registry(),
        player_meta=player_meta,
        diagnostic_config=load_decision_diagnostic_config(),
    )
    assert result["core_adp_decision_events"] == 1
    assert result["total"] == 1


def test_chunk_reordering_invariant_for_prefix_rows():
    """Partition order must not change draw-level rows."""
    draws = _make_draws(8)
    chunk_a = draws[draws["draw"].isin([0, 1, 2, 3])]
    chunk_b = draws[draws["draw"].isin([4, 5, 6, 7])]
    reordered = pd.concat([chunk_b, chunk_a], ignore_index=True)
    assert prefix_is_nested_subset(draws, reordered, max_draw_id=4)
