"""Contracts for the accuracy-first v1/v2/v3/ADP bake-off."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_accuracy_first_ensemble import (
    ARM_MODELS,
    _apply_arms,
    _parity_check,
    apply_selected_2026,
    verify_published_board,
)
from src.projection.evaluation.accuracy_first import (
    POSITIONS,
    apply_market_curves,
    apply_position_weights,
    canonical_json_hash,
    fit_market_curves,
    fit_position_weights,
    incumbent_points,
    load_consensus_snapshot,
    resolve_candidate_inputs,
    simplex_weights,
)


def _all_positions_frame() -> pd.DataFrame:
    rows = []
    for pos_idx, position in enumerate(POSITIONS):
        for rank in range(1, 11):
            actual = 300.0 - pos_idx * 20.0 - rank * 8.0
            rows.append({
                "player_id": f"{position}-{rank}",
                "position": position,
                "actual_points": actual,
                "v1_pred": actual + 10.0,
                "v2_pred": actual + 5.0,
                "v3_p50": actual,
                "adp_points": actual - 5.0,
                "adp": float(rank),
            })
    return pd.DataFrame(rows)


def test_market_snapshot_must_be_preseason_and_match_target(tmp_path):
    path = tmp_path / "consensus.json"
    path.write_text(json.dumps({
        "meta": {"season": 2025, "as_of": "2025-09-01"},
        "rows": [{"player_id": "a", "position": "WR", "adp": 1, "ecr": None}],
    }), encoding="utf-8")
    rows, meta = load_consensus_snapshot(path, expected_season=2025)
    assert meta["as_of"] == "2025-09-01"
    assert rows.loc[0, "adp"] == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["as_of"] = "2025-09-08"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not demonstrably preseason"):
        load_consensus_snapshot(path, expected_season=2025)
    with pytest.raises(ValueError, match="season"):
        load_consensus_snapshot(path, expected_season=2024)


def test_market_snapshot_accepts_explicit_adp_window_end_date(tmp_path):
    path = tmp_path / "consensus.json"
    path.write_text(json.dumps({
        "meta": {"season": 2026, "adp": {"end_date": "2026-08-23"}},
        "rows": [{"player_id": "a", "position": "WR", "adp": 1}],
    }), encoding="utf-8")
    _, meta = load_consensus_snapshot(path, expected_season=2026)
    assert meta["as_of"] == "2026-08-23"
    assert meta["as_of_source"] == "meta.adp.end_date"


def test_market_curve_is_monotonic_and_deterministic():
    frame = _all_positions_frame()
    curves_a = fit_market_curves(frame)
    curves_b = fit_market_curves(frame.sample(frac=1.0, random_state=99))
    scored_a = apply_market_curves(frame, curves_a)
    scored_b = apply_market_curves(frame, curves_b)
    np.testing.assert_allclose(scored_a, scored_b)
    for position in POSITIONS:
        points = scored_a[frame["position"].eq(position)].to_numpy()
        assert np.all(np.diff(points) <= 1e-12)


def test_simplex_and_fitted_weights_are_nonnegative_and_sum_to_one():
    assert len(list(simplex_weights(4))) == 1771
    weights = fit_position_weights(
        _all_positions_frame(), ("v1_pred", "v2_pred", "v3_p50", "adp_points")
    )
    for position in POSITIONS:
        assert sum(weights[position].values()) == pytest.approx(1.0)
        assert min(weights[position].values()) >= 0.0
        # v3 is exact in the fixture, so the MAE fit must find it.
        assert weights[position]["v3_p50"] == pytest.approx(1.0)


def test_weight_fit_and_application_are_deterministic():
    frame = _all_positions_frame()
    columns = ("v1_pred", "v2_pred", "v3_p50")
    first = fit_position_weights(frame, columns)
    second = fit_position_weights(frame.sample(frac=1.0, random_state=7), columns)
    assert first == second
    out_a = apply_position_weights(frame, first, out_col="pred")
    out_b = apply_position_weights(frame, second, out_col="pred")
    pd.testing.assert_series_equal(out_a["pred"], out_b["pred"])


def test_missing_v2_and_v3_fall_back_to_v1_and_incumbent():
    frame = pd.DataFrame({
        "player_id": ["a", "b"],
        "position": ["QB", "WR"],
        "v1_pred": [100.0, 80.0],
        "v2_pred": [np.nan, 90.0],
        "v3_p50": [np.nan, 70.0],
        "adp_points": [110.0, 75.0],
    })
    resolved = resolve_candidate_inputs(frame)
    assert resolved.loc[0, "v2_pred"] == 100.0
    assert resolved.loc[0, "v3_p50"] == 100.0
    weights = {
        "QB": {"v1_pred": 0.4, "v2_pred": 0.6},
        "WR": {"v1_pred": 1.0, "v2_pred": 0.0},
    }
    assert incumbent_points(resolved, weights).tolist() == [100.0, 80.0]


def test_2026_application_only_changes_top120_selected_positions():
    frame = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["QB", "QB", "WR"],
        "draft_relevant_top120": [True, False, True],
        "incumbent_pred": [100.0, 90.0, 80.0],
        "v1_pred": [100.0, 90.0, 80.0],
        "v2_pred": [120.0, 110.0, 70.0],
        "v3_p50": [130.0, 120.0, 60.0],
        "adp_points": [140.0, np.nan, 50.0],
    })
    specs = {
        "QB": {"arm": "full", "weights": {
            "v1_pred": 0.25, "v2_pred": 0.25, "v3_p50": 0.25, "adp_points": 0.25,
        }},
        "WR": {"arm": "incumbent", "weights": {"v1_pred": 1.0, "v2_pred": 0.0}},
    }
    out = apply_selected_2026(frame, specs)
    assert out.loc[0, "accuracy_ensemble_pred"] == pytest.approx(122.5)
    assert bool(out.loc[0, "accuracy_ensemble_applied"]) is True
    assert out.loc[1, "accuracy_ensemble_pred"] == 90.0
    assert out.loc[2, "accuracy_ensemble_pred"] == 80.0


def test_candidate_arms_score_rows_not_the_population_mask():
    frame = _all_positions_frame()
    frame["draft_relevant_top120"] = frame["adp"].le(5)
    weights = {
        arm: {
            position: {
                column: (1.0 if column == columns[0] else 0.0)
                for column in columns
            }
            for position in POSITIONS
        }
        for arm, columns in ARM_MODELS.items()
    }
    out = _apply_arms(frame, weights)
    for arm in ARM_MODELS:
        assert out.loc[out["draft_relevant_top120"], f"{arm}_pred"].notna().all()
        assert out.loc[~out["draft_relevant_top120"], f"{arm}_pred"].isna().all()


def test_equal_candidate_forecasts_prefer_the_simpler_arm():
    from src.projection.evaluation.accuracy_first import choose_position_arms

    frame = _all_positions_frame()
    frame["incumbent_pred"] = frame["v1_pred"]
    frame["small_pred"] = frame["v3_p50"]
    frame["large_pred"] = frame["v3_p50"]
    selected, _ = choose_position_arms(
        frame,
        {"small": "small_pred", "large": "large_pred"},
        arm_complexity={"small": 2, "large": 4},
    )
    assert set(selected.values()) == {"small"}


def test_published_board_reproduces_selected_points():
    signals = pd.DataFrame({
        "player_id": ["a", "b"], "accuracy_ensemble_pred": [100.0, 80.0],
    })
    board = pd.DataFrame({
        "player_id": ["a", "b"], "fantasy_pts_season": [100.0, 80.0],
    })
    assert verify_published_board(board, signals)["pass"] is True
    board.loc[1, "fantasy_pts_season"] = 81.0
    with pytest.raises(RuntimeError, match="does not reproduce"):
        verify_published_board(board, signals)


def test_exact_v3_parity_requires_identical_live_path_metrics():
    expected = {
        "folds": [{
            "target_season": 2025,
            "joint_bootstrap": {"overall": {
                "n": 100, "coverage": 0.75, "p50_mae": 30.0, "p50_spearman": 0.8,
            }},
        }],
    }
    observed = {"overall": {
        "n": 100, "coverage": 0.75, "p50_mae": 30.0, "p50_spearman": 0.8,
    }}
    assert _parity_check(2025, observed, expected)["pass"] is True
    observed["overall"]["p50_mae"] = 30.01
    assert _parity_check(2025, observed, expected)["pass"] is False


def test_canonical_hash_detects_weight_or_provenance_change():
    payload = {"weights": {"WR": {"v1": 1.0}}, "source_hash": "abc"}
    first = canonical_json_hash(payload)
    assert first == canonical_json_hash(dict(payload))
    changed = {**payload, "source_hash": "def"}
    assert canonical_json_hash(changed) != first
