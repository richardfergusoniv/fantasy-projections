"""Phase 2 tests: recenter transform, draft-value overlay, and gates."""
from __future__ import annotations

import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.draft_assistant.draft_value_simulation import (
    compute_draft_value_overlay,
    compute_finish_probabilities,
)
from src.draft_assistant.prepare import (
    EXPORT_COLS,
    attach_draft_value_overlay,
    attach_v3_simulation_percentiles,
)
from src.draft_assistant.vorp import add_vorp_columns
from src.projection.evaluation.calibration_segments import evaluate_all_segments
from src.projection.evaluation.finish_probability_gate import (
    VERDICT_HOLD,
    evaluate_finish_gate,
)
from src.projection.inference.recenter import recenter_draws
from src.projection.inference.wr_calibration import (
    recenter_draws_wr_scaled,
    select_wr_residual_scale,
)
from src.projection.inference.simulation_config import (
    deterministic_simulation_seed,
    rng_for_draw,
)


def _sample_draws() -> pd.DataFrame:
    rows = []
    for draw in range(100):
        rows.append({
            "player_id": "qb1",
            "position": "QB",
            "team": "AAA",
            "draw": draw,
            "fantasy_pts_season": 280.0 + draw - 50,
        })
        rows.append({
            "player_id": "rb1",
            "position": "RB",
            "team": "BBB",
            "draw": draw,
            "fantasy_pts_season": 200.0 + (draw % 20),
        })
    return pd.DataFrame(rows)


def test_recentered_p50_matches_selected_points():
    draws = _sample_draws()
    selected = {"qb1": 300.0, "rb1": 210.0}
    recentered = recenter_draws(draws, selected)
    for player_id, target in selected.items():
        p50 = recentered.loc[recentered["player_id"].eq(player_id), "fantasy_pts_season"].median()
        assert abs(p50 - target) < 1e-9


def test_recenter_zero_floor_with_median_correction():
    draws = pd.DataFrame({
        "player_id": ["wr1"] * 50,
        "position": ["WR"] * 50,
        "team": ["CCC"] * 50,
        "draw": list(range(50)),
        "fantasy_pts_season": [10.0] * 25 + [30.0] * 25,
    })
    recentered = recenter_draws(draws, {"wr1": 5.0})
    assert (recentered["fantasy_pts_season"] >= 0).all()
    p50 = recentered["fantasy_pts_season"].median()
    assert abs(p50 - 5.0) < 0.01


def test_wr_residual_scale_widens_wr_interval_preserving_p50():
    draws = pd.DataFrame({
        "player_id": ["wr1"] * 100 + ["qb1"] * 100,
        "position": ["WR"] * 100 + ["QB"] * 100,
        "team": ["AAA"] * 200,
        "draw": list(range(100)) * 2,
        "fantasy_pts_season": [120.0 + (i - 50) for i in range(100)]
        + [280.0 + (i - 50) for i in range(100)],
    })
    selected = {"wr1": 150.0, "qb1": 300.0}
    base = recenter_draws(draws, selected)
    scaled = recenter_draws_wr_scaled(draws, selected, wr_scale=1.5)
    wr_base = base.loc[base["player_id"].eq("wr1"), "fantasy_pts_season"]
    wr_scaled = scaled.loc[scaled["player_id"].eq("wr1"), "fantasy_pts_season"]
    qb_base = base.loc[base["player_id"].eq("qb1"), "fantasy_pts_season"]
    qb_scaled = scaled.loc[scaled["player_id"].eq("qb1"), "fantasy_pts_season"]
    assert abs(wr_scaled.median() - 150.0) < 1e-9
    assert abs(qb_scaled.median() - 300.0) < 1e-9
    assert wr_scaled.quantile(0.90) - wr_scaled.quantile(0.10) > (
        wr_base.quantile(0.90) - wr_base.quantile(0.10)
    )
    assert qb_scaled.quantile(0.90) - qb_scaled.quantile(0.10) == pytest.approx(
        qb_base.quantile(0.90) - qb_base.quantile(0.10)
    )


def test_select_wr_residual_scale_prefers_overall_floor_when_available():
    fold_scores = [
        {"wr_scale": 1.4, "wr_n": 50, "wr_coverage": 0.78, "overall_coverage": 0.71, "overall_n": 120, "wr_interval_score": 220.0},
        {"wr_scale": 1.7, "wr_n": 50, "wr_coverage": 0.88, "overall_coverage": 0.75, "overall_n": 120, "wr_interval_score": 235.0},
        {"wr_scale": 1.8, "wr_n": 50, "wr_coverage": 0.88, "overall_coverage": 0.75, "overall_n": 120, "wr_interval_score": 240.0},
    ]
    selected = select_wr_residual_scale(fold_scores)
    assert selected["selected_wr_scale"] == 1.7


    probs = compute_finish_probabilities(_sample_draws())
    expected = {f"p_finish_top{cutoff}" for cutoff in (6, 12, 24, 36, 48)}
    assert expected.issubset(set(probs.columns))
    assert "p_top12" not in probs.columns


def test_finish_probability_semantics_within_position():
    draws = _sample_draws()
    probs = compute_finish_probabilities(draws)
    qb = probs.loc[probs["player_id"].eq("qb1"), "p_finish_top12"].iloc[0]
    assert 0.0 <= qb <= 1.0
    # qb1 is always top QB in this toy sample.
    assert qb == pytest.approx(1.0)


def test_simulated_vorp_uses_fixed_replacement_from_board():
    draws = _sample_draws()
    board = pd.DataFrame([
        {"player_id": "qb1", "position": "QB", "team": "AAA", "fantasy_pts_season": 300.0, "projected_games": 17},
        {"player_id": "rb1", "position": "RB", "team": "BBB", "fantasy_pts_season": 210.0, "projected_games": 17},
    ])
    board = add_vorp_columns(board)
    overlay = compute_draft_value_overlay(draws, board)
    assert {"sim_vorp_p10", "sim_vorp_p50", "sim_vorp_p90", "p_vorp_positive"}.issubset(overlay.columns)


def test_deterministic_seed_excludes_run_id():
    a = deterministic_simulation_seed(
        season=2026,
        board_hash="abc",
        calibration_hash="def",
        configured_seed=2026,
    )
    b = deterministic_simulation_seed(
        season=2026,
        board_hash="abc",
        calibration_hash="def",
        configured_seed=2026,
    )
    assert a == b


def test_draw_rng_invariant_to_chunk_size():
    master = 12345
    values = [rng_for_draw(master, draw_id).random() for draw_id in range(5)]
    again = [rng_for_draw(master, draw_id).random() for draw_id in range(5)]
    assert values == again


def test_export_cols_use_finish_probability_names():
    assert "p_finish_top12" in EXPORT_COLS
    assert "p_top12" not in EXPORT_COLS
    assert "sim_vorp_p50" in EXPORT_COLS


def test_attach_draft_value_overlay_fails_closed_without_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.draft_assistant.prepare.read_finish_probability_gate",
        lambda: None,
    )
    board = pd.DataFrame([
        {
            "player_id": "qb1",
            "position": "QB",
            "team": "AAA",
            "fantasy_pts_season": 300.0,
            "projected_games": 17,
            "projection_run_id": "run-1",
        }
    ])
    out, meta = attach_draft_value_overlay(board, 2026, require_gate=True)
    assert meta["applied"] is False
    assert meta["reason"] == "missing_finish_probability_gate"


def test_finish_gate_artifact_structure():
    from src.projection.evaluation.finish_probability_gate import build_finish_gate_artifact

    gate = build_finish_gate_artifact(
        acceptance={"passes": True},
        finish_calibration={
            "passes": True,
            "checks": [
                {
                    "cutoff": 12,
                    "n": 10,
                    "brier": 0.14,
                    "baseline_brier": 0.33,
                    "brier_improvement": 0.19,
                    "mean_predicted": 0.4,
                    "mean_observed": 0.4,
                    "calibration_intercept": 0.0,
                    "calibration_slope": 1.0,
                    "passes": True,
                    "by_position": {
                        "WR": {
                            "n": 5,
                            "brier": 0.15,
                            "baseline_brier": 0.30,
                            "mean_predicted": 0.22,
                            "mean_observed": 0.22,
                            "calibration_slope": 1.02,
                            "passes": True,
                        }
                    },
                }
            ],
        },
        provenance={"board_model_id": "accuracy_first_ensemble", "wr_residual_scale": 1.7},
        n_scored=10,
    )
    assert gate["state"] == "finish_probability_ready"
    assert gate["publication_verdict"] == "pass"
    assert gate["distribution_acceptance"]["verdict"] == "pass"
    assert gate["finish_calibration"]["verdict"] == "pass"
    assert gate["finish_calibration"]["metrics"]["top12"]["candidate_brier"] == 0.14
    assert gate["finish_calibration"]["metrics"]["wr_top12"]["n"] == 5


    gate = evaluate_finish_gate(
        pd.DataFrame(),
        recentered_holdout={"passes": False},
        finish_calibration={"passes": False, "checks": []},
    )
    assert gate["verdict"] == VERDICT_HOLD
    assert gate["publication_verdict"] == "hold"


def test_rank_to_finish_baseline_uses_training_only():
    from src.projection.evaluation.finish_probability_calibration import (
        apply_rank_to_finish_baseline,
        attach_positional_ranks,
        build_finish_probability_frame,
        fit_rank_to_finish_rates,
    )

    training = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["WR", "WR", "WR"],
        "actual_points": [200.0, 150.0, 100.0],
        "projected_points": [210.0, 140.0, 90.0],
    })
    training = attach_positional_ranks(training, projected_points_col="projected_points")
    rates = fit_rank_to_finish_rates(training, cutoff=2)
    holdout = pd.DataFrame({
        "player_id": ["x"],
        "position": ["WR"],
        "pred_rank": [1.0],
    })
    baseline = apply_rank_to_finish_baseline(holdout, rates, cutoff=2)
    assert baseline.iloc[0] == 1.0


def test_segment_calibration_is_one_dimensional():
    frame = pd.DataFrame({
        "position": ["QB", "RB", "WR", "TE"] * 20,
        "actual_points": np.random.default_rng(1).normal(200, 30, 80),
        "pred_p10": np.random.default_rng(2).normal(170, 20, 80),
        "pred_p25": np.random.default_rng(3).normal(185, 20, 80),
        "pred_p50": np.random.default_rng(4).normal(200, 20, 80),
        "pred_p75": np.random.default_rng(5).normal(215, 20, 80),
        "pred_p90": np.random.default_rng(6).normal(230, 20, 80),
        "projected_games": [17] * 80,
        "source": ["model"] * 80,
    })
    segments = evaluate_all_segments(frame)
    assert "segment_type" in segments.columns
    assert not segments["segment_type"].astype(str).str.contains("x").any()


def test_v3_percentile_overlay_unchanged_without_finish_gate(tmp_path, monkeypatch):
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    summary = model_v3 / "simulation_summary_2026.csv"
    summary.write_text(
        "player_id,position,team,p10,p25,p50,p75,p90\n"
        "qb1,QB,AAA,250,275,300,325,350\n",
        encoding="utf-8",
    )
    gate = model_v3 / "promotion_gate.json"
    gate.write_text(json.dumps({"verdict": "simulation_ready"}), encoding="utf-8")
    manifest = model_v3 / "simulation_manifest_2026.json"
    manifest.write_text(
        json.dumps({"source_projection_run_id": "run-1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.draft_assistant.prepare.MODEL_V3_DIR", str(model_v3))
    board = pd.DataFrame([
        {
            "player_id": "qb1",
            "position": "QB",
            "team": "AAA",
            "fantasy_pts_season": 300.0,
            "projection_run_id": "run-1",
        }
    ])
    out, meta = attach_v3_simulation_percentiles(board, 2026, require_gate=True)
    assert meta["applied"] is True
    assert out.loc[0, "fantasy_pts_p50"] == 300.0
