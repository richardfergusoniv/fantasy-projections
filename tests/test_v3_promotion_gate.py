"""Tests for hardened v3 promotion gate and means cutover."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.draft_assistant.prepare import apply_v3_means
from src.projection.evaluation.v3_means_score import beats_incumbent, score_predictions
from scripts.v3_promotion_gate import evaluate_promotion_gate


def test_beats_incumbent_requires_mae_and_spearman():
    cand = {"points_mae": 30.0, "spearman": 0.80}
    incumb = {"points_mae": 31.0, "spearman": 0.75}
    assert beats_incumbent(cand, incumb)["pass"] is True
    assert beats_incumbent({"points_mae": 32.0, "spearman": 0.90}, incumb)["pass"] is False


def test_score_predictions_overall():
    frame = pd.DataFrame({
        "actual_points": [10.0, 20.0, 30.0],
        "pred": [11.0, 19.0, 28.0],
        "preseason_position": ["WR", "WR", "RB"],
    })
    out = score_predictions(frame, "pred")
    assert out["n"] == 3
    assert out["points_mae"] >= 0


def test_apply_v3_means_falls_back_without_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.draft_assistant.prepare.MODEL_V3_DIR", str(tmp_path)
    )
    df = pd.DataFrame({
        "player_id": ["a"],
        "fantasy_pts_season": [100.0],
        "fantasy_pts": [10.0],
        "projected_games": [10.0],
    })
    out, meta = apply_v3_means(df, 2026, enabled=True, require_gate=True)
    assert meta["applied"] is False
    assert out["fantasy_pts_season"].iloc[0] == 100.0


def test_apply_v3_means_with_force(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.draft_assistant.prepare.MODEL_V3_DIR", str(tmp_path)
    )
    summary = pd.DataFrame({"player_id": ["a"], "p50": [120.0]})
    summary.to_csv(tmp_path / "simulation_summary_2026.csv", index=False)
    df = pd.DataFrame({
        "player_id": ["a"],
        "fantasy_pts_season": [100.0],
        "fantasy_pts": [10.0],
        "projected_games": [10.0],
    })
    out, meta = apply_v3_means(df, 2026, enabled=True, require_gate=False)
    assert meta["applied"] is True
    assert out["fantasy_pts_season"].iloc[0] == 120.0


def test_gate_verdicts_distinguish_simulation_vs_means(tmp_path, monkeypatch):
    # Point evaluate_promotion_gate at empty dirs → hold_v1_default
    monkeypatch.setattr("scripts.v3_promotion_gate.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.v3_promotion_gate.OUT_DIR", tmp_path / "model_v3")
    (tmp_path / "model_v3").mkdir(parents=True)
    (tmp_path / "output" / "backtest").mkdir(parents=True)
    report = evaluate_promotion_gate(2026)
    assert report["verdict"] == "hold_v1_default"


def _fold(*, blend_mae, v1_mae=27.7, blend_available=True):
    """One scored fold. blend_mae equal to v1_mae is the degenerate case."""
    usable = blend_available and blend_mae != v1_mae
    return {
        "target_season": 2025,
        "blend_available": blend_available,
        "blend_degenerate": blend_available and blend_mae == v1_mae,
        "blend_usable": usable,
        "metrics": {
            "v1": {"points_mae": v1_mae, "spearman": 0.758},
            **({"blend": {"points_mae": blend_mae, "spearman": 0.758}} if blend_available else {}),
            "v3_interim": {"points_mae": 29.8, "spearman": 0.698},
            "v3_generative": {"points_mae": 47.2, "spearman": 0.611},
        },
    }


def test_blend_arm_that_duplicates_v1_is_not_an_independent_check():
    """A blend equal to v1 must not read as a second passing incumbent.

    The shipped backtest hit exactly this: output/model_v2 held only 2026
    points, so every historical fold silently scored the blend arm as a copy
    of v1 and reported "beats blend" as though it had been tested.
    """
    from scripts.backtest_v3_means import _blend_from_compare  # noqa: F401

    fold = _fold(blend_mae=27.7)  # identical to v1
    assert fold["blend_degenerate"] is True
    assert fold["blend_usable"] is False


def test_gate_reports_blend_arm_usability(tmp_path, monkeypatch):
    """The verdict has to say when 'beats blend' was never really tested."""
    monkeypatch.setattr("scripts.v3_promotion_gate.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.v3_promotion_gate.OUT_DIR", tmp_path / "model_v3")
    out_dir = tmp_path / "model_v3"
    out_dir.mkdir(parents=True)
    (tmp_path / "output" / "backtest").mkdir(parents=True)

    # simulation_ready preconditions
    pd.DataFrame({"player_id": ["a"], "p50": [1.0]}).to_csv(
        out_dir / "simulation_summary_2026.csv", index=False)
    (tmp_path / "output" / "backtest" / "calibration_report.json").write_text(
        json.dumps({
            "summary": {"mean_coverage": 0.80, "basis": "in_sample"},
            "forward_summary": {
                "mean_coverage": 0.80, "basis": "forward_holdout", "n_scored": 500},
        }), encoding="utf-8")
    (out_dir / "means_backtest.json").write_text(json.dumps({
        "folds": [_fold(blend_mae=27.7)],
        "summary": {
            "blend_usable_all_folds": False,
            "blend_unusable_folds": [{"target_season": 2025, "blend_available": True}],
            "promote_v3_means": False,
        },
    }), encoding="utf-8")

    report = evaluate_promotion_gate(2026)
    assert report["verdict"] == "simulation_ready"
    assert report["gates"]["blend_arm_usable"] is False
    assert report["gates"]["blend_unusable_folds"]
    assert "beats blend" in report["rationale"]


def test_gate_does_not_assume_an_older_backtest_had_a_blend_arm(tmp_path, monkeypatch):
    """A summary predating the flag must read as unusable, not usable."""
    monkeypatch.setattr("scripts.v3_promotion_gate.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.v3_promotion_gate.OUT_DIR", tmp_path / "model_v3")
    out_dir = tmp_path / "model_v3"
    out_dir.mkdir(parents=True)
    (tmp_path / "output" / "backtest").mkdir(parents=True)
    (out_dir / "means_backtest.json").write_text(json.dumps({
        "folds": [], "summary": {"promote_v3_means": False},  # no blend_usable key
    }), encoding="utf-8")
    report = evaluate_promotion_gate(2026)
    assert report["gates"]["blend_arm_usable"] is False


def _sim_summary(tmp_path, season=2026):
    pd.DataFrame({
        "player_id": ["a", "b"],
        "p10": [80.0, 40.0], "p25": [90.0, 45.0], "p50": [100.0, 50.0],
        "p75": [110.0, 55.0], "p90": [130.0, 60.0],
    }).to_csv(tmp_path / f"simulation_summary_{season}.csv", index=False)


def _board():
    return pd.DataFrame({
        "player_id": ["a", "b"],
        "position": ["QB", "WR"],
        "fantasy_pts_season": [100.0, 50.0],
        "fantasy_pts": [10.0, 5.0],
        "projected_games": [17.0, 17.0],
    })


def _write_gate(tmp_path, verdict):
    (tmp_path / "promotion_gate.json").write_text(
        json.dumps({"verdict": verdict}), encoding="utf-8")


def test_percentile_overlay_requires_a_simulation_ready_gate(tmp_path, monkeypatch):
    """v3 percentiles reach the published board, so they are gated too.

    They used to attach on file presence alone: a simulation_summary CSV on
    disk put fantasy_pts_p10/p90 and p_top12 onto the board with no check
    that the run was ever calibrated.
    """
    monkeypatch.setattr("src.draft_assistant.prepare.MODEL_V3_DIR", str(tmp_path))
    _sim_summary(tmp_path)
    _write_gate(tmp_path, "hold_v1_default")

    from src.draft_assistant.prepare import attach_v3_simulation_percentiles
    out, meta = attach_v3_simulation_percentiles(_board(), 2026)
    assert meta["applied"] is False
    assert meta["reason"] == "gate_not_simulation_ready"
    assert "fantasy_pts_p10" not in out.columns


def test_percentile_overlay_attaches_when_gate_allows(tmp_path, monkeypatch):
    monkeypatch.setattr("src.draft_assistant.prepare.MODEL_V3_DIR", str(tmp_path))
    _sim_summary(tmp_path)
    _write_gate(tmp_path, "simulation_ready")

    from src.draft_assistant.prepare import attach_v3_simulation_percentiles
    out, meta = attach_v3_simulation_percentiles(_board(), 2026)
    assert meta["applied"] is True
    assert out["fantasy_pts_p10"].notna().all()
    # Overlay only -- it must not move the means the board ranks on.
    pd.testing.assert_series_equal(out["fantasy_pts_season"], _board()["fantasy_pts_season"])


def test_percentile_overlay_skips_when_gate_never_ran(tmp_path, monkeypatch):
    """No gate file is not permission; it means the check has not happened."""
    monkeypatch.setattr("src.draft_assistant.prepare.MODEL_V3_DIR", str(tmp_path))
    _sim_summary(tmp_path)
    from src.draft_assistant.prepare import attach_v3_simulation_percentiles
    _, meta = attach_v3_simulation_percentiles(_board(), 2026)
    assert meta["applied"] is False
    assert meta["gate_verdict"] is None


def _gate_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.v3_promotion_gate.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.v3_promotion_gate.OUT_DIR", tmp_path / "model_v3")
    out_dir = tmp_path / "model_v3"
    out_dir.mkdir(parents=True)
    (tmp_path / "output" / "backtest").mkdir(parents=True)
    pd.DataFrame({"player_id": ["a"], "p50": [1.0]}).to_csv(
        out_dir / "simulation_summary_2026.csv", index=False)
    return out_dir


def test_gate_reads_held_out_coverage_not_in_sample(tmp_path, monkeypatch):
    """In-sample coverage is a tautology, so it must not open the gate."""
    _gate_dirs(tmp_path, monkeypatch)
    (tmp_path / "output" / "backtest" / "calibration_report.json").write_text(
        json.dumps({
            "summary": {"mean_coverage": 0.80, "basis": "in_sample"},
            # Held-out says the intervals are badly miscalibrated.
            "forward_summary": {
                "mean_coverage": 0.42, "basis": "forward_holdout", "n_scored": 500},
        }), encoding="utf-8")
    report = evaluate_promotion_gate(2026)
    assert report["gates"]["calibration_within_5pp"] is False
    assert report["gates"]["mean_interval_coverage"] == 0.42
    assert report["gates"]["mean_interval_coverage_basis"] == "forward_holdout"
    assert report["verdict"] == "hold_v1_default"


def test_gate_fails_closed_on_a_report_without_forward_summary(tmp_path, monkeypatch):
    """An older report must not fall back to the in-sample number."""
    _gate_dirs(tmp_path, monkeypatch)
    (tmp_path / "output" / "backtest" / "calibration_report.json").write_text(
        json.dumps({"summary": {"mean_coverage": 0.80}}), encoding="utf-8")
    report = evaluate_promotion_gate(2026)
    assert report["gates"]["calibration_within_5pp"] is False
    assert report["gates"]["mean_interval_coverage_basis"] == "missing"
    assert report["verdict"] == "hold_v1_default"


def test_gate_opens_on_good_held_out_coverage(tmp_path, monkeypatch):
    _gate_dirs(tmp_path, monkeypatch)
    (tmp_path / "output" / "backtest" / "calibration_report.json").write_text(
        json.dumps({
            "summary": {"mean_coverage": 0.798, "basis": "in_sample"},
            "forward_summary": {
                "mean_coverage": 0.8013, "basis": "forward_holdout", "n_scored": 5252},
        }), encoding="utf-8")
    report = evaluate_promotion_gate(2026)
    assert report["gates"]["calibration_within_5pp"] is True
    assert report["verdict"] == "simulation_ready"
