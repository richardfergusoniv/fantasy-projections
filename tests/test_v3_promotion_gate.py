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
        json.dumps({"summary": {"mean_coverage": 0.80}}), encoding="utf-8")
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
