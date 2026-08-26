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
