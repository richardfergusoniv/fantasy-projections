"""Tests for two-stage release monitoring reports."""
from __future__ import annotations

import json
from pathlib import Path

from src.projection.evaluation.draw_count_rollout import DRAW_COUNT_RISK_FLAG_10K
from src.projection.evaluation.release_report import (
    build_release_report_board,
    build_release_report_simulation,
    merge_release_reports,
)
from src.projection.release_bundle import SCHEMA_VERSION, SCHEMA_VERSION_V2


def test_build_release_report_simulation_flags_missing_draw_stability(tmp_path, monkeypatch):
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("src.projection.evaluation.release_report.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.evaluation.release_report.OUTPUT_DIR", str(output))
    report = build_release_report_simulation(season=2026)
    assert "draw_stability: not_run" in report["summary_risks"]


def test_build_release_report_propagates_10k_risk_when_strict_gate_false(tmp_path, monkeypatch):
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("src.projection.evaluation.release_report.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.evaluation.release_report.OUTPUT_DIR", str(output))
    (model_v3 / "draw_count_rollout_decision.json").write_text(
        json.dumps(
            {
                "schema_version": "draw_count_rollout_decision_v2",
                "strict_gate_promotion": False,
                "release_report_risk_flag": DRAW_COUNT_RISK_FLAG_10K,
                "current_production_profile": "decision_stable_compromise_10000",
                "chosen_production_draw_count": 10000,
                "phase_2_status": "closed",
            }
        ),
        encoding="utf-8",
    )
    report = build_release_report_simulation(season=2026)
    assert DRAW_COUNT_RISK_FLAG_10K in report["summary_risks"]
    assert report["simulation"]["draw_count_policy"]["strict_gate_promotion"] is False
    assert DRAW_COUNT_RISK_FLAG_10K in report["simulation"]["summary_risks"]


def test_build_release_report_skips_10k_risk_when_strict_gate_true(tmp_path, monkeypatch):
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("src.projection.evaluation.release_report.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.evaluation.release_report.OUTPUT_DIR", str(output))
    (model_v3 / "draw_count_rollout_decision.json").write_text(
        json.dumps(
            {
                "strict_gate_promotion": True,
                "release_report_risk_flag": DRAW_COUNT_RISK_FLAG_10K,
            }
        ),
        encoding="utf-8",
    )
    report = build_release_report_simulation(season=2026)
    assert DRAW_COUNT_RISK_FLAG_10K not in report["summary_risks"]


def test_merge_release_reports_combines_risks():
    sim = build_release_report_simulation(season=2026, projection_run={"run_id": "r1"})
    board = {
        "schema_version": "release_report_v1",
        "stage": "board",
        "summary_risks": ["sim_vorp_*: not_attached"],
    }
    merged = merge_release_reports(sim, board)
    assert "board" in merged
    assert "sim_vorp_*: not_attached" in merged["summary_risks"]
