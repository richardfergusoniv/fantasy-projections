"""Tests for two-stage release monitoring reports."""
from __future__ import annotations

import json
from pathlib import Path

from src.projection.evaluation.release_report import (
    build_release_report_board,
    build_release_report_simulation,
    merge_release_reports,
)


def test_build_release_report_simulation_flags_missing_draw_stability(tmp_path, monkeypatch):
    model_v3 = tmp_path / "model_v3"
    model_v3.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("src.projection.evaluation.release_report.MODEL_V3_DIR", str(model_v3))
    monkeypatch.setattr("src.projection.evaluation.release_report.OUTPUT_DIR", str(output))
    report = build_release_report_simulation(season=2026)
    assert "draw_stability: not_run" in report["summary_risks"]


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
