"""Tests for unified go-live gate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.live_go_live_gate import run_gate


def test_go_live_gate_reports_objective_status(tmp_path: Path, monkeypatch):
    sleeper = {
        "status": "passed",
        "go_no_go": "go",
        "league_selection": {"imported_count": 6},
        "recommendations": [{"league_id": "1"}],
        "completeness": {"rostered_player_ids": 100},
    }
    readiness = {
        "sealed_bundle": {"namespace": "v2_baseline_20260830", "caveats": []},
        "scoring_summary": {"league_count": 6, "all_publishable": True},
        "verdict": {"first_live_overlay_promoted": True},
        "identity": {"unresolved_count": 0},
        "injury_research": {"mode": "disabled", "synthetic": False},
        "conservation": {"exit_code": 0},
        "overlay_candidate": {"promoted": True, "from_live_data": True, "overlay_hash": "abc"},
        "overlay_promotion": {"fixture_citations_in_artifact": 0},
        "projection_deltas": {"adjustment_count": 13},
    }
    infra = {
        "verdict": {"phone_access_ready": False, "blockers": ["config:test"]},
        "runtime": {"postgresql": {"status": "ok"}, "backup_script": True},
    }

    sleeper_path = tmp_path / "sleeper.json"
    readiness_path = tmp_path / "readiness.json"
    infra_path = tmp_path / "infra.json"
    sleeper_path.write_text(json.dumps(sleeper), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    infra_path.write_text(json.dumps(infra), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.live_go_live_gate._run",
        lambda *a, **k: (0, ""),
    )

    gate = run_gate(
        sleeper_report=sleeper_path,
        readiness_report=readiness_path,
        infra_report=infra_path,
        backup_dir=tmp_path,
        report_path=tmp_path / "gate.json",
    )
    assert gate["summary"]["passed_count"] == 7
    assert gate["summary"]["total"] == 8
    assert "8_phone_access_infrastructure" in gate["summary"]["failed_objectives"]
    assert gate["objectives"]["6_live_overlay_promoted"]["passed"] is True
