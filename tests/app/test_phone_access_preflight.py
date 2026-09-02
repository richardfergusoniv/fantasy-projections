"""Tests for phone access preflight script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.phone_access_preflight import run_preflight


def test_preflight_reports_missing_secrets(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "web" / "node_modules").mkdir(parents=True)
    (tmp_path / "config" / "sleeper_owner.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config" / "phone_access.secrets.example.json").write_text("{}", encoding="utf-8")
    (tmp_path / "output" / "live_pg").mkdir(parents=True)
    (tmp_path / "output" / "live_pg" / "go_live_gate.json").write_text(
        json.dumps({"summary": {"passed_count": 7, "total": 8}}),
        encoding="utf-8",
    )

    report = run_preflight(root=tmp_path)
    assert report["blockers"] == ["phone_access_secrets"]
    assert report["ready_for_stack"] is True
