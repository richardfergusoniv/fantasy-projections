"""CLI dry-run writes a Role 2 summary JSON and does not promote."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.projection.weekly_eval.cli import DEFAULT_OUTPUT_DIR, main


def test_dry_run_writes_summary_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_dir = tmp_path / "shadow_vegas_props_compare"
    monkeypatch.delenv("APP_PROJECTION_SOURCE", raising=False)
    code = main(["--dry-run", "--output", str(out_dir)])
    assert code == 0
    summary_path = out_dir / "summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["role"] == "evaluation_comparator"
    assert payload["promoting"] is False
    assert payload["gate_verdict"] == "not_promoting"
    assert payload["dry_run"] is True
    assert payload["n_matched"] >= 3
    assert "not a promotion sample" in payload["caveat"].lower()
    assert "APP_PROJECTION_SOURCE" not in payload
    assert os.environ.get("APP_PROJECTION_SOURCE") is None


def test_default_output_dir_is_shadow_path():
    assert DEFAULT_OUTPUT_DIR.as_posix().endswith("output/shadow_vegas_props_compare")
