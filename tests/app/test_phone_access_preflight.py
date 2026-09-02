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
    assert "phone_access_secrets" in report["blockers"]
    assert report["ready_for_production"] is False
    assert "production_env" in {check["name"] for check in report["checks"]}


def test_preflight_production_ready_with_env(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "web" / "node_modules").mkdir(parents=True)
    (tmp_path / "config" / "sleeper_owner.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config" / "phone_access.secrets.example.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config" / "phone_access.secrets.json").write_text(
        json.dumps({"allowed_email": "owner@realdomain.com", "resend_api_key": "re_live_key"}),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "APP_SECRET_KEY=" + ("x" * 64),
                "APP_ALLOWED_EMAIL=owner@realdomain.com",
                "APP_PUBLIC_URL=https://fantasy.realdomain.com",
                "APP_CORS_ORIGINS=https://fantasy.realdomain.com",
                "TRUSTED_HOSTS=fantasy.realdomain.com",
                "DATABASE_URL=postgresql+psycopg://runtime@localhost:5432/fantasy_app",
                "CRON_SECRET=" + ("y" * 48),
            ]
        ),
        encoding="utf-8",
    )

    report = run_preflight(root=tmp_path)
    assert report["ready_for_production"] is True
    assert report["blockers"] == []
