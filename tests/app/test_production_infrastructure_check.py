"""Tests for production infrastructure audit script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.production_infrastructure_check import _parse_database_url, run_audit


def test_parse_database_url_extracts_host_and_db():
    info = _parse_database_url("postgresql+psycopg://fantasy:secret@localhost:5432/fantasy_app")
    assert info["user"] == "fantasy"
    assert info["host"] == "localhost"
    assert info["port"] == 5432
    assert info["dbname"] == "fantasy_app"


def test_production_audit_reports_config_blockers(tmp_path: Path):
    audit = run_audit(
        database_url="postgresql+psycopg://fantasy:fantasy@localhost:5432/fantasy_app",
        report_path=tmp_path / "audit.json",
    )
    assert audit["verdict"]["phone_access_ready"] is False
    assert audit["verdict"]["cloud_configuration_ready"] is False
    assert audit["verdict"]["config_blockers"]
    assert "production_config_problems" in audit["configuration"]
    assert audit["runtime"]["backup_script"] is True
    assert (tmp_path / "audit.json").is_file()
    payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert payload["verdict"]["phone_access_ready"] is False


def test_production_audit_passes_with_valid_env_file(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "APP_SECRET_KEY=" + ("x" * 64),
                "APP_ALLOWED_EMAIL=owner@realdomain.com",
                "APP_PUBLIC_URL=https://fantasy.realdomain.com",
                "APP_CORS_ORIGINS=https://fantasy.realdomain.com",
                "TRUSTED_HOSTS=fantasy.realdomain.com",
                "EMAIL_PROVIDER=resend",
                "RESEND_API_KEY=re_test_key",
                "CRON_SECRET=" + ("y" * 48),
                "DATABASE_URL=postgresql+psycopg://fantasy:secret@localhost:5432/fantasy_app",
            ]
        ),
        encoding="utf-8",
    )
    audit = run_audit(
        env_file=env,
        database_url="postgresql+psycopg://fantasy:secret@localhost:5432/fantasy_app",
        report_path=tmp_path / "pass.json",
    )
    assert audit["configuration"]["production_ready"] is True
    assert audit["verdict"]["cloud_configuration_ready"] is True
    assert audit["verdict"]["phone_access_ready"] is True
