"""PgBouncer / serverless engine connect-arg safety."""

from __future__ import annotations

from src.app.persistence import database as dbmod
from src.app.persistence.database import _needs_pgbouncer_safe_connect


def test_supabase_transaction_pooler_dsn_detected() -> None:
    url = (
        "postgresql+psycopg://fantasy_app_runtime:x@"
        "aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    )
    assert _needs_pgbouncer_safe_connect(url)


def test_explicit_pgbouncer_flag_detected() -> None:
    url = "postgresql+psycopg://u:p@db.example.com:5432/postgres?pgbouncer=true"
    assert _needs_pgbouncer_safe_connect(url)


def test_direct_supabase_db_host_not_flagged() -> None:
    url = "postgresql+psycopg://u:p@db.dbvwgfefdorugdtpxgcj.supabase.co:5432/postgres"
    assert not _needs_pgbouncer_safe_connect(url)


def _capture_connect_args(url: str, *, serverless: bool) -> dict[str, object]:
    captured: dict[str, object] = {}
    real_create = dbmod.create_engine

    def _capture(url_arg, **kwargs):  # type: ignore[no-untyped-def]
        captured["connect_args"] = dict(kwargs.get("connect_args") or {})
        captured["poolclass"] = kwargs.get("poolclass")
        return real_create("sqlite+pysqlite:///:memory:", future=True)

    original = dbmod.create_engine
    try:
        dbmod.create_engine = _capture  # type: ignore[assignment]
        dbmod._build_engine(url, serverless=serverless)
    finally:
        dbmod.create_engine = original  # type: ignore[assignment]
    return captured


def test_serverless_pooler_engine_disables_psycopg_prepared_statements() -> None:
    url = (
        "postgresql+psycopg://fantasy_app_runtime:x@"
        "aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    )
    captured = _capture_connect_args(url, serverless=True)
    assert captured["poolclass"] is dbmod.NullPool
    assert captured["connect_args"].get("prepare_threshold") is None


def test_non_pooler_serverless_still_disables_prepares() -> None:
    """NullPool serverless paths always disable prepares as defense in depth."""
    url = "postgresql+psycopg://u:p@db.example.com:5432/postgres"
    captured = _capture_connect_args(url, serverless=True)
    assert captured["connect_args"].get("prepare_threshold") is None


def test_session_pooler_non_serverless_disables_prepares_when_pooler_host() -> None:
    url = (
        "postgresql+psycopg://fantasy_app_runtime:x@"
        "aws-0-us-west-1.pooler.supabase.com:5432/postgres"
    )
    captured = _capture_connect_args(url, serverless=False)
    assert captured["connect_args"].get("prepare_threshold") is None
    assert captured.get("poolclass") is None
