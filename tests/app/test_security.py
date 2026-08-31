"""Regression tests for hardened security behaviour.

Each test here maps to a defect that was reproducible before the fix: unsafe
production config, presence-only CSRF, insecure session cookies, replayable
magic links, leaked exception strings, unbounded request bodies, and
unredacted logs.
"""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

os.environ["APP_ENV"] = "test"
os.environ["APP_ENABLE_DEV_AUTH"] = "true"
os.environ["TEST_DATABASE_URL"] = "sqlite+pysqlite:///:memory:?cache=shared"

PRODUCTION_SAFE = {
    "app_env": "production",
    "app_secret_key": "x" * 48,
    "app_enable_dev_auth": False,
    "email_provider": "smtp",
    "app_allowed_email": "real.owner@example.org",
    "app_cors_origins": "https://app.example.org",
    "app_public_url": "https://app.example.org",
    "database_url": "postgresql+psycopg://u:p@db:5432/app",
    "artifact_backend": "local",
    "trusted_hosts": "app.example.org",
}


def _settings(**overrides):
    from src.app.config import Settings

    return Settings(**{**PRODUCTION_SAFE, **overrides})


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_ENABLE_DEV_AUTH", "true")
    from src.app.config import get_settings
    from src.app.factory import create_app
    from src.app.middleware.rate_limit import limiter
    from src.app.persistence.database import get_session, init_db
    from src.app.seed import seed_development_data

    get_settings.cache_clear()
    # The limiter is a process-global; isolate this test from earlier files.
    limiter.reset()
    init_db()
    with get_session() as session:
        seed_development_data(session, email="owner@example.com")
    with TestClient(create_app()) as test_client:
        yield test_client
    limiter.reset()


def _login(client: TestClient) -> str:
    response = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
    token = response.json()["development_link"].split("token=")[-1]
    return client.post("/api/v1/auth/verify", json={"token": token}).json()["csrf_token"]


# --------------------------------------------------------------------------
# A. Production configuration fails closed
# --------------------------------------------------------------------------


def test_safe_production_settings_validate():
    _settings().validate_production()


def test_non_production_env_is_not_gated():
    from src.app.config import Settings

    Settings(app_env="test", app_secret_key="dev-only-change-me").validate_production()


def test_production_validation_reports_every_problem_at_once():
    from src.app.config import ProductionConfigError

    settings = _settings(
        app_secret_key="dev-only-change-me",
        app_enable_dev_auth=True,
        email_provider="development",
        app_allowed_email="owner@example.com",
        app_cors_origins="*,http://evil.example.com",
        app_public_url="http://app.example.org",
        database_url="sqlite+pysqlite:///./app.db",
        artifact_backend="s3",
        s3_access_key_id=None,
        s3_secret_access_key=None,
        s3_bucket="",
        trusted_hosts="*",
        sleeper_use_fixtures=True,
    )
    with pytest.raises(ProductionConfigError) as excinfo:
        settings.validate_production()

    problems = " | ".join(excinfo.value.problems)
    for expected in (
        "APP_SECRET_KEY",
        "APP_ENABLE_DEV_AUTH",
        "EMAIL_PROVIDER",
        "APP_ALLOWED_EMAIL",
        "APP_CORS_ORIGINS",
        "APP_PUBLIC_URL",
        "TRUSTED_HOSTS",
        "SLEEPER_USE_FIXTURES",
        "DATABASE_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_BUCKET",
    ):
        assert expected in problems, f"missing {expected} in: {problems}"
    # All problems surface together rather than one restart at a time.
    assert len(excinfo.value.problems) >= 10


@pytest.mark.parametrize(
    "overrides",
    [
        {"app_secret_key": "short-but-not-default"},
        {"app_allowed_email": "not-an-email"},
        {"app_allowed_email": ""},
        {"app_cors_origins": "https://ok.example.org,http://evil.example.org"},
        {"app_cors_origins": "https://*.example.org"},
        {"app_public_url": "http://app.example.org"},
        {"database_url": "sqlite+pysqlite:///:memory:"},
    ],
)
def test_individual_production_rejections(overrides):
    from src.app.config import ProductionConfigError

    with pytest.raises(ProductionConfigError):
        _settings(**overrides).validate_production()


def test_localhost_http_origin_allowed_https_required_elsewhere():
    _settings(app_cors_origins="http://localhost:5173,http://127.0.0.1:5173").validate_production()


def test_session_cookie_secure_by_environment():
    from src.app.config import Settings

    assert _settings().session_cookie_secure is True
    assert Settings(app_env="development").session_cookie_secure is False
    assert Settings(app_env="test").session_cookie_secure is False


def test_trusted_host_list_parsing():
    from src.app.config import Settings

    assert Settings(trusted_hosts="app.example.org, api.example.org").trusted_host_list == [
        "app.example.org",
        "api.example.org",
    ]
    assert Settings().trusted_host_list == ["*"]


def test_new_bounded_settings_have_defaults():
    from src.app.config import Settings

    settings = Settings()
    assert settings.assistant_max_message_chars == 4000
    assert settings.openai_request_timeout_seconds == 30.0
    assert settings.openai_max_output_tokens == 1200


def test_lifespan_startup_refuses_unsafe_production(monkeypatch):
    """Startup must fail closed, not just expose a helper method."""
    from src.app.config import ProductionConfigError, get_settings
    from src.app.factory import create_app

    unsafe = _settings(app_secret_key="dev-only-change-me")
    monkeypatch.setattr("src.app.factory.get_settings", lambda: unsafe)
    get_settings.cache_clear()

    app = create_app()
    with pytest.raises(ProductionConfigError):
        with TestClient(app):
            pass


# --------------------------------------------------------------------------
# B. CSRF must compare against the stored session token
# --------------------------------------------------------------------------


def test_csrf_missing_header_rejected(client: TestClient):
    _login(client)
    response = client.post(
        "/api/v1/sync", headers={"Idempotency-Key": "csrf-missing-1"}
    )
    assert response.status_code == 403


def test_csrf_wrong_header_rejected(client: TestClient):
    """Presence-only CSRF checks accept this; a real check must not."""
    _login(client)
    response = client.post(
        "/api/v1/sync",
        headers={"X-CSRF-Token": "attacker-invented-value", "Idempotency-Key": "csrf-wrong-1"},
    )
    assert response.status_code == 403
    assert "csrf" in json.dumps(response.json()).lower()


def test_csrf_correct_header_accepted(client: TestClient):
    csrf = _login(client)
    response = client.post(
        "/api/v1/sync",
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "csrf-ok-1"},
    )
    assert response.status_code == 200


def test_csrf_token_from_another_session_rejected(client: TestClient):
    """A CSRF token is bound to one session row, not globally valid."""
    from src.app.auth.service import AuthService
    from src.app.persistence.database import get_session
    from src.app.persistence.models import AppUser

    _login(client)
    with get_session() as session:
        user = session.query(AppUser).filter(AppUser.email == "owner@example.com").one()
        other = AuthService(session).verify_magic_link  # noqa: F841 - documented below
        # Mint a second, unrelated session row for the same user.
        from src.app.persistence.models import SessionRecord

        foreign = SessionRecord(
            user_id=user.id,
            session_hash="0" * 64,
            csrf_token="foreign-csrf-token",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        session.add(foreign)
        session.commit()

    response = client.post(
        "/api/v1/sync",
        headers={"X-CSRF-Token": "foreign-csrf-token", "Idempotency-Key": "csrf-foreign-1"},
    )
    assert response.status_code == 403


# --------------------------------------------------------------------------
# C. Session cookie flags, verify rate limiting, logout
# --------------------------------------------------------------------------


def test_session_cookie_flags_follow_settings(monkeypatch, client: TestClient):
    response = client.post("/api/v1/auth/magic-link", json={"email": "owner@example.com"})
    token = response.json()["development_link"].split("token=")[-1]
    verify = client.post("/api/v1/auth/verify", json={"token": token})
    set_cookie = verify.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert "samesite=lax" in set_cookie.lower()
    # Test env is explicitly insecure-by-design; production is not.
    assert "secure" not in set_cookie.lower()


def test_auth_verify_route_reads_secure_flag_from_settings():
    """The route must not hardcode secure=False."""
    import inspect

    from src.app.api.v1 import auth as auth_routes

    source = inspect.getsource(auth_routes)
    assert "secure=False" not in source
    assert "settings.session_cookie_secure" in source


def test_auth_verify_is_rate_limited(client: TestClient):
    from src.app.config import get_settings
    from src.app.middleware.rate_limit import limiter

    limiter.reset()
    limit = get_settings().auth_rate_limit_per_minute
    statuses = [
        client.post("/api/v1/auth/verify", json={"token": "z" * 43}).status_code
        for _ in range(limit + 2)
    ]
    assert 429 in statuses, f"magic-link verification was not rate limited: {statuses}"
    limiter.reset()


def test_logout_revokes_session_server_side(client: TestClient):
    csrf = _login(client)
    assert client.get("/api/v1/me").status_code == 200

    logout = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 200
    assert logout.json()["revoked"] is True

    # Cookie cleared client-side...
    assert client.get("/api/v1/me").status_code == 401


def test_logout_leaves_no_usable_server_side_session(client: TestClient):
    """Replaying the stolen cookie after logout must fail."""
    csrf = _login(client)
    stolen = client.cookies.get("session")
    assert stolen

    client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})

    replay = client.get("/api/v1/me", cookies={"session": stolen})
    assert replay.status_code == 401


# --------------------------------------------------------------------------
# D. Magic-link and session hardening
# --------------------------------------------------------------------------


def test_non_allowlisted_email_creates_no_token_and_looks_identical(db_session):
    from src.app.auth.service import AuthService
    from src.app.persistence.models import MagicLinkToken

    service = AuthService(db_session)
    before = db_session.query(MagicLinkToken).count()
    rejected = service.request_magic_link("attacker@example.net")
    after = db_session.query(MagicLinkToken).count()

    assert after == before, "a token row was created for a non-allowlisted email"
    assert rejected == {"status": "sent"}

    allowed = service.request_magic_link("owner@example.com")
    # No enumeration signal: the rejected response is a subset of the accepted
    # one apart from the dev-only link, and status is identical.
    assert rejected["status"] == allowed["status"]
    assert set(rejected) <= {"status", "development_link"}


def test_expired_magic_link_token_rejected(db_session):
    from src.app.auth.service import AuthService, _hash_token
    from src.app.persistence.models import MagicLinkToken

    service = AuthService(db_session)
    service.request_magic_link("owner@example.com")
    record = db_session.query(MagicLinkToken).order_by(MagicLinkToken.created_at.desc()).first()
    record.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    db_session.flush()

    raw = "expired-token-value-000000000000000000"
    record.token_hash = _hash_token(raw)
    db_session.flush()

    with pytest.raises(ValueError, match="Expired"):
        service.verify_magic_link(raw)


def test_magic_link_token_cannot_be_replayed(db_session):
    from src.app.auth.service import AuthService

    service = AuthService(db_session)
    link = service.request_magic_link("owner@example.com")["development_link"]
    token = link.split("token=")[-1]

    first = service.verify_magic_link(token)
    assert first is not None
    with pytest.raises(ValueError, match="Invalid token"):
        service.verify_magic_link(token)


def test_token_for_email_no_longer_allowlisted_is_rejected(db_session, monkeypatch):
    """Rotating APP_ALLOWED_EMAIL must invalidate outstanding links."""
    from src.app.auth.service import AuthService
    from src.app.config import Settings

    settings = Settings(app_env="test", app_enable_dev_auth=True, app_allowed_email="owner@example.com")
    service = AuthService(db_session, settings)
    token = service.request_magic_link("owner@example.com")["development_link"].split("token=")[-1]

    rotated = Settings(
        app_env="test", app_enable_dev_auth=True, app_allowed_email="new.owner@example.com"
    )
    rotated_service = AuthService(db_session, rotated)
    with pytest.raises(ValueError, match="Invalid token"):
        rotated_service.verify_magic_link(token)


def test_expired_session_row_is_removed_not_merely_ignored(db_session):
    from src.app.auth.service import AuthService, _hash_token
    from src.app.persistence.models import SessionRecord

    service = AuthService(db_session)
    token = service.request_magic_link("owner@example.com")["development_link"].split("token=")[-1]
    session_record = service.verify_magic_link(token)
    raw_session = session_record._raw_session_token

    session_record.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db_session.flush()

    assert service.get_user_for_session(raw_session) is None
    remaining = (
        db_session.query(SessionRecord)
        .filter(SessionRecord.session_hash == _hash_token(raw_session))
        .one_or_none()
    )
    assert remaining is None, "expired session row survived and could be resurrected"


def test_resend_provider_requires_api_key():
    from src.app.auth.service import EmailProviderConfigError, ResendEmailProvider, get_email_provider
    from src.app.config import Settings

    missing = Settings(app_env="production", email_provider="resend", resend_api_key=None)
    with pytest.raises(EmailProviderConfigError):
        get_email_provider(missing)

    configured = Settings(app_env="production", email_provider="resend", resend_api_key="re_test")
    provider = get_email_provider(configured)
    assert isinstance(provider, ResendEmailProvider)


def test_resend_provider_posts_to_resend_without_network(monkeypatch):
    """Resend must actually send; it must not silently fall back to dev."""
    from src.app.auth.service import ResendEmailProvider
    from src.app.config import Settings

    calls: list[dict] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _Response()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = ResendEmailProvider(
        Settings(app_env="production", email_provider="resend", resend_api_key="re_test")
    )
    assert provider.send_magic_link("owner@example.com", "https://app/x?token=abc") is None
    assert len(calls) == 1
    assert calls[0]["url"].startswith("https://api.resend.com")
    assert calls[0]["json"]["to"] == ["owner@example.com"]
    assert calls[0]["timeout"] == 10.0


# --------------------------------------------------------------------------
# E. Transport, host, correlation id, error envelope
# --------------------------------------------------------------------------


def test_trusted_host_middleware_rejects_unknown_host(monkeypatch):
    from src.app.config import Settings
    from src.app.factory import create_app
    from src.app.persistence.database import init_db

    pinned = Settings(app_env="test", app_enable_dev_auth=True, trusted_hosts="app.example.org")
    monkeypatch.setattr("src.app.factory.get_settings", lambda: pinned)
    init_db()
    with TestClient(create_app()) as pinned_client:
        assert pinned_client.get("/health/live").status_code == 400
        allowed = pinned_client.get("/health/live", headers={"Host": "app.example.org"})
        assert allowed.status_code == 200


def test_cors_is_not_a_wildcard_for_methods_and_headers():
    from src.app.factory import ALLOWED_CORS_HEADERS, ALLOWED_CORS_METHODS

    assert "*" not in ALLOWED_CORS_METHODS
    assert "*" not in ALLOWED_CORS_HEADERS
    assert set(ALLOWED_CORS_METHODS) == {"GET", "POST", "PUT", "OPTIONS"}
    assert set(ALLOWED_CORS_HEADERS) == {
        "Content-Type",
        "X-CSRF-Token",
        "X-Correlation-ID",
        "Idempotency-Key",
    }


@pytest.mark.parametrize(
    "hostile",
    [
        "not-a-uuid\r\nX-Injected: 1",
        "a" * 500,
        "<script>alert(1)</script>",
        "",
        "../../etc/passwd",
    ],
)
def test_hostile_correlation_id_is_replaced(hostile, client: TestClient):
    from src.app.factory import CORRELATION_ID_PATTERN

    response = client.get("/health/live", headers={"X-Correlation-ID": hostile})
    echoed = response.headers["X-Correlation-ID"]
    assert echoed != hostile
    assert CORRELATION_ID_PATTERN.match(echoed)


def test_wellformed_correlation_id_is_preserved(client: TestClient):
    cid = "0123456789abcdef0123456789abcdef"
    response = client.get("/health/live", headers={"X-Correlation-ID": cid})
    assert response.headers["X-Correlation-ID"] == cid


def test_readiness_failure_does_not_leak_exception_text(monkeypatch, client: TestClient):
    secret_dsn = "postgresql://fantasy:sup3r-s3cret@db.internal:5432/app"

    def boom():
        raise RuntimeError(f"could not connect to {secret_dsn}")

    monkeypatch.setattr("src.app.persistence.database.get_engine", boom)
    response = client.get("/health/ready")
    body = response.text
    assert response.status_code == 503
    assert "sup3r-s3cret" not in body
    assert "db.internal" not in body
    assert response.json()["error"]["code"] == "dependency_unavailable"
    assert response.json()["error"]["correlation_id"]


def test_unhandled_exception_returns_envelope_without_traceback(monkeypatch):
    from src.app.config import Settings
    from src.app.factory import create_app
    from src.app.persistence.database import init_db

    production_like = Settings(
        app_env="production",
        app_secret_key="y" * 48,
        app_enable_dev_auth=False,
        email_provider="smtp",
        app_allowed_email="real.owner@example.org",
        app_cors_origins="https://app.example.org",
        app_public_url="https://app.example.org",
        database_url="postgresql+psycopg://u:p@db:5432/app",
        test_database_url="sqlite+pysqlite:///:memory:?cache=shared",
        trusted_hosts="app.example.org,testserver",
    )
    monkeypatch.setattr("src.app.factory.get_settings", lambda: production_like)

    app = create_app()

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret-internal-detail-12345")

    init_db()
    with TestClient(app, raise_server_exceptions=False) as prod_client:
        response = prod_client.get("/boom")

    assert response.status_code == 500
    body = response.text
    assert "secret-internal-detail-12345" not in body
    assert "Traceback" not in body
    assert "RuntimeError" not in body
    envelope = response.json()["error"]
    assert envelope["code"] == "internal_error"
    assert envelope["correlation_id"]
    assert "debug_message" not in envelope


def test_http_exception_body_carries_error_envelope(client: TestClient):
    response = client.get("/api/v1/me")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "http_401"
    assert body["error"]["correlation_id"]
    # Legacy key retained so existing clients keep working.
    assert "detail" in body


# --------------------------------------------------------------------------
# F. Error-string leakage in routes
# --------------------------------------------------------------------------


def test_lineup_failure_returns_code_not_exception_text(monkeypatch, client: TestClient):
    _login(client)
    secret = "roster snapshot /srv/secrets/roster.json missing for owner@example.com"

    def boom(self, league_id, week, opponent_mode="current"):
        raise ValueError(secret)

    monkeypatch.setattr("src.app.decisions.services.LineupService.recommend", boom)
    response = client.get("/api/v1/leagues/fixture-standard/lineup/1")
    assert response.status_code == 400
    assert secret not in response.text
    assert response.json()["detail"]["code"] == "lineup_unavailable"


def test_waiver_failure_returns_code_not_exception_text(monkeypatch, client: TestClient):
    _login(client)
    secret = "/srv/secrets/faab.json unreadable"

    def boom(self, league_id, week, **kwargs):
        raise ValueError(secret)

    monkeypatch.setattr("src.app.decisions.services.WaiverService.recommend", boom)
    response = client.get("/api/v1/leagues/fixture-standard/waivers/1")
    assert response.status_code == 400
    assert secret not in response.text
    assert response.json()["detail"]["code"] == "waivers_unavailable"


def test_job_error_is_typed_and_redacted(client: TestClient):
    from src.app.persistence.database import get_session
    from src.app.persistence.models import JobRun

    _login(client)
    raw = "OperationalError: could not connect to postgresql://u:hunter2@db.internal/app for owner@example.com"
    with get_session() as session:
        job = JobRun(
            job_name="sync-leagues",
            correlation_id="a" * 32,
            status="failed",
            error=raw,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert raw not in response.text
    assert "hunter2" not in response.text
    assert body["error"] == "job_failed"
    assert body["error_summary"]["kind"] == "database_unavailable"


def test_job_error_debug_message_redacts_email():
    from src.app.api.v1.jobs import summarize_job_error
    from src.app.config import Settings, get_settings

    original = get_settings.__wrapped__
    try:
        get_settings.cache_clear()
        # Development is the only mode that echoes any raw text at all.
        import src.app.api.v1.jobs as jobs_module

        jobs_module.get_settings = lambda: Settings(app_env="development")
        summary = summarize_job_error("failed for owner@example.com token=abc")
        assert "owner@example.com" not in summary["debug_message"]
        assert "[REDACTED_EMAIL]" in summary["debug_message"]
    finally:
        import src.app.api.v1.jobs as jobs_module

        jobs_module.get_settings = get_settings
        get_settings.cache_clear()
        assert get_settings.__wrapped__ is original


def test_missing_projection_is_labeled_unavailable_not_fabricated(client: TestClient):
    """A player with no promoted projection must not get invented points."""
    _login(client)
    response = client.get("/api/v1/projections/players/does-not-exist-9999")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "unavailable"
    assert body["mean"] is None
    assert body["projection_run_id"] is None
    assert "12.5" not in response.text


# --------------------------------------------------------------------------
# G. Input limits
# --------------------------------------------------------------------------


def test_assistant_message_length_is_bounded(client: TestClient):
    from src.app.api.v1.assistant import MAX_MESSAGE_CHARS

    csrf = _login(client)
    headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "assistant-bounds-1"}
    oversized = client.post(
        "/api/v1/assistant/responses",
        json={"message": "x" * (MAX_MESSAGE_CHARS + 1), "league_id": "fixture-standard", "week": 1},
        headers=headers,
    )
    assert oversized.status_code == 422

    empty = client.post(
        "/api/v1/assistant/responses",
        json={"message": "", "league_id": "fixture-standard", "week": 1},
        headers={**headers, "Idempotency-Key": "assistant-bounds-2"},
    )
    assert empty.status_code == 422


@pytest.mark.parametrize("week", [0, 26, 9999, -1])
def test_assistant_week_is_bounded(week, client: TestClient):
    csrf = _login(client)
    response = client.post(
        "/api/v1/assistant/responses",
        json={"message": "lineup", "league_id": "fixture-standard", "week": week},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": f"assistant-week-{week}"},
    )
    assert response.status_code == 422


def test_assistant_league_id_length_is_bounded(client: TestClient):
    csrf = _login(client)
    response = client.post(
        "/api/v1/assistant/responses",
        json={"message": "lineup", "league_id": "L" * 500, "week": 1},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "assistant-league-1"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        # Client-asserted pick value must be refused, not trusted.
        {
            "side_a": {"roster_id": 1, "player_ids": [], "pick_assets": [{"season": 2027, "round": 1, "value": 9999}]},
            "side_b": {"roster_id": 2, "player_ids": []},
        },
        # Unknown top-level and side-level keys.
        {"side_a": {"roster_id": 1}, "side_b": {"roster_id": 2}, "sneaky": True},
        {"side_a": {"roster_id": 1, "value": 1000}, "side_b": {"roster_id": 2}},
        # Bounded list lengths.
        {
            "side_a": {"roster_id": 1, "player_ids": [f"00-{i:07d}" for i in range(13)]},
            "side_b": {"roster_id": 2},
        },
        {
            "side_a": {"roster_id": 1, "pick_assets": [{"season": 2027, "round": 1}] * 13},
            "side_b": {"roster_id": 2},
        },
        # Player id charset / length.
        {"side_a": {"roster_id": 1, "player_ids": ["../../etc/passwd"]}, "side_b": {"roster_id": 2}},
        {"side_a": {"roster_id": 1, "player_ids": ["a" * 64]}, "side_b": {"roster_id": 2}},
        # Roster id range and horizon enum.
        {"side_a": {"roster_id": 0}, "side_b": {"roster_id": 2}},
        {"side_a": {"roster_id": 1}, "side_b": {"roster_id": 2}, "horizon": "forever"},
        # Untyped dict no longer accepted.
        {"side_a": "everything", "side_b": {"roster_id": 2}},
    ],
)
def test_trade_evaluate_rejects_malformed_bodies(payload, client: TestClient):
    csrf = _login(client)
    response = client.post(
        "/api/v1/leagues/fixture-standard/trades/evaluate",
        json=payload,
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "trade-bad-1"},
    )
    assert response.status_code == 422, response.text


def test_trade_evaluate_accepts_valid_body_with_picks(client: TestClient):
    """Valid pick_assets must survive validation (dynasty league accepts picks)."""
    csrf = _login(client)
    response = client.post(
        "/api/v1/leagues/fixture-dynasty/trades/evaluate",
        json={
            "side_a": {
                "roster_id": 1,
                "player_ids": ["00-0034857"],
                "pick_assets": [{"season": 2027, "round": 1, "original_roster_id": 3}],
            },
            "side_b": {"roster_id": 2, "player_ids": ["00-0033280"], "pick_assets": []},
            "horizon": "dynasty",
        },
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "trade-good-1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["horizon"] == "dynasty"


def test_trade_domain_rejection_is_a_typed_422_not_a_500(client: TestClient):
    """A redraft league refusing picks must be a client error with a code."""
    csrf = _login(client)
    response = client.post(
        "/api/v1/leagues/fixture-standard/trades/evaluate",
        json={
            "side_a": {"roster_id": 1, "pick_assets": [{"season": 2027, "round": 1}]},
            "side_b": {"roster_id": 2, "player_ids": ["00-0033280"]},
        },
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": "trade-redraft-pick-1"},
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "trade_not_evaluable"
    assert detail["reason"] == "picks_not_tradeable_in_redraft"
    assert "dynasty league" not in response.text


# --------------------------------------------------------------------------
# I. Artifact traversal via the shared resolver (store tests live in
#    test_artifacts.py; this covers the configured-default store)
# --------------------------------------------------------------------------


def test_default_store_rejects_absolute_windows_path():
    from src.app.artifacts.store import ArtifactPathError, get_artifact_store

    store = get_artifact_store()
    with pytest.raises(ArtifactPathError):
        store.get_bytes("local://C:/Windows/System32/config/SAM")


# --------------------------------------------------------------------------
# J. Logging redaction and correlation propagation
# --------------------------------------------------------------------------


def test_log_event_redacts_secrets_and_emails():
    import structlog

    from src.app.logging import redact_processor

    buffer = io.StringIO()
    # Snapshot the whole structlog config; restoring only the processors would
    # leave this test's logger factory installed for every later test.
    saved_config = structlog.get_config().copy()
    try:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                redact_processor,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.PrintLoggerFactory(file=buffer),
            cache_logger_on_first_use=False,
        )
        structlog.get_logger("test").warning(
            "provider_call_failed",
            api_key="sk-live-should-never-appear",
            session_token="cookie-value-should-never-appear",
            authorization="Bearer nope",
            nested={"password": "hunter2", "note": "contact owner@example.com"},
            detail="failed for owner@example.com",
            safe_field="visible",
        )
        rendered = buffer.getvalue()
    finally:
        structlog.configure(**saved_config)

    for secret in (
        "sk-live-should-never-appear",
        "cookie-value-should-never-appear",
        "Bearer nope",
        "hunter2",
        "owner@example.com",
    ):
        assert secret not in rendered, f"{secret!r} leaked into logs: {rendered}"
    assert "[REDACTED]" in rendered
    assert "[REDACTED_EMAIL]" in rendered
    assert "visible" in rendered


def test_correlation_scope_propagates_and_restores():
    from src.app.logging import bind_correlation_id, correlation_scope, current_correlation_id, run_with_correlation

    outer = bind_correlation_id("f" * 32)
    with correlation_scope("b" * 32) as inner:
        assert inner == "b" * 32
        assert current_correlation_id() == "b" * 32
    assert current_correlation_id() == outer

    captured = run_with_correlation("c" * 32, current_correlation_id)
    assert captured == "c" * 32
    assert current_correlation_id() == outer
