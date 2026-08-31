"""Environment-driven application configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET_KEY = "dev-only-change-me"
DEFAULT_ALLOWED_EMAIL = "owner@example.com"
MIN_PRODUCTION_SECRET_LENGTH = 32
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})


class ProductionConfigError(RuntimeError):
    """Raised when production configuration is unsafe.

    Carries the full list of problems so an operator can fix every one of them
    in a single pass instead of rediscovering them one restart at a time.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        joined = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"Refusing to start in production with {len(self.problems)} unsafe setting(s):\n{joined}"
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_secret_key: str = Field(
        default=DEFAULT_SECRET_KEY,
        min_length=16,
        description="Session signing secret",
    )
    app_timezone: str = "America/Los_Angeles"
    app_allowed_email: str = DEFAULT_ALLOWED_EMAIL
    app_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    app_enable_dev_auth: bool = False
    trusted_hosts: str = "*"

    database_url: str = "postgresql+psycopg://fantasy:fantasy@localhost:5432/fantasy_app"
    test_database_url: str = "sqlite+pysqlite:///:memory:?cache=shared"

    artifact_backend: Literal["local", "s3"] = "local"
    artifact_local_root: str = "output/app_artifacts"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "fantasy-app"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "us-east-1"

    sleeper_username: str | None = None
    sleeper_user_id: str | None = None
    #: Explicit override for the Sleeper data source. ``None`` means "derive
    #: from APP_ENV". Making it settable means an operator can run the real
    #: read-only API from a staging environment, and — more importantly — that
    #: `operations/status` can report which source is actually in use rather
    #: than leaving fixture data indistinguishable from live data.
    sleeper_use_fixtures: bool | None = None

    email_provider: Literal["development", "smtp", "resend"] = "development"
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    resend_api_key: str | None = None
    email_from: str = "noreply@example.com"
    app_public_url: str = "http://localhost:5173"

    openai_api_key: str | None = None
    openai_cost_sensitive_model: str = "gpt-4.1-mini"
    openai_balanced_model: str = "gpt-4.1"
    openai_monthly_soft_limit_usd: float = 45.0
    openai_monthly_hard_limit_usd: float = 55.0
    openai_monthly_warning_usd: float = 30.0
    openai_request_timeout_seconds: float = 30.0
    openai_max_output_tokens: int = 1200

    auth_rate_limit_per_minute: int = 10
    assistant_rate_limit_per_minute: int = 20
    assistant_max_message_chars: int = 4000

    simulation_draw_count: int = 10_000
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("sleeper_use_fixtures", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat a blank env value as "not set" rather than a parse error.

        `.env` files routinely carry `KEY=` for an optional setting; refusing to
        start on that is a worse failure than falling back to the APP_ENV rule.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("app_cors_origins")
    @classmethod
    def _split_origins(cls, value: str) -> str:
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.app_cors_origins.split(",") if origin.strip()]

    @property
    def trusted_host_list(self) -> list[str]:
        hosts = [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]
        return hosts or ["*"]

    @property
    def sqlalchemy_url(self) -> str:
        if self.app_env == "test":
            return self.test_database_url
        return self.database_url

    @property
    def use_sleeper_fixtures(self) -> bool:
        """True when Sleeper reads come from recorded fixtures, not the API."""
        if self.sleeper_use_fixtures is not None:
            return self.sleeper_use_fixtures
        return self.app_env != "production"

    @property
    def sleeper_mode(self) -> str:
        return "fixture" if self.use_sleeper_fixtures else "live"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def session_cookie_secure(self) -> bool:
        """Session cookies are Secure everywhere except local dev and tests."""
        return self.app_env not in {"development", "test"}

    def production_config_problems(self) -> list[str]:
        """Return every unsafe production setting. Empty list means safe."""
        if self.app_env != "production":
            return []
        problems: list[str] = []

        if self.app_secret_key == DEFAULT_SECRET_KEY:
            problems.append("APP_SECRET_KEY is still the built-in development default")
        elif len(self.app_secret_key) < MIN_PRODUCTION_SECRET_LENGTH:
            problems.append(
                f"APP_SECRET_KEY must be at least {MIN_PRODUCTION_SECRET_LENGTH} characters "
                f"in production (got {len(self.app_secret_key)})"
            )

        if self.app_enable_dev_auth:
            problems.append("APP_ENABLE_DEV_AUTH must be false in production")

        if self.email_provider == "development":
            problems.append(
                "EMAIL_PROVIDER='development' echoes magic links to API responses; "
                "use 'smtp' or 'resend' in production"
            )

        allowed_email = (self.app_allowed_email or "").strip()
        if allowed_email == DEFAULT_ALLOWED_EMAIL:
            problems.append("APP_ALLOWED_EMAIL is still the built-in default owner@example.com")
        elif not allowed_email:
            problems.append("APP_ALLOWED_EMAIL must be set in production")
        elif "@" not in allowed_email:
            problems.append("APP_ALLOWED_EMAIL must be a real email address")

        for origin in self.cors_origin_list:
            if "*" in origin:
                problems.append(f"APP_CORS_ORIGINS may not contain a wildcard in production: {origin!r}")
                continue
            if origin.startswith("http://"):
                host = urlparse(origin).hostname or ""
                if host not in LOCAL_HOSTS:
                    problems.append(
                        f"APP_CORS_ORIGINS must use https:// in production (got {origin!r})"
                    )

        if not self.app_public_url.startswith("https://"):
            problems.append("APP_PUBLIC_URL must start with https:// in production")

        if "sqlite" in self.database_url.lower():
            problems.append("DATABASE_URL must not point at SQLite in production")

        if self.use_sleeper_fixtures:
            problems.append(
                "SLEEPER_USE_FIXTURES must be false in production; recorded fixtures "
                "would be published as if they were live league data"
            )

        if "*" in self.trusted_host_list:
            problems.append(
                "TRUSTED_HOSTS must name the deployment's hostnames in production, not '*'"
            )

        if self.artifact_backend == "s3":
            for name, value in (
                ("S3_ACCESS_KEY_ID", self.s3_access_key_id),
                ("S3_SECRET_ACCESS_KEY", self.s3_secret_access_key),
                ("S3_BUCKET", self.s3_bucket),
            ):
                if not value:
                    problems.append(f"{name} is required when ARTIFACT_BACKEND='s3'")

        return problems

    def validate_production(self) -> None:
        """Fail closed when running in production with unsafe settings."""
        problems = self.production_config_problems()
        if problems:
            raise ProductionConfigError(problems)


@lru_cache
def get_settings() -> Settings:
    return Settings()
