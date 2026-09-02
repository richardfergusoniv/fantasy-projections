"""Start a disposable, seeded API for the PWA end-to-end tests.

The browser tests need a real API and a real database, not a mocked client, so
this builds one from scratch on every run: a fresh SQLite file, migrated to
head, seeded with the six fixture leagues, then served by uvicorn.

It is deliberately isolated from any database an operator is using locally —
the file lives under ``web/.e2e/`` and is deleted and rebuilt each time — so
running the browser suite can never touch real data.

Usage: ``uv run python scripts/e2e_api.py [--port 8000]``
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT / "web" / ".e2e"


def _prepare_environment(port: int) -> None:
    """Set configuration before anything imports (and caches) the settings."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    db_path = WORKDIR / "e2e.db"
    if db_path.exists():
        db_path.unlink()
    artifacts = WORKDIR / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)

    web_port = os.environ.get("E2E_WEB_PORT", "5173")
    web_origin = f"http://127.0.0.1:{web_port}"

    os.environ.update(
        {
            "APP_ENV": "development",
            "APP_SECRET_KEY": "e2e-only-secret-key-not-used-anywhere-else",
            "APP_ALLOWED_EMAIL": "owner@example.com",
            "APP_ENABLE_DEV_AUTH": "true",
            "APP_PUBLIC_URL": web_origin,
            "APP_CORS_ORIGINS": f"{web_origin},http://localhost:{web_port}",
            "TRUSTED_HOSTS": "*",
            "DATABASE_URL": f"sqlite+pysqlite:///{db_path.as_posix()}",
            "ARTIFACT_BACKEND": "local",
            "ARTIFACT_LOCAL_ROOT": str(artifacts),
            "EMAIL_PROVIDER": "development",
            "LOG_JSON": "false",
            "SLEEPER_USE_FIXTURES": "true",
            "INJURY_RESEARCH_MODE": "fixture",
            # No API key: the browser journey must exercise the deterministic
            # assistant, and CI must never make a paid external call.
            "OPENAI_API_KEY": "",
            "AUTH_RATE_LIMIT_PER_MINUTE": "1000",
            "E2E_API_PORT": str(port),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _prepare_environment(args.port)

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")

    from src.app.persistence.database import get_session
    from src.app.seed import seed_development_data

    with get_session() as session:
        seeded = seed_development_data(session, email="owner@example.com")
    print(f"e2e database seeded with {len(seeded['leagues'])} leagues", flush=True)

    import uvicorn

    uvicorn.run("src.app.main:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
