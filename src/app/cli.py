"""CLI entrypoints for app operations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.app.config import get_settings
from src.app.persistence.database import get_session, init_db
from src.app.seed import seed_development_data


def _migrate(_: argparse.Namespace) -> None:
    result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=False)
    if result.returncode != 0:
        init_db()
        print("alembic unavailable or failed; fell back to create_all")


def _seed(args: argparse.Namespace) -> None:
    settings = get_settings()
    _migrate(args)
    with get_session() as session:
        result = seed_development_data(session, email=args.email or settings.app_allowed_email)
    print(result)


def _api(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("src.app.main:app", host=args.host, port=args.port, reload=get_settings().is_development)


def _sleeper_shadow_sync(args: argparse.Namespace) -> None:
    from src.app.league.sleeper.shadow_sync import ShadowSyncOptions, run_shadow_sync

    options = ShadowSyncOptions(
        config_path=Path(args.config),
        season=args.season,
        database_url=args.database_url,
        artifact_root=args.artifact_root,
        report_path=Path(args.report),
        allow_production_database=args.allow_production_database,
        inject_failure=args.inject_failure,
        skip_second_run=args.skip_second_run,
    )
    raise SystemExit(run_shadow_sync(options))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fantasy decision app commands")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser("migrate", help="Run database migrations")
    migrate.set_defaults(func=_migrate)

    seed = sub.add_parser("seed", help="Seed fixture development data")
    seed.add_argument("--email", default=None)
    seed.set_defaults(func=_seed)

    api = sub.add_parser("api", help="Run API server")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.set_defaults(func=_api)

    shadow = sub.add_parser(
        "sleeper-shadow-sync",
        help="Opt-in live Sleeper read-only shadow sync in an isolated database",
    )
    shadow.add_argument("--config", required=True, help="Path to owner league config JSON")
    shadow.add_argument("--season", type=int, default=None)
    shadow.add_argument(
        "--database-url",
        default=f"sqlite+pysqlite:///output/live_shadow/shadow_app.db",
        help="Shadow database URL (never the production database)",
    )
    shadow.add_argument(
        "--artifact-root",
        default="output/live_shadow/artifacts",
        help="Isolated artifact prefix",
    )
    shadow.add_argument(
        "--report",
        default="output/live_shadow/sleeper_sync_report.json",
        help="Machine-readable report path",
    )
    shadow.add_argument(
        "--allow-production-database",
        action="store_true",
        help="Explicit acknowledgement to run against a production-looking database URL",
    )
    shadow.add_argument(
        "--inject-failure",
        action="store_true",
        help="After sync, inject a publication failure and verify the active pointer is unchanged",
    )
    shadow.add_argument(
        "--skip-second-run",
        action="store_true",
        help="Skip the idempotent second sync (for faster debugging)",
    )
    shadow.set_defaults(func=_sleeper_shadow_sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
