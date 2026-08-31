"""CLI entrypoints for app operations."""

from __future__ import annotations

import argparse
import subprocess
import sys

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
