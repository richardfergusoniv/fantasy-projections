#!/usr/bin/env python3
"""Update Supabase Vault production_app_url for pg_cron callbacks."""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text

_VAULT_SQL = text(
    """
    SELECT vault.update_secret(
      (SELECT id FROM vault.secrets WHERE name = 'production_app_url'),
      :secret,
      'production_app_url',
      'Canonical production URL'
    )
    """
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: update_production_app_url.py <https://...>", file=sys.stderr)
        return 2

    public_url = sys.argv[1].strip()
    if not public_url.startswith("https://"):
        print("URL must start with https://", file=sys.stderr)
        return 2

    db_url = os.environ.get("MIGRATION_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        print("MIGRATION_DATABASE_URL or DATABASE_URL required", file=sys.stderr)
        return 2

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(_VAULT_SQL, {"secret": public_url})

    print(f"Updated production_app_url -> {public_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
