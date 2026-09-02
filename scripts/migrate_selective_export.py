#!/usr/bin/env python3
"""Selective export of durable domain state for blue-green migration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLES_PRESERVE = [
    "app_user",
    "league",
    "league_member",
    "league_rule_snapshot",
    "league_draft_rule",
    "roster_snapshot",
    "matchup_snapshot",
    "player_identity",
    "trade_proposal",
    "traded_pick",
    "league_transaction",
    "injury_evidence",
    "availability_event",
    "projection_run",
    "player_projection",
    "simulation_partition",
    "promotion_event",
    "active_projection_pointer",
    "release_pointer",
    "release_pointer_history",
    "status_overlay_pointer",
    "status_overlay_pointer_history",
    "source_snapshot",
]

TABLES_EXCLUDE = [
    "magic_link_token",
    "session_record",
    "job_run",
    "job_lease",
    "job_outbox",
    "rate_limit_bucket",
    "assistant_audit",
]


def export_rows(session, table: str) -> list[dict]:
    from sqlalchemy import text

    rows = session.execute(text(f"SELECT * FROM {table}")).mappings().all()
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uri-map", type=Path, help="old_uri -> new_uri JSON map")
    args = parser.parse_args()

    from src.app.persistence.database import get_session

    uri_map = {}
    if args.uri_map and args.uri_map.is_file():
        uri_map = json.loads(args.uri_map.read_text(encoding="utf-8"))

    payload: dict = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "tables": {},
        "excluded_tables": TABLES_EXCLUDE,
    }
    with get_session() as session:
        for table in TABLES_PRESERVE:
            try:
                payload["tables"][table] = export_rows(session, table)
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                payload["tables"][table] = {"error": str(exc)}

    if uri_map:
        text_blob = json.dumps(payload)
        for old, new in uri_map.items():
            text_blob = text_blob.replace(old, new)
        payload = json.loads(text_blob)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "table_counts": {k: len(v) if isinstance(v, list) else 0 for k, v in payload["tables"].items()}}, indent=2))


if __name__ == "__main__":
    main()
