#!/usr/bin/env python3
"""Selective import of durable domain state with acceptance gate checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize_row(row: dict) -> dict:
    normalized: dict = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value)
        else:
            normalized[key] = value
    return normalized


def _upsert_row(session, table: str, row: dict) -> None:
    from sqlalchemy import text

    row = _normalize_row(row)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f":{key}" for key in row.keys())
    conflict_col = "id" if "id" in row else ("player_id" if table == "player_identity" else None)
    if conflict_col:
        updates = ", ".join(
            f"{key} = EXCLUDED.{key}" for key in row.keys() if key != conflict_col
        )
        if updates:
            sql = (
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_col}) DO UPDATE SET {updates}"
            )
        else:
            sql = (
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_col}) DO NOTHING"
            )
    else:
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    session.execute(text(sql), row)


def _seed_owner_draft_rules(session, config_path: Path) -> int:
    if not config_path.is_file():
        return 0
    from src.app.league.sleeper.owner_config import load_owner_config
    from src.app.league.sleeper.sync import SleeperSyncService

    owner = load_owner_config(config_path)
    sync = SleeperSyncService(session)
    seeded = 0
    for entry in owner.leagues:
        if entry.league_type == "dynasty" and entry.rookie_pick_rule:
            sync.persist_owner_confirmed_draft_rule(entry.league_id, entry.rookie_pick_rule)
            seeded += 1
    return seeded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-leagues", type=int, default=6)
    parser.add_argument("--expected-dynasty-rules", type=int, default=4)
    parser.add_argument(
        "--owner-config",
        type=Path,
        default=ROOT / "config" / "sleeper_owner.json",
        help="Seed dynasty rookie-pick rules from owner config before gate checks",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Skip row import; only seed draft rules (if configured) and run gate checks",
    )
    parser.add_argument("--require-uri-map", type=Path, help="Fail if artifact URIs remain unmapped")
    args = parser.parse_args()

    from sqlalchemy import text

    from src.app.persistence.database import get_session

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.require_uri_map:
        uri_map = json.loads(args.require_uri_map.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        unresolved = [old for old in uri_map if old in blob and uri_map[old] not in blob]
        if unresolved:
            raise SystemExit(f"unresolved artifact URIs: {unresolved[:5]}")

    imported: dict[str, int] = {}
    with get_session() as session:
        if not args.gate_only:
            for table, rows in payload.get("tables", {}).items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    _upsert_row(session, table, row)
                imported[table] = len(rows)
            session.commit()

        seeded_rules = _seed_owner_draft_rules(session, args.owner_config)
        if seeded_rules:
            session.commit()

        league_count = session.execute(text("SELECT COUNT(*) FROM league")).scalar_one()
        evidence_count = session.execute(text("SELECT COUNT(*) FROM injury_evidence")).scalar_one()
        roster_count = session.execute(text("SELECT COUNT(*) FROM roster_snapshot")).scalar_one()
        unresolved_players = session.execute(
            text(
                "SELECT COUNT(*) FROM player_identity "
                "WHERE sleeper_id IS NULL OR sleeper_id = ''"
            )
        ).scalar_one()
        dynasty_rules = session.execute(
            text("SELECT COUNT(*) FROM league_draft_rule")
        ).scalar_one()
        scoring_hashes = session.execute(
            text("SELECT COUNT(DISTINCT contract_hash) FROM league_rule_snapshot")
        ).scalar_one()

    gate = {
        "league_count": league_count,
        "league_count_ok": league_count == args.expected_leagues,
        "dynasty_rules": dynasty_rules,
        "dynasty_rules_ok": dynasty_rules >= args.expected_dynasty_rules,
        "evidence_count": evidence_count,
        "roster_count": roster_count,
        "unresolved_player_identities": unresolved_players,
        "player_identities_ok": unresolved_players == 0,
        "scoring_contract_hashes": scoring_hashes,
        "seeded_draft_rules": seeded_rules,
        "imported_tables": imported,
    }
    passed = all(
        gate[key]
        for key in (
            "league_count_ok",
            "dynasty_rules_ok",
            "player_identities_ok",
        )
    )
    print(json.dumps({"gate": gate, "passed": passed}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
