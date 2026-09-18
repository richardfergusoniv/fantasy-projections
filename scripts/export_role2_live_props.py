#!/usr/bin/env python3
"""Export Role 1 weekly_props → Role 2 live_snapshots.csv (shadow measure only).

Does not promote, blend Role 3, or change APP_PROJECTION_SOURCE / sealed pointers.

Live (DB + S3 artifact store):
  export DATABASE_URL=...
  export ARTIFACT_BACKEND=s3
  export S3_ENDPOINT_URL=https://<ref>.storage.supabase.co/storage/v1/s3
  export S3_BUCKET=fantasy-app
  export S3_ACCESS_KEY_ID=...
  export S3_SECRET_ACCESS_KEY=...
  export S3_REGION=us-east-1
  uv run python scripts/export_role2_live_props.py --season 2026 --week 2

Fixture / CI (no S3):
  uv run python scripts/export_role2_live_props.py --from-fixtures --season 2026 --week 1

Then compare (Role 2 only):
  uv run python scripts/compare_shadow_vegas_props.py \\
    --board output/shadow_weekly_schedule_m3/shadow_board_role2.csv \\
    --props output/shadow_vegas_props_compare/live_snapshots.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest.props.contracts import ProviderSnapshot  # noqa: E402
from src.ingest.props.providers.base import FixturePropProvider  # noqa: E402
from src.projection.weekly_eval.live_export import (  # noqa: E402
    DEFAULT_SCHEDULE_REL,
    build_identity_map_from_records,
    build_identity_map_from_records_with_stats,
    describe_s3_env_blocker,
    export_role2_snapshot_rows,
    load_schedule_kickoffs,
    missing_live_export_env,
    write_live_snapshots_csv,
)

DEFAULT_OUT = REPO_ROOT / "output" / "shadow_vegas_props_compare" / "live_snapshots.csv"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "data" / "props" / "fixtures" / "providers"


def _assert_snapshot_slate(
    snapshots: list[ProviderSnapshot],
    *,
    season: int,
    week: int,
) -> None:
    """Fail closed when loaded snapshots disagree with CLI --season/--week."""
    bad: list[str] = []
    for snap in snapshots:
        if int(snap.season) != int(season) or int(snap.week) != int(week):
            bad.append(
                f"{snap.source}: snapshot season/week={snap.season}/{snap.week} "
                f"!= cli {season}/{week}"
            )
    if bad:
        raise ValueError(
            "snapshot season/week mismatch (refusing to gate quotes against "
            "the wrong week's kickoffs): " + "; ".join(bad)
        )


def _load_snapshots_from_fixtures(
    *,
    season: int,
    week: int,
    fixtures_dir: Path,
    sources: tuple[str, ...],
) -> list[ProviderSnapshot]:
    loaded: list[ProviderSnapshot] = []
    for source in sources:
        path = fixtures_dir / f"{source}.json"
        if not path.is_file():
            continue
        snap = FixturePropProvider(source, path).fetch(season=season, week=week)
        if snap.success:
            loaded.append(snap)
    _assert_snapshot_slate(loaded, season=season, week=week)
    return loaded


def _load_snapshots_from_json(
    paths: list[Path],
    *,
    season: int,
    week: int,
) -> list[ProviderSnapshot]:
    out: list[ProviderSnapshot] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Fixture-shaped (players[]) vs already-normalized ProviderSnapshot dict.
        if "quotes" in payload:
            out.append(ProviderSnapshot.from_dict(payload))
            continue
        source = str(payload.get("source") or path.stem)
        snap = FixturePropProvider(source, path).fetch(season=season, week=week)
        if snap.success:
            out.append(snap)
    _assert_snapshot_slate(out, season=season, week=week)
    return out


def _load_snapshots_from_db(
    *,
    season: int,
    week: int,
    sources: tuple[str, ...],
) -> tuple[list[ProviderSnapshot], Any]:
    from src.app.persistence.database import get_session
    from src.projection.weekly_props.persist import load_latest_provider_snapshots

    with get_session() as session:
        snaps = load_latest_provider_snapshots(
            session, season=season, week=week, sources=sources
        )
        # Materialize identity rows while the session is open.
        from src.app.persistence.models import PlayerIdentity

        identity_rows = session.query(PlayerIdentity).all()
        # Detach: copy into plain dicts before session closes.
        records = [
            {
                "player_id": row.player_id,
                "gsis_id": row.gsis_id,
                "sleeper_id": row.sleeper_id,
                "name": row.name,
                "position": row.position,
                "team": row.team,
            }
            for row in identity_rows
        ]
        return snaps, records


def _parse_sources(raw: str) -> tuple[str, ...]:
    parts = tuple(s.strip().lower() for s in raw.split(",") if s.strip())
    return parts or ("draftkings", "fanduel")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=2)
    parser.add_argument(
        "--sources",
        default="draftkings,fanduel",
        help="Comma-separated provider names (default: draftkings,fanduel).",
    )
    parser.add_argument(
        "--mode",
        choices=("consensus", "single"),
        default="consensus",
        help="consensus merges books; single keeps a source column per book.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"CSV path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=REPO_ROOT / DEFAULT_SCHEDULE_REL,
        help="NFL schedule CSV used for kickoff_at (not closing line).",
    )
    parser.add_argument(
        "--from-fixtures",
        action="store_true",
        help="Load data/props/fixtures/providers/*.json (no DB/S3).",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
    )
    parser.add_argument(
        "--snapshot-json",
        action="append",
        type=Path,
        default=[],
        help="Provider snapshot JSON path (repeatable). Offline alternative to DB+S3.",
    )
    parser.add_argument(
        "--identity-json",
        type=Path,
        default=None,
        help="Optional JSON list of {player_id,gsis_id,name,team,position} records.",
    )
    parser.add_argument(
        "--allow-non-gsis",
        action="store_true",
        help="Keep unresolved / non-gsis player ids (default: drop; sealed board needs gsis).",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Print missing live-export env vars and exit 0/1.",
    )
    args = parser.parse_args(argv)
    sources = _parse_sources(args.sources)

    if args.check_env:
        missing = missing_live_export_env()
        if missing:
            print("missing:", ", ".join(missing))
            print(describe_s3_env_blocker())
            return 1
        print("live export env looks complete")
        return 0

    identity_map: dict[str, dict] = {}
    snapshots: list[ProviderSnapshot] = []

    if args.from_fixtures:
        try:
            snapshots = _load_snapshots_from_fixtures(
                season=args.season,
                week=args.week,
                fixtures_dir=args.fixtures_dir,
                sources=sources,
            )
        except ValueError as exc:
            print(f"BLOCKER: {exc}", file=sys.stderr)
            return 1
    elif args.snapshot_json:
        try:
            snapshots = _load_snapshots_from_json(
                list(args.snapshot_json),
                season=args.season,
                week=args.week,
            )
        except ValueError as exc:
            print(f"BLOCKER: {exc}", file=sys.stderr)
            return 1
    else:
        missing = missing_live_export_env()
        if missing:
            print("BLOCKER: live DB+S3 export env incomplete.", file=sys.stderr)
            print("missing:", ", ".join(missing), file=sys.stderr)
            print(describe_s3_env_blocker(), file=sys.stderr)
            print(
                "\nUse --from-fixtures or --snapshot-json for an offline export.",
                file=sys.stderr,
            )
            return 2
        try:
            snapshots, identity_records = _load_snapshots_from_db(
                season=args.season, week=args.week, sources=sources
            )
        except Exception as exc:  # noqa: BLE001
            print(f"BLOCKER: failed to load source_snapshot / artifacts: {exc}", file=sys.stderr)
            print(describe_s3_env_blocker(), file=sys.stderr)
            return 2
        identity_map, ambiguous = build_identity_map_from_records_with_stats(
            identity_records
        )
        if ambiguous:
            print(
                f"warning: dropped {ambiguous} ambiguous identity key(s)",
                file=sys.stderr,
            )

    if args.identity_json is not None:
        payload = json.loads(args.identity_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        patch, patch_ambiguous = build_identity_map_from_records_with_stats(payload)
        if patch_ambiguous:
            print(
                f"warning: identity-json dropped {patch_ambiguous} ambiguous key(s)",
                file=sys.stderr,
            )
        # Merge patch into DB (or empty) map — do not replace.
        identity_map.update(patch)

    if not snapshots:
        print(
            f"No provider snapshots loaded for season={args.season} week={args.week} "
            f"sources={sources}",
            file=sys.stderr,
        )
        return 1

    kickoffs = load_schedule_kickoffs(
        args.schedule, season=args.season, week=args.week
    )
    rows = export_role2_snapshot_rows(
        snapshots,
        identity_map=identity_map,
        kickoffs_by_team=kickoffs,
        mode=args.mode,
        require_gsis=not args.allow_non_gsis,
    )
    if not rows:
        print(
            "Export produced 0 rows (check identity resolution, schedule kickoffs, "
            "and as_of <= kickoff_at).",
            file=sys.stderr,
        )
        return 1

    out = write_live_snapshots_csv(rows, args.output)
    n_players = len({r["player_id"] for r in rows})
    n_markets = len({r["market"] for r in rows})
    print(
        f"wrote {out} rows={len(rows)} players={n_players} markets={n_markets} "
        f"mode={args.mode} season={args.season} week={args.week}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
