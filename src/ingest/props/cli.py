"""CLI for weekly player-prop ingest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest.props.service import build_providers, run_ingest
from src.ingest.props.snapshot import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO_ROOT / "data" / "props" / "fixtures" / "providers"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest weekly NFL player props")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("live", "fixture"),
        default="live",
        help="live DraftKings/FanDuel fetch (default) or local fixtures",
    )
    parser.add_argument(
        "--providers",
        default="draftkings,fanduel",
        help="Comma-separated provider names",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="Directory of provider fixture JSON files (fixture mode)",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=None,
        help="Local snapshot root (fixtures/regression only)",
    )
    args = parser.parse_args(argv)
    providers = build_providers(
        mode=args.mode,
        provider_names=args.providers,
        fixtures_dir=args.fixtures_dir,
    )
    if not providers:
        raise SystemExit("no providers configured")
    store = SnapshotStore(args.snapshot_root) if args.snapshot_root else SnapshotStore()
    result = run_ingest(
        season=args.season,
        week=args.week,
        providers=providers,
        store=store,
        mode=args.mode,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
