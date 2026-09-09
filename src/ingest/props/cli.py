"""CLI for weekly player-prop ingest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest.props.service import default_fixture_providers, run_ingest
from src.ingest.props.snapshot import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = REPO_ROOT / "data" / "props" / "fixtures" / "providers"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest weekly NFL player props")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="Directory of provider fixture JSON files",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=None,
        help="Local snapshot root (fixtures/regression only)",
    )
    args = parser.parse_args(argv)
    providers = default_fixture_providers(args.fixtures_dir)
    if not providers:
        raise SystemExit(f"no provider fixtures found under {args.fixtures_dir}")
    store = SnapshotStore(args.snapshot_root) if args.snapshot_root else SnapshotStore()
    result = run_ingest(
        season=args.season,
        week=args.week,
        providers=providers,
        store=store,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
