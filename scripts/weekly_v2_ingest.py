#!/usr/bin/env python3
"""Download and cache nflverse tables."""

from __future__ import annotations

import argparse
import logging
import sys

from src.projection.weekly.config.paths import TRAIN_START_SEASON, VALIDATE_SEASON, ensure_dirs
from src.projection.weekly.data.espn_injuries import fetch_espn_injuries
from src.projection.weekly.data.nflverse_loader import ingest_all
from src.projection.weekly.data.sleeper import fetch_sleeper_players


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest nflverse + ESPN injury data")
    parser.add_argument("--start", type=int, default=TRAIN_START_SEASON)
    parser.add_argument("--end", type=int, default=VALIDATE_SEASON)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()
    seasons = list(range(args.start, args.end + 1))
    tables = ingest_all(seasons, force=args.force)
    for name, df in tables.items():
        print(f"  {name}: {df.height} rows, {len(df.columns)} cols")

    injuries = fetch_espn_injuries(force=args.force)
    print(f"  espn_injuries: {injuries.height} rows")
    sleeper = fetch_sleeper_players(force=args.force)
    print(f"  sleeper_players: {sleeper.height} rows")
    print("Ingest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
