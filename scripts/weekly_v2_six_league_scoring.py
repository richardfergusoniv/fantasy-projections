#!/usr/bin/env python3
"""Score weekly-v2 component stats under the six live Sleeper league contracts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl

from src.app.league.sleeper.owner_config import load_owner_config
from src.app.league.sleeper.shadow_sync import ShadowSyncOptions, _configure_shadow_environment
from src.app.persistence.database import get_session
from src.app.persistence.models import League
from src.app.projections.weekly_league_scoring import (
    LeagueScoringContract,
    score_weekly_frame_for_leagues,
)

DEFAULT_PARQUET = Path("output/weekly_v2/season=2026/week=01/weekly_projections.parquet")
DEFAULT_OUTPUT = Path("output/weekly_v2/six_league_scoring_shadow.json")
DEFAULT_SHADOW_DB = ShadowSyncOptions(config_path=Path("config/sleeper_owner.example.json")).database_url


def _load_league_contracts(config_path: Path) -> list[LeagueScoringContract]:
    config = load_owner_config(config_path)
    contracts: list[LeagueScoringContract] = []
    with get_session() as session:
        for entry in config.leagues:
            league = (
                session.query(League).filter(League.league_id == entry.league_id).one_or_none()
            )
            if league is None or not league.raw_json:
                raise RuntimeError(
                    f"league {entry.league_id} missing from shadow database; "
                    "run sleeper-shadow-sync first"
                )
            contracts.append(
                LeagueScoringContract.from_league_json(
                    league_id=entry.league_id,
                    display_name=entry.display_name,
                    raw_json=league.raw_json,
                )
            )
    return contracts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/sleeper_owner.json"))
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--database-url", default=DEFAULT_SHADOW_DB)
    args = parser.parse_args(argv)

    if not args.parquet.exists():
        print(f"missing weekly output: {args.parquet}", file=sys.stderr)
        return 2
    if not args.config.exists():
        print(f"missing owner config: {args.config}", file=sys.stderr)
        return 2

    options = ShadowSyncOptions(config_path=args.config, database_url=args.database_url)
    _configure_shadow_environment(options)

    frame = pl.read_parquet(args.parquet)
    leagues = _load_league_contracts(args.config)
    artifact = score_weekly_frame_for_leagues(frame, leagues)
    artifact["source"] = {
        "parquet": str(args.parquet),
        "config": str(args.config),
        "database_url": args.database_url,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["validation"], indent=2))
    return 0 if all(artifact["validation"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
