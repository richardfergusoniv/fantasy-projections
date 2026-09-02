#!/usr/bin/env python3
"""Project a full season (or preseason outlook) for draft boards."""

from __future__ import annotations

import argparse
import logging
import sys

from src.projection.weekly.config.paths import TRAIN_END_SEASON, TRAIN_START_SEASON
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.features.panel import load_panel
from src.projection.weekly.pipeline.season_projector import project_season, write_season_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Season fantasy projections for draft")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--scoring",
        default="half_ppr",
        help="ppr | half_ppr | standard (draft board uses half_ppr)",
    )
    parser.add_argument(
        "--projected-games",
        type=int,
        default=None,
        help="Override player-specific availability estimates with a fixed game count",
    )
    parser.add_argument("--train-end", type=int, default=None, help="Last train season (inclusive)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scoring = ScoringConfig.from_name(args.scoring)
    panel = load_panel()
    train_end = args.train_end if args.train_end is not None else min(TRAIN_END_SEASON, args.season - 1)
    train_seasons = list(range(TRAIN_START_SEASON, train_end + 1))

    weekly, season_df = project_season(
        panel,
        season=args.season,
        scoring=scoring,
        train_seasons=train_seasons,
        projected_games=args.projected_games,
    )
    paths = write_season_outputs(season_df, weekly, season=args.season)

    preview = [
        c
        for c in (
            "player_name",
            "position",
            "team",
            "fantasy_pts",
            "fantasy_pts_season",
            "projected_games",
            "source",
        )
        if c in season_df.columns
    ]
    print(f"Season {args.season}: {season_df.height} players")
    print(season_df.select(preview).head(25).write_csv())
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
