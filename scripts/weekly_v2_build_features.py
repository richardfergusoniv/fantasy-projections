#!/usr/bin/env python3
"""Build the player-week feature panel."""

from __future__ import annotations

import argparse
import logging
import sys

from src.projection.weekly.config.paths import TRAIN_START_SEASON, VALIDATE_SEASON
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.features.panel import build_player_week_panel, save_panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build player-week feature panel")
    parser.add_argument("--start", type=int, default=TRAIN_START_SEASON)
    parser.add_argument("--end", type=int, default=VALIDATE_SEASON)
    parser.add_argument("--scoring", default="ppr", help="ppr | half_ppr | standard")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    scoring = ScoringConfig.from_name(args.scoring)
    seasons = list(range(args.start, args.end + 1))
    panel = build_player_week_panel(seasons=seasons, scoring=scoring, force_reload=args.force)
    path = save_panel(panel)
    print(f"Panel saved to {path} ({panel.height} rows, {len(panel.columns)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
