#!/usr/bin/env python3
"""Generate weekly fantasy projections."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import polars as pl

from src.projection.weekly.config.paths import MODELS_DIR, TRAIN_END_SEASON, TRAIN_START_SEASON, VALIDATE_SEASON
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.evaluate.metrics import evaluate_projections, format_report
from src.projection.weekly.features.panel import load_panel
from src.projection.weekly.pipeline.rookie_projector import project_week_with_rookies
from src.projection.weekly.pipeline.season_projector import build_outlook_panel
from src.projection.weekly.pipeline.veteran_projector import project_veterans_week, write_projections


def _models_root(season: int) -> Path:
    season_dir = MODELS_DIR / f"season={season}"
    return season_dir if season_dir.exists() else MODELS_DIR


def _load_models(season: int) -> tuple[object, dict, dict, dict | None]:
    root = _models_root(season)
    team = joblib.load(root / "team_totals.joblib")
    volume = {pos: joblib.load(root / f"volume_{pos}.joblib") for pos in ("QB", "RB", "WR", "TE")}
    efficiency = {pos: joblib.load(root / f"efficiency_{pos}.joblib") for pos in ("QB", "RB", "WR", "TE")}
    rookie = {}
    for pos in ("QB", "RB", "WR", "TE"):
        path = root / f"rookie_{pos}.joblib"
        if path.exists():
            rookie[pos] = joblib.load(path)
    return team, volume, efficiency, rookie or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project fantasy points for a week")
    parser.add_argument("--season", type=int, default=VALIDATE_SEASON)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--scoring", default="ppr", help="ppr | half_ppr | standard")
    parser.add_argument("--no-rookies", action="store_true", help="Veteran pipeline only")
    parser.add_argument("--evaluate", action="store_true", help="Score against actuals if present")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scoring = ScoringConfig.from_name(args.scoring)
    panel = load_panel()
    train_seasons = list(range(TRAIN_START_SEASON, min(TRAIN_END_SEASON, args.season - 1) + 1))
    week_rows = panel.filter((pl.col("season") == args.season) & (pl.col("week") == args.week))
    if week_rows.is_empty():
        history = panel.filter(pl.col("season") < args.season)
        outlook = build_outlook_panel(history, target_season=args.season)
        panel = pl.concat([history, outlook], how="diagonal_relaxed")

    team, volume, efficiency, rookie = _load_models(args.season)
    common = dict(
        season=args.season,
        week=args.week,
        scoring=scoring,
        train_seasons=train_seasons,
        team_totals_model=team,
        volume_models=volume,
        efficiency_models=efficiency,
    )

    if args.no_rookies:
        proj = project_veterans_week(panel, **common)
    else:
        proj = project_week_with_rookies(panel, rookie_models=rookie, **common)

    path = write_projections(proj)
    print(f"Wrote {path}")
    preview_cols = [
        c
        for c in ("player_name", "position", "team", "fantasy_points", "floor", "ceiling")
        if c in proj.columns
    ]
    # Avoid Windows cp1252 crashes on Polars box-drawing characters
    print(proj.select(preview_cols).head(20).write_csv())

    if args.evaluate:
        actuals = panel.filter((pl.col("season") == args.season) & (pl.col("week") == args.week))
        report = evaluate_projections(proj, actuals)
        print("Evaluation:")
        print(format_report(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
