#!/usr/bin/env python3
"""Walk-forward holdout vs last-5 and prior-season PPG baselines.

Retrains once per holdout season (persist=False), scores weekly + season-level
metrics, and writes outputs/walkforward_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import polars as pl

from src.projection.weekly.config.paths import OUTPUTS_DIR, TRAIN_START_SEASON, ensure_dirs
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.evaluate.metrics import (
    build_last5_baseline,
    build_prior_season_ppg_baseline,
    evaluate_projections,
    format_report,
)
from src.projection.weekly.features.panel import load_panel
from src.projection.weekly.models.efficiency import train_efficiency_models
from src.projection.weekly.models.rookie import train_rookie_model
from src.projection.weekly.models.team_totals import train_team_totals
from src.projection.weekly.models.volume import train_volume_models
from src.projection.weekly.pipeline.rookie_projector import project_week_with_rookies


def _train(panel: pl.DataFrame, train_seasons: list[int], *, skip_rookie: bool):
    team = train_team_totals(panel, train_seasons=train_seasons, persist=False)
    volume = train_volume_models(panel, train_seasons=train_seasons, persist=False)
    efficiency = train_efficiency_models(panel, train_seasons=train_seasons, persist=False)
    rookie = None
    if not skip_rookie:
        try:
            rookie = train_rookie_model(panel, train_seasons=train_seasons, persist=False)
        except Exception as exc:
            logging.warning("Rookie train skipped: %s", exc)
    return team, volume, efficiency, rookie


def eval_season(
    panel: pl.DataFrame,
    season: int,
    *,
    max_week: int,
    scoring: ScoringConfig,
    skip_rookie: bool,
) -> dict:
    train_seasons = list(range(TRAIN_START_SEASON, season))
    if not train_seasons:
        raise ValueError(f"No train seasons before {season}")
    team, volume, efficiency, rookie = _train(panel, train_seasons, skip_rookie=skip_rookie)

    weeks = (
        panel.filter(pl.col("season") == season)["week"].unique().sort().to_list()
    )
    weeks = [w for w in weeks if w <= max_week]
    frames = []
    for week in weeks:
        try:
            frames.append(
                project_week_with_rookies(
                    panel,
                    season=season,
                    week=week,
                    scoring=scoring,
                    train_seasons=train_seasons,
                    team_totals_model=team,
                    volume_models=volume,
                    efficiency_models=efficiency,
                    rookie_models=rookie,
                )
            )
        except Exception as exc:
            logging.warning("%s w%s failed: %s", season, week, exc)
    if not frames:
        return {"season": season, "error": "no projections"}

    all_proj = pl.concat(frames, how="diagonal_relaxed")
    actuals = panel.filter(pl.col("season") == season)
    report = evaluate_projections(all_proj, actuals)
    report["train_seasons"] = train_seasons
    report["eval_season"] = season

    last5 = build_last5_baseline(panel, season=season)
    if not last5.is_empty():
        report["baseline_last5"] = evaluate_projections(last5, actuals)
    prior = build_prior_season_ppg_baseline(panel, season=season)
    if not prior.is_empty():
        report["baseline_prior_ppg"] = evaluate_projections(prior, actuals)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--max-week", type=int, default=18)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--skip-rookie", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    panel = load_panel()
    scoring = ScoringConfig.from_name(args.scoring)
    summary: dict = {"seasons": {}}
    for season in range(args.start, args.end + 1):
        print(f"\n=== Walk-forward {season} ===")
        report = eval_season(
            panel,
            season,
            max_week=args.max_week,
            scoring=scoring,
            skip_rookie=args.skip_rookie,
        )
        print(format_report(report))
        summary["seasons"][str(season)] = {
            "mae": report.get("mae"),
            "rank_corr": report.get("rank_corr"),
            "dispersion_ratio": report.get("dispersion_ratio"),
            "share_mae": report.get("share_mae"),
            "season_level": report.get("season_level"),
            "baseline_last5": {
                "mae": (report.get("baseline_last5") or {}).get("mae"),
                "rank_corr": (report.get("baseline_last5") or {}).get("rank_corr"),
            },
            "baseline_prior_ppg": {
                "mae": (report.get("baseline_prior_ppg") or {}).get("mae"),
                "rank_corr": (report.get("baseline_prior_ppg") or {}).get("rank_corr"),
            },
            "by_position": report.get("by_position"),
        }

    ensure_dirs()
    out = OUTPUTS_DIR / "walkforward_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
