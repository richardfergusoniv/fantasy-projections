#!/usr/bin/env python3
"""Honest holdout evaluation: retrain on prior seasons, project eval season."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import polars as pl

from src.projection.weekly.config.paths import (
    OUTPUTS_DIR,
    TRAIN_START_SEASON,
    VALIDATE_SEASON,
    ensure_dirs,
)
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


def _train_holdout_models(panel: pl.DataFrame, train_seasons: list[int], *, skip_rookie: bool):
    logging.info("Retraining models on seasons %s–%s", train_seasons[0], train_seasons[-1])
    team_model = train_team_totals(
        panel, train_seasons=train_seasons, model_type="ridge", persist=False
    )
    volume_models = train_volume_models(
        panel, train_seasons=train_seasons, model_type="hgb", persist=False
    )
    efficiency_models = train_efficiency_models(
        panel, train_seasons=train_seasons, model_type="ridge", persist=False
    )
    rookie_models = None
    if not skip_rookie:
        try:
            rookie_models = train_rookie_model(
                panel, train_seasons=train_seasons, model_type="ridge", persist=False
            )
        except Exception as exc:
            logging.warning("Rookie train skipped: %s", exc)
            rookie_models = None
    return team_model, volume_models, efficiency_models, rookie_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Honest holdout evaluation")
    parser.add_argument("--season", type=int, default=VALIDATE_SEASON)
    parser.add_argument("--max-week", type=int, default=18)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--skip-rookie", action="store_true")
    parser.add_argument("--skip-retrain", action="store_true", help="Use disk models (legacy)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scoring = ScoringConfig.from_name(args.scoring)
    panel = load_panel()
    train_seasons = list(range(TRAIN_START_SEASON, args.season))
    if not train_seasons:
        print(f"No train seasons before eval season {args.season}")
        return 1
    if args.season in train_seasons:
        raise AssertionError(f"Eval season {args.season} must be excluded from train_seasons")

    team_model = volume_models = efficiency_models = rookie_models = None
    if not args.skip_retrain:
        team_model, volume_models, efficiency_models, rookie_models = _train_holdout_models(
            panel, train_seasons, skip_rookie=args.skip_rookie
        )

    weeks = (
        panel.filter(pl.col("season") == args.season)["week"].unique().sort().to_list()
    )
    weeks = [w for w in weeks if w <= args.max_week]

    frames = []
    for week in weeks:
        try:
            proj = project_week_with_rookies(
                panel,
                season=args.season,
                week=week,
                scoring=scoring,
                train_seasons=train_seasons,
                team_totals_model=team_model,
                volume_models=volume_models,
                efficiency_models=efficiency_models,
                rookie_models=rookie_models,
            )
            frames.append(proj)
            print(f"Projected {args.season} week {week}: {proj.height} players")
        except Exception as exc:
            logging.warning("Week %s failed: %s", week, exc)

    if not frames:
        print("No projections produced.")
        return 1

    all_proj = pl.concat(frames, how="diagonal_relaxed")
    actuals = panel.filter(pl.col("season") == args.season)
    report = evaluate_projections(all_proj, actuals)
    report["train_seasons"] = train_seasons
    report["eval_season"] = args.season

    last5 = build_last5_baseline(panel, season=args.season)
    if not last5.is_empty():
        report["baseline_last5"] = evaluate_projections(last5, actuals)
    prior = build_prior_season_ppg_baseline(panel, season=args.season)
    if not prior.is_empty():
        report["baseline_prior_ppg"] = evaluate_projections(prior, actuals)

    print("\n=== Holdout Evaluation ===")
    print(format_report(report))

    ensure_dirs()
    out = OUTPUTS_DIR / f"eval_{args.season}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved report to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
