"""Veteran weekly projection orchestrator."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.projection.weekly.config.paths import MODELS_DIR, OUTPUTS_DIR, ensure_dirs
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.features.injuries import apply_injury_haircut
from src.projection.weekly.models.efficiency import predict_efficiency
from src.projection.weekly.models.calibration import (
    apply_position_calibration,
    load_calibration_for_season,
)
from src.projection.weekly.models.team_totals import build_team_week_labels, predict_team_totals
from src.projection.weekly.models.volume import predict_volume
from src.projection.weekly.pipeline.accounting import apply_accounting
from src.projection.weekly.scoring.fantasy_points import compute_fantasy_points

logger = logging.getLogger(__name__)


def _calibration_path(season: int) -> Path:
    for candidate in (
        MODELS_DIR / f"season={season}" / "calibration.json",
        MODELS_DIR / "calibration.json",
    ):
        if candidate.exists():
            return candidate
    return MODELS_DIR / "calibration.json"


def _residual_bands(panel: pl.DataFrame, train_seasons: list[int]) -> dict[str, tuple[float, float]]:
    """Floor/ceiling offsets from per-player residual percentiles by position.

    Residuals are (week FP − player season mean) so bands reflect within-player
    variance rather than cross-player point dispersion.
    """
    bands: dict[str, tuple[float, float]] = {}
    hist = panel.filter(pl.col("season").is_in(train_seasons))
    if hist.is_empty() or "fantasy_points" not in hist.columns:
        return {pos: (-5.0, 8.0) for pos in ("QB", "RB", "WR", "TE")}

    for pos in ("QB", "RB", "WR", "TE"):
        sub = hist.filter(pl.col("position") == pos)
        if sub.height < 50 or "gsis_id" not in sub.columns:
            bands[pos] = (-5.0, 8.0)
            continue
        with_mean = sub.with_columns(
            pl.col("fantasy_points").mean().over(["gsis_id", "season"]).alias("_player_mean")
        )
        resid = (
            with_mean["fantasy_points"] - with_mean["_player_mean"]
        ).drop_nulls().to_numpy()
        if len(resid) < 50:
            bands[pos] = (-5.0, 8.0)
            continue
        bands[pos] = (float(np.nanpercentile(resid, 10)), float(np.nanpercentile(resid, 90)))
    return bands


def project_veterans_week(
    panel: pl.DataFrame,
    *,
    season: int,
    week: int,
    scoring: ScoringConfig | None = None,
    train_seasons: list[int] | None = None,
    team_totals_model=None,
    volume_models: dict | None = None,
    efficiency_models: dict | None = None,
) -> pl.DataFrame:
    """Generate veteran projections for a single season-week."""
    scoring = scoring or ScoringConfig()
    train_seasons = train_seasons or list(range(2016, season))
    if season in train_seasons:
        raise ValueError(
            f"Eval/project season {season} must be excluded from train_seasons={train_seasons}"
        )

    # Feature rows for this week (lagged features already on panel)
    week_df = panel.filter((pl.col("season") == season) & (pl.col("week") == week))
    if week_df.is_empty():
        raise ValueError(f"No panel rows for season={season} week={week}")

    from src.projection.weekly.features.depth import clip_depth_rank_for_models

    week_df = clip_depth_rank_for_models(week_df)
    # Team totals from context
    team_ctx = build_team_week_labels(week_df)
    hist_labels = build_team_week_labels(
        panel.filter(pl.col("season").is_in(train_seasons))
    )
    team_preds = predict_team_totals(
        team_ctx, model=team_totals_model, history=hist_labels
    )

    # Volume + efficiency
    with_vol = predict_volume(week_df, models=volume_models)
    with_eff = predict_efficiency(with_vol, models=efficiency_models)
    # Scale predicted shares by play_prob before accounting so teammates absorb Out volume
    with_eff = apply_injury_haircut(with_eff, mode="shares")

    # Accounting -> box scores
    accounted = apply_accounting(with_eff, team_preds)
    scored = compute_fantasy_points(accounted, scoring, alias="fantasy_points")

    calibration = load_calibration_for_season(
        _calibration_path(season), target_season=season
    )
    if calibration is not None:
        scored = apply_position_calibration(scored, calibration)
    else:
        bands = _residual_bands(panel, train_seasons)
        floor_vals = []
        ceil_vals = []
        for pos, fp in zip(
            scored["position"].to_list(), scored["fantasy_points"].to_list()
        ):
            lo, hi = bands.get(pos, (-5.0, 8.0))
            floor_vals.append(max(0.0, float(fp) + lo))
            ceil_vals.append(max(0.0, float(fp) + hi))

        scored = scored.with_columns(
            [
                pl.Series("floor", floor_vals),
                pl.Series("ceiling", ceil_vals),
            ]
        )

    # Hard-zero Out / play_prob ~ 0 (works even before volume models are retrained)
    scored = apply_injury_haircut(scored, mode="stats")

    keep = [
        c
        for c in (
            "season",
            "week",
            "gsis_id",
            "player_name",
            "position",
            "team",
            "is_rookie",
            "attempts",
            "completions",
            "passing_yards",
            "passing_tds",
            "interceptions",
            "carries",
            "rushing_yards",
            "rushing_tds",
            "targets",
            "receptions",
            "receiving_yards",
            "receiving_tds",
            "fantasy_points",
            "fantasy_points_raw",
            "floor",
            "ceiling",
            "pred_target_share",
            "pred_carry_share",
            "pred_dropback_share",
            "depth_rank",
            "is_out",
            "is_doubtful",
            "is_questionable",
            "play_prob",
            "injury_status",
            # Season aggregation needs the empirical-Bayes availability
            # estimate that was attached to the outlook feature rows.
            "projected_games_estimate",
        )
        if c in scored.columns
    ]
    return scored.select(keep).sort(["position", "fantasy_points"], descending=[False, True])


def write_projections(df: pl.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    if path is None:
        season = df["season"][0]
        week = df["week"][0]
        path = OUTPUTS_DIR / f"projections_{season}_w{week:02d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    logger.info("Wrote projections -> %s (%d players)", path, df.height)
    return path
