"""Authoritative preseason evaluation harness for tuning, training, and promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.projection.weekly.config.paths import DATA_DIR, TRAIN_START_SEASON
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.data.nflverse_loader import (
    load_depth_charts,
    load_rosters_weekly,
    load_schedules,
)
from src.projection.weekly.evaluate.preseason import (
    PromotionPolicy,
    assert_strict_preseason_asof,
    complete_roster_week_outcomes,
    evaluate_complete_preseason,
    file_fingerprint,
    promotion_gate,
    roster_week_cohort,
)
from src.projection.weekly.features.panel import load_panel
from src.projection.weekly.models.calibration import (
    apply_position_calibration,
    fit_position_calibration,
)
from src.projection.weekly.models.efficiency import train_efficiency_models
from src.projection.weekly.models.rookie import train_rookie_model
from src.projection.weekly.models.team_totals import train_team_totals
from src.projection.weekly.models.volume import train_volume_models
from src.projection.weekly.models.volume_config import VolumeModelConfig
from src.projection.weekly.pipeline.rookie_projector import project_week_with_rookies
from src.projection.weekly.pipeline.season_projector import build_outlook_panel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreseasonEvalConfig:
    """Full configuration fingerprint for a preseason evaluation run."""

    panel_path: Path
    outer_start: int = 2022
    outer_end: int = 2025
    scoring: str = "half_ppr"
    volume_options: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)

    def fingerprint(self, *, code_revision: str | None = None) -> str:
        payload = {
            "panel": file_fingerprint(self.panel_path),
            "outer_start": self.outer_start,
            "outer_end": self.outer_end,
            "scoring": self.scoring,
            "volume_options": self.volume_options,
            "random_seed": self.random_seed,
            "promotion_policy": asdict(self.promotion_policy),
            "code_revision": code_revision,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def evaluate_season(
    panel: pl.DataFrame,
    season: int,
    scoring: ScoringConfig,
    *,
    return_oof: bool = False,
    volume_options: dict[str, Any] | VolumeModelConfig | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pl.DataFrame]:
    """Leave-one-season-out preseason evaluation for a single target season."""
    if isinstance(volume_options, VolumeModelConfig):
        volume_kwargs = volume_options.to_options()
    else:
        volume_kwargs = dict(volume_options or {})

    history = panel.filter(pl.col("season") < season)
    assert_strict_preseason_asof(history, season=season)
    train_seasons = list(range(TRAIN_START_SEASON, season))
    team = train_team_totals(history, train_seasons=train_seasons, persist=False)
    volume = train_volume_models(
        history,
        train_seasons=train_seasons,
        persist=False,
        **volume_kwargs,
    )
    efficiency = train_efficiency_models(history, train_seasons=train_seasons, persist=False)
    try:
        rookie = train_rookie_model(history, train_seasons=train_seasons, persist=False)
    except Exception:
        rookie = None

    weekly_rosters = load_rosters_weekly([season])
    preseason_rosters = weekly_rosters.filter(
        (pl.col("season") == season)
        & (pl.col("week") == pl.col("week").min())
        & (
            pl.col("status").is_in(["ACT", "RES", "INA", "PUP", "NFI", "SUS"])
            | pl.col("status").is_null()
        )
    )
    depth = load_depth_charts([season])
    if not depth.is_empty() and "week" in depth.columns:
        depth = depth.filter(pl.col("week") == pl.col("week").min())
    outlook = build_outlook_panel(
        history,
        target_season=season,
        force_rosters=False,
        roster_data=preseason_rosters,
        depth_data=depth,
    )
    work = pl.concat([history, outlook], how="diagonal_relaxed")
    frames = []
    for week in sorted(outlook["week"].unique().to_list()):
        frames.append(
            project_week_with_rookies(
                work,
                season=season,
                week=int(week),
                scoring=scoring,
                train_seasons=train_seasons,
                team_totals_model=team,
                volume_models=volume,
                efficiency_models=efficiency,
                rookie_models=rookie,
            )
        )
    projections = pl.concat(frames, how="diagonal_relaxed")
    schedules = load_schedules([season])
    cohort = roster_week_cohort(preseason_rosters, schedules, season=season)
    actual = complete_roster_week_outcomes(cohort, panel.filter(pl.col("season") == season))
    report = evaluate_complete_preseason(projections, actual)
    prior_ppg = (
        panel.filter(pl.col("season") == season - 1)
        .group_by("gsis_id")
        .agg(pl.col("fantasy_points").mean().alias("fantasy_points"))
    )
    baseline = actual.select(["gsis_id", "season", "week"]).join(
        prior_ppg, on="gsis_id", how="left"
    ).with_columns(pl.col("fantasy_points").fill_null(0.0))
    report["baseline"] = evaluate_complete_preseason(baseline, actual)
    report["season"] = season
    report["train_seasons"] = train_seasons
    if not return_oof:
        return report
    actual_cols = ["gsis_id", "season", "week"]
    if "position" in actual.columns:
        actual_cols.append("position")
    oof = actual.select(
        actual_cols + [pl.col("fantasy_points").alias("actual_fantasy_points")]
    ).join(
        projections.select(
            ["gsis_id", "season", "week"]
            + [
                pl.col("fantasy_points").alias("projected_fantasy_points"),
                *(
                    [pl.col("floor").alias("projected_floor")]
                    if "floor" in projections.columns
                    else []
                ),
                *(
                    [pl.col("ceiling").alias("projected_ceiling")]
                    if "ceiling" in projections.columns
                    else []
                ),
            ]
        ),
        on=["gsis_id", "season", "week"],
        how="left",
    )
    return report, oof


def run_preseason_backtest(
    panel: pl.DataFrame,
    *,
    config: PreseasonEvalConfig,
    scoring: ScoringConfig | None = None,
) -> dict[str, Any]:
    """Run leave-one-season-out evaluation with nested calibration."""
    scoring = scoring or ScoringConfig.from_name(config.scoring)
    reports: list[dict[str, Any]] = []
    calibrated_reports: list[dict[str, Any]] = []
    oof_frames: list[pl.DataFrame] = []
    warmup_seasons: list[int] = []

    for year in range(config.outer_start, config.outer_end + 1):
        report, rows = evaluate_season(
            panel,
            year,
            scoring,
            return_oof=True,
            volume_options=config.volume_options,
        )
        reports.append(report)
        if oof_frames:
            prior_oof = pl.concat(oof_frames, how="diagonal_relaxed")
            calibration = fit_position_calibration(prior_oof)
            calibrated = apply_position_calibration(
                rows, calibration, point_col="projected_fantasy_points"
            )
            pred = calibrated.select(
                [
                    "gsis_id",
                    "season",
                    "week",
                    pl.col("projected_fantasy_points").alias("fantasy_points"),
                    "floor",
                    "ceiling",
                ]
            )
            actual = rows.select(
                [
                    "gsis_id",
                    "season",
                    "week",
                    "position",
                    pl.col("actual_fantasy_points").alias("fantasy_points"),
                ]
            )
            calibrated_report = evaluate_complete_preseason(pred, actual)
            calibrated_report.update(
                {
                    "season": year,
                    "train_seasons": report["train_seasons"],
                    "calibration_train_seasons": calibration["trained_seasons"],
                    "baseline": report["baseline"],
                }
            )
            calibrated_reports.append(calibrated_report)
        else:
            warmup_seasons.append(year)
        oof_frames.append(rows)

    return {
        "config_fingerprint": config.fingerprint(),
        "volume_options": config.volume_options,
        "seasons": reports,
        "calibrated_seasons": calibrated_reports,
        "warmup_seasons": warmup_seasons,
        "raw_promotion": promotion_gate(reports, policy=config.promotion_policy),
        "promotion": promotion_gate(calibrated_reports, policy=config.promotion_policy),
        "oof": pl.concat(oof_frames, how="diagonal_relaxed") if oof_frames else None,
    }


def write_preseason_backtest(
    result: dict[str, Any],
    output_path: Path,
    *,
    write_oof: bool = True,
) -> Path:
    """Persist backtest JSON and optional OOF parquet."""
    oof = result.pop("oof", None)
    payload = {k: v for k, v in result.items() if k != "oof"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if write_oof and oof is not None and not oof.is_empty():
        oof_path = output_path.parent / "preseason_oof.parquet"
        oof.write_parquet(oof_path)
    return output_path


def default_panel_path() -> Path:
    return DATA_DIR / "processed" / "player_week_panel.parquet"


def load_panel_for_eval(panel_path: Path | None = None) -> pl.DataFrame:
  path = panel_path or default_panel_path()
  if path == default_panel_path():
      return load_panel()
  return pl.read_parquet(path)
