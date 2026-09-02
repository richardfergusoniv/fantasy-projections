"""Preseason backtesting, coverage accounting, and model promotion gates.

These helpers deliberately keep governance separate from the serving pipeline.
They make the evaluation cohort explicit and treat a rostered player without a
box-score row as a zero, rather than silently dropping that player in an inner
join.
"""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from src.projection.weekly.config.paths import SKILL_POSITIONS
from src.projection.weekly.evaluate.metrics import evaluate_projections


KEYS = ("gsis_id", "season", "week")


def assert_strict_preseason_asof(panel: pl.DataFrame, *, season: int) -> None:
    """Reject any training/feature history from the season being evaluated."""
    if "season" not in panel.columns:
        raise ValueError("panel must contain season")
    bad = panel.filter(pl.col("season") >= season)
    if not bad.is_empty():
        years = sorted(int(x) for x in bad["season"].unique().to_list())
        raise ValueError(f"preseason {season} history contains non-as-of seasons: {years}")


def roster_week_cohort(
    rosters: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    season: int,
) -> pl.DataFrame:
    """Build the recoverable roster-week evaluation universe.

    Annual rosters are crossed with each team's regular-season schedule. When
    weekly roster snapshots exist, a row is only eligible from its roster week
    onward. This is intentionally broader than the player-stats table so DNPs
    remain in the denominator.
    """
    required = {"gsis_id", "team"}
    if not required.issubset(rosters.columns):
        raise ValueError(f"rosters missing {sorted(required - set(rosters.columns))}")
    sched = schedules.filter(pl.col("season") == season)
    if "game_type" in sched.columns:
        sched = sched.filter(pl.col("game_type") == "REG")
    home = sched.select(["season", "week", pl.col("home_team").alias("team")])
    away = sched.select(["season", "week", pl.col("away_team").alias("team")])
    team_weeks = pl.concat([home, away], how="vertical_relaxed").unique()

    roster = rosters
    if "season" in roster.columns:
        roster = roster.filter(pl.col("season") == season)
    if "position" in roster.columns:
        roster = roster.filter(pl.col("position").is_in(SKILL_POSITIONS))
    if "status" in roster.columns:
        # Camp cuts, retired/traded rows, and practice-squad/development rows
        # are not part of the recoverable game-day cohort. RES/INA remain so
        # injuries and DNPs are scored as zero rather than disappearing.
        roster = roster.filter(
            pl.col("status").is_in(["ACT", "RES", "INA", "PUP", "NFI", "SUS"])
            | pl.col("status").is_null()
        )
    keep = [c for c in ("gsis_id", "team", "position", "player_name", "week") if c in roster.columns]
    roster = roster.select(keep).filter(pl.col("gsis_id").is_not_null()).unique()
    # nflverse seasonal rosters commonly expose one terminal `week` value per
    # player; it is not a weekly snapshot. Only apply activation semantics when
    # the source actually contains repeated player/team snapshots.
    roster_week_col = None
    if "week" in roster.columns:
        pairs = roster.select(["gsis_id", "team"]).unique().height
        snapshots = roster.select(["gsis_id", "team", "week"]).unique().height
        if snapshots > pairs:
            roster_week_col = "week"
    if roster_week_col:
        roster = roster.rename({"week": "_roster_week"})
    elif "week" in roster.columns:
        # nflverse's annual roster table often carries one terminal week value
        # per player.  It is metadata about the source snapshot, not the game
        # week to evaluate.  Leaving it in place shadows the schedule's week
        # during the team join and collapses a player-season to one pseudo-week.
        roster = roster.drop("week")
    cohort = roster.join(team_weeks, on="team", how="inner")
    if roster_week_col:
        cohort = cohort.filter(
            pl.col("_roster_week").is_null() | (pl.col("_roster_week") <= pl.col("week"))
        ).drop("_roster_week")
    return cohort.unique(subset=list(KEYS), keep="first")


def complete_roster_week_outcomes(
    cohort: pl.DataFrame,
    actuals: pl.DataFrame,
    *,
    outcome_col: str = "fantasy_points",
) -> pl.DataFrame:
    """Left-join outcomes to a roster-week cohort and recover DNPs as zero."""
    missing = [k for k in KEYS if k not in cohort.columns or k not in actuals.columns]
    if missing:
        raise ValueError(f"cohort/actuals missing keys: {sorted(set(missing))}")
    act = actuals.select(list(KEYS) + [outcome_col]).group_by(list(KEYS)).agg(
        pl.col(outcome_col).sum().alias(outcome_col)
    )
    return cohort.join(act, on=list(KEYS), how="left").with_columns(
        [
            pl.col(outcome_col).fill_null(0.0).alias(outcome_col),
            pl.col(outcome_col).is_not_null().alias("has_boxscore"),
        ]
    )


def evaluate_complete_preseason(
    projections: pl.DataFrame,
    outcomes: pl.DataFrame,
) -> dict[str, Any]:
    """Evaluate while penalizing missing projections and reporting coverage."""
    base_cols = list(KEYS) + (["position"] if "position" in outcomes.columns else [])
    projection_cols = [c for c in projections.columns if c != "position" or "position" not in base_cols]
    pred = outcomes.select(base_cols).join(projections.select(projection_cols), on=list(KEYS), how="left")
    if "fantasy_points" not in pred.columns:
        pred = pred.with_columns(pl.lit(None).cast(pl.Float64).alias("fantasy_points"))
    covered = pred["fantasy_points"].is_not_null().sum()
    total = pred.height
    # Missing projections are zero for scoring, but coverage remains explicit.
    pred = pred.with_columns(pl.col("fantasy_points").fill_null(0.0))
    report = evaluate_projections(pred, outcomes)
    report["coverage"] = covered / total if total else 0.0
    report["cohort_n"] = total
    report["missing_projection_n"] = total - covered
    return report


@dataclass(frozen=True)
class PromotionPolicy:
    min_coverage: float = 0.95
    min_mae_improvement: float = 0.02
    min_rank_improvement: float = 0.0
    min_dispersion_ratio: float = 0.70
    max_dispersion_ratio: float = 1.30
    min_seasons: int = 3
    min_interval_coverage: float = 0.72
    max_interval_coverage: float = 0.90


def promotion_gate(
    season_reports: list[dict[str, Any]],
    *,
    policy: PromotionPolicy | None = None,
) -> dict[str, Any]:
    """Return a machine-readable promotion decision against named baselines."""
    policy = policy or PromotionPolicy()
    failures: list[str] = []
    if len(season_reports) < policy.min_seasons:
        failures.append(f"requires {policy.min_seasons} seasons, got {len(season_reports)}")
    for report in season_reports:
        year = report.get("season", "unknown")
        baseline = report.get("baseline") or {}
        mae = report.get("mae")
        base_mae = baseline.get("mae")
        rank = report.get("rank_corr")
        base_rank = baseline.get("rank_corr")
        coverage = float(report.get("coverage") or 0.0)
        dispersion = report.get("dispersion_ratio")
        interval = report.get("interval") or {}
        if coverage < policy.min_coverage:
            failures.append(f"{year}: coverage {coverage:.3f} < {policy.min_coverage:.3f}")
        if mae is None or base_mae is None or base_mae <= 0:
            failures.append(f"{year}: missing MAE baseline")
        elif (base_mae - mae) / base_mae < policy.min_mae_improvement:
            failures.append(f"{year}: MAE improvement below threshold")
        if rank is None or base_rank is None or rank - base_rank < policy.min_rank_improvement:
            failures.append(f"{year}: rank improvement below threshold")
        if dispersion is None or not policy.min_dispersion_ratio <= dispersion <= policy.max_dispersion_ratio:
            failures.append(f"{year}: dispersion outside policy")
        interval_coverage = interval.get("coverage")
        if interval_coverage is not None and not (
            policy.min_interval_coverage
            <= interval_coverage
            <= policy.max_interval_coverage
        ):
            failures.append(f"{year}: interval coverage outside policy")
    return {"promote": not failures, "failures": failures, "policy": asdict(policy)}


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Stable freshness metadata without hashing an entire large parquet file."""
    stat = path.stat()
    token = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fingerprint": hashlib.sha256(token).hexdigest(),
    }


def write_freshness_manifest(
    path: Path,
    *,
    train_seasons: list[int],
    data_files: list[Path],
    artifacts: list[Path],
) -> Path:
    """Persist the exact training window and filesystem inputs/outputs."""
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "train_seasons": sorted(int(x) for x in train_seasons),
        "data": [file_fingerprint(p) for p in data_files if p.exists()],
        "artifacts": [file_fingerprint(p) for p in artifacts if p.exists()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
