"""Complete roster-week cohort for weekly mixture contract v2."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.projection.weekly.config.paths import DATA_DIR, SKILL_POSITIONS
from src.projection.weekly.data.nflverse_loader import load_rosters, load_rosters_weekly, load_schedules
from src.projection.weekly.evaluate.preseason import roster_week_cohort
from src.projection.weekly.draws.contracts_v2 import CONTRACT_VERSION_V2, derive_event_labels

COHORT_SCHEMA_VERSION = 2
RECOVERABLE_ROSTER_STATUSES = frozenset({"ACT", "RES", "INA", "PUP", "NFI", "SUS"})
NOT_ON_ROSTER_STATUSES = frozenset({"DEV", "CUT", "RET", "EXE", "TRC", "TRD", "E01"})


@dataclass(frozen=True)
class CohortPanelArtifact:
    path: Path
    schema_version: int
    contract_version: str
    row_count: int
    content_hash: str
    source_hashes: dict[str, str]
    cutoff_policy: str
    exclusion_summary: dict[str, Any]
    event_rates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame_full(df: pl.DataFrame) -> str:
    """Content hash over all rows (not a header/sample shortcut)."""
    buf = io.BytesIO()
    df.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _load_schedules_for_seasons(seasons: list[int], *, data_dir: Path | None = None) -> pl.DataFrame:
    data_dir = data_dir or DATA_DIR
    frames: list[pl.DataFrame] = []
    for season in seasons:
        path = data_dir / "raw" / f"schedules_{season}_{season}.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
        else:
            loaded = load_schedules([season], force=False)
            if not loaded.is_empty():
                frames.append(loaded.filter(pl.col("season") == season))
    if not frames:
        return load_schedules(seasons, force=False)
    return pl.concat(frames, how="vertical_relaxed").unique()


def _load_weekly_rosters_for_seasons(seasons: list[int], *, data_dir: Path | None = None) -> pl.DataFrame:
    data_dir = data_dir or DATA_DIR
    frames: list[pl.DataFrame] = []
    for season in seasons:
        path = data_dir / "raw" / f"rosters_weekly_{season}_{season}.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
        else:
            loaded = load_rosters_weekly([season], force=False)
            if not loaded.is_empty():
                frames.append(loaded)
    if frames:
        return pl.concat(frames, how="vertical_relaxed")
    return load_rosters_weekly(seasons, force=False)


def _load_annual_rosters_for_seasons(seasons: list[int]) -> pl.DataFrame:
    return load_rosters(seasons, force=False)


def _schedule_team_games(schedules: pl.DataFrame, season: int) -> pl.DataFrame:
    sched = schedules.filter(pl.col("season") == season)
    if "game_type" in sched.columns:
        sched = sched.filter(pl.col("game_type") == "REG")
    home = sched.select(
        ["season", "week", "game_id", pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent")]
    )
    away = sched.select(
        ["season", "week", "game_id", pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent")]
    )
    return pl.concat([home, away], how="vertical_relaxed").unique()


def _cohort_from_weekly_rosters(
    weekly_rosters: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    season: int,
) -> pl.DataFrame:
    rw = weekly_rosters.filter(pl.col("season") == season)
    if rw.is_empty():
        return pl.DataFrame()
    status_col = "status" if "status" in rw.columns else "status_description_abbr"
    rw = rw.filter(pl.col("position").is_in(SKILL_POSITIONS) & pl.col("gsis_id").is_not_null())
    if "week" not in rw.columns:
        return pl.DataFrame()
    team_games = _schedule_team_games(schedules, season)
    cohort = rw.select(
        [
            pl.col("gsis_id"),
            pl.col("season"),
            pl.col("week"),
            pl.col("team"),
            pl.col("position"),
            pl.col(status_col).cast(pl.Utf8).alias("roster_status"),
        ]
    ).join(team_games, on=["season", "week", "team"], how="left")
    return cohort.unique(subset=["gsis_id", "season", "week"], keep="first")


def _cohort_from_annual_rosters(
    rosters: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    season: int,
) -> pl.DataFrame:
    return roster_week_cohort(rosters, schedules, season=season).with_columns(
        pl.lit(None).cast(pl.Utf8).alias("roster_status")
    )


def build_complete_roster_cohort(
    panel: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """One row per recoverable rostered skill player x scheduled team-week.

    Outcomes are left-joined from ``panel``; DNPs without box-score rows remain.
    """
    seasons = seasons or sorted(int(s) for s in panel["season"].unique().to_list())
    seasons = [s for s in seasons if s >= 2016]
    schedules = _load_schedules_for_seasons(seasons, data_dir=data_dir)
    weekly = _load_weekly_rosters_for_seasons(seasons, data_dir=data_dir)
    annual = _load_annual_rosters_for_seasons(seasons)

    cohort_parts: list[pl.DataFrame] = []
    for season in seasons:
        part = _cohort_from_weekly_rosters(weekly, schedules, season=season)
        if part.is_empty():
            part = _cohort_from_annual_rosters(annual, schedules, season=season)
        if not part.is_empty():
            cohort_parts.append(part)

    if not cohort_parts:
        raise ValueError(f"no recoverable roster-week cohort for seasons={seasons}")

    cohort = pl.concat(cohort_parts, how="vertical_relaxed").unique(
        subset=["gsis_id", "season", "week"], keep="first"
    )

    # Pre-kickoff features and same-week outcomes from panel (split before inference).
    panel_sub = panel.filter(pl.col("season").is_in(seasons))
    overlap = [c for c in panel_sub.columns if c in cohort.columns and c not in {"gsis_id", "season", "week"}]
    if overlap:
        panel_sub = panel_sub.drop(overlap)
    merged = cohort.join(panel_sub, on=["gsis_id", "season", "week"], how="left")

    merged = merged.with_columns(pl.col("game_id").is_not_null().alias("has_scheduled_game"))
    merged = derive_event_labels(merged)
    return merged


def summarize_cohort_exclusions(df: pl.DataFrame) -> dict[str, Any]:
    """Row-state counts and conditional event rates by season/position."""
    out: dict[str, Any] = {
        "rows": df.height,
        "row_state_counts": {},
        "by_season": {},
        "by_position": {},
    }
    if "row_outcome_state" in df.columns:
        vc = df["row_outcome_state"].value_counts()
        out["row_state_counts"] = {
            str(r["row_outcome_state"]): int(r["count"]) for r in vc.iter_rows(named=True)
        }

    scheduled = df.filter(pl.col("has_scheduled_game"))
    out["scheduled_rows"] = scheduled.height
    out["bye_or_unscheduled"] = int((~df["has_scheduled_game"]).sum())

    def _rates(sub: pl.DataFrame) -> dict[str, Any]:
        active_denom = sub.filter(pl.col("active_label").is_not_null())
        part_denom = sub.filter(pl.col("active_label") == True)  # noqa: E712
        pos_denom = sub.filter(pl.col("participated_label") == True)  # noqa: E712
        return {
            "rows": sub.height,
            "active_denominator": active_denom.height,
            "participation_denominator": part_denom.height,
            "positive_usage_denominator": pos_denom.height,
            "observed_active_rate": (
                float(active_denom["active_label"].mean()) if active_denom.height else None
            ),
            "participation_rate_given_active": (
                float(part_denom["participated_label"].mean()) if part_denom.height else None
            ),
            "positive_usage_rate_given_participation": (
                float(pos_denom["positive_usage_label"].mean()) if pos_denom.height else None
            ),
            "fantasy_point_zero_rate": (
                float((sub["fantasy_points"].fill_null(0.0) == 0).mean()) if "fantasy_points" in sub.columns else None
            ),
            "missing_outcome_rate": float(sub["outcome_missing"].mean()) if "outcome_missing" in sub.columns else None,
        }

    out["aggregate_rates"] = _rates(scheduled)
    for season in sorted(scheduled["season"].unique().to_list()):
        out["by_season"][str(season)] = _rates(scheduled.filter(pl.col("season") == season))
    for pos in SKILL_POSITIONS:
        sub = scheduled.filter(pl.col("position") == pos)
        if not sub.is_empty():
            out["by_position"][pos] = _rates(sub)
    return out


def persist_cohort_panel(
    df: pl.DataFrame,
    output_dir: Path,
    *,
    source_hashes: dict[str, str] | None = None,
    cutoff_policy: str = "weekly roster membership as-of nflverse snapshot; schedule REG games only",
) -> CohortPanelArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "complete_roster_cohort.parquet"
    df.write_parquet(path)
    content_hash = sha256_frame_full(df)
    exclusion = summarize_cohort_exclusions(df)
    meta = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION_V2,
        "row_count": df.height,
        "content_hash": content_hash,
        "file_hash": _sha256_file(path),
        "source_hashes": source_hashes or {},
        "cutoff_policy": cutoff_policy,
        "exclusion_summary": exclusion,
    }
    (output_dir / "cohort_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return CohortPanelArtifact(
        path=path,
        schema_version=COHORT_SCHEMA_VERSION,
        contract_version=CONTRACT_VERSION_V2,
        row_count=df.height,
        content_hash=content_hash,
        source_hashes=source_hashes or {},
        cutoff_policy=cutoff_policy,
        exclusion_summary=exclusion,
        event_rates=exclusion.get("aggregate_rates", {}),
    )


def build_mixture_panel_v2(
    cohort: pl.DataFrame,
) -> pl.DataFrame:
    """Return cohort with v2 event labels; alias for explicit contract versioning."""
    if "active_label" not in cohort.columns:
        cohort = derive_event_labels(cohort)
    return cohort
