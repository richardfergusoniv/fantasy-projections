"""Effective depth charts: roles + availability (injury/roster aware).

Combines nflverse depth snapshots with roster status and ESPN injuries to
produce ``effective_depth_rank`` / ``role_slot`` used by accounting and the
draft board — separate from raw chart ranks.
"""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from typing import Literal

import polars as pl

from src.projection.weekly.data.espn_injuries import fetch_espn_injuries, injury_status_flags
from src.projection.weekly.data.ids import coerce_id_columns
from src.projection.weekly.data.nflverse_loader import load_ff_playerids
from src.projection.weekly.data.teams import normalize_team_column
from src.projection.weekly.features.injuries import prepare_espn_injury_features

logger = logging.getLogger(__name__)

Horizon = Literal["season", "weekly"]

SKILL_ABB = {
    "QB",
    "RB",
    "HB",
    "WR",
    "TE",
    "LWR",
    "RWR",
    "SWR",
    "SLWR",
    "SRWR",
}
# Explicitly excluded from skill role ranks (package / ST bleed)
EXCLUDED_ABB = {"FB", "PR", "KR", "PS", "H", "LS"}

ROSTER_UNAVAILABLE = {"SUS", "PUP", "NFI", "RET", "CUT", "RES"}

EFF_DEPTH_COLS = (
    "effective_depth_rank",
    "raw_depth_rank",
    "role_slot",
    "is_effective_starter",
    "available",
    "availability_reason",
)

_EMPTY_SCHEMA = {
    "gsis_id": pl.Utf8,
    "team": pl.Utf8,
    "position": pl.Utf8,
    "effective_depth_rank": pl.Float64,
    "raw_depth_rank": pl.Float64,
    "role_slot": pl.Utf8,
    "is_effective_starter": pl.Int8,
    "available": pl.Boolean,
    "availability_reason": pl.Utf8,
    "horizon": pl.Utf8,
    "as_of_dt": pl.Utf8,
    "player_name": pl.Utf8,
    "roster_status": pl.Utf8,
    "injury_status": pl.Utf8,
}


def _map_skill_position(pos_abb: str | None) -> str | None:
    if pos_abb is None:
        return None
    p = str(pos_abb).strip().upper()
    if p in EXCLUDED_ABB or p not in SKILL_ABB:
        return None
    if p in {"HB"}:
        return "RB"
    if p in {"LWR", "RWR", "SWR", "SLWR", "SRWR"}:
        return "WR"
    return p


def _latest_snapshot(depth: pl.DataFrame) -> pl.DataFrame:
    if depth.is_empty():
        return depth
    df = depth
    if "team" in df.columns:
        df = normalize_team_column(df, "team")
    if "dt" not in df.columns:
        return df
    if df.schema.get("dt") in (pl.Utf8, pl.String):
        df = df.with_columns(
            pl.col("dt")
            .str.replace(r"Z$", "+00:00")
            .str.to_datetime(time_zone="UTC", strict=False)
            .dt.replace_time_zone(None)
            .alias("dt")
        )
    else:
        df = df.with_columns(pl.col("dt").cast(pl.Datetime, strict=False))
    dt_max = df["dt"].max()
    return df.filter(pl.col("dt") == dt_max)


def _skill_depth_rows(depth: pl.DataFrame) -> pl.DataFrame:
    """One raw skill row per gsis_id from latest offense package (no FB/ST)."""
    df = _latest_snapshot(depth)
    empty = pl.DataFrame(
        schema={
            "gsis_id": pl.Utf8,
            "team": pl.Utf8,
            "position": pl.Utf8,
            "raw_depth_rank": pl.Float64,
            "player_name": pl.Utf8,
            "as_of_dt": pl.Utf8,
            "_package_only": pl.Boolean,
        }
    )
    if df.is_empty():
        return empty

    if "pos_grp" in df.columns:
        grp = pl.col("pos_grp").str.to_lowercase()
        df = df.filter(
            pl.col("pos_grp").is_null()
            | grp.str.contains("off")
            | (
                (~grp.str.contains("base"))
                & (~grp.str.contains("special"))
                & (~grp.str.contains(" d"))
            )
        )

    abb_col = "pos_abb" if "pos_abb" in df.columns else None
    rank_col = "pos_rank" if "pos_rank" in df.columns else ("depth_team" if "depth_team" in df.columns else None)
    if abb_col is None or rank_col is None or "gsis_id" not in df.columns:
        return empty

    df = df.with_columns(
        [
            pl.col("gsis_id").cast(pl.Utf8),
            pl.col(abb_col).cast(pl.Utf8).str.to_uppercase().alias("_abb"),
            pl.col(rank_col).cast(pl.Float64).alias("raw_depth_rank"),
        ]
    ).filter(pl.col("gsis_id").is_not_null() & pl.col("_abb").is_not_null())

    # Skill fantasy positions; FB tagged package-only (excluded from ranks, blocked in accounting)
    positions = []
    package_only = []
    for a in df["_abb"].to_list():
        mapped = _map_skill_position(a)
        if mapped is not None:
            positions.append(mapped)
            package_only.append(False)
        elif str(a).upper() == "FB":
            positions.append("RB")  # roster often maps FB→RB; mark package-only
            package_only.append(True)
        else:
            positions.append(None)
            package_only.append(True)

    df = df.with_columns(
        [
            pl.Series("position", positions, dtype=pl.Utf8),
            pl.Series("_package_only", package_only),
        ]
    ).filter(pl.col("position").is_not_null())
    if df.is_empty():
        return empty

    # Prefer true skill rows over FB package when both exist
    df = df.sort(["gsis_id", "_package_only", "raw_depth_rank"]).unique(
        subset=["gsis_id"], keep="first"
    )

    name_expr = (
        pl.col("player_name").cast(pl.Utf8)
        if "player_name" in df.columns
        else pl.lit(None).cast(pl.Utf8)
    )
    as_of = (
        pl.col("dt").cast(pl.Utf8)
        if "dt" in df.columns
        else pl.lit(None).cast(pl.Utf8)
    )
    return df.select(
        [
            "gsis_id",
            pl.col("team").cast(pl.Utf8) if "team" in df.columns else pl.lit(None).cast(pl.Utf8).alias("team"),
            "position",
            "raw_depth_rank",
            name_expr.alias("player_name"),
            as_of.alias("as_of_dt"),
            "_package_only",
        ]
    )


def _injury_long_term(status: str | None) -> bool:
    if not status:
        return False
    s = str(status).lower()
    return (
        "injured reserve" in s
        or s.strip() == "ir"
        or "suspend" in s
        or "pup" in s
        or "nfi" in s
    )


def _injury_short_term_out(status: str | None, *, is_out: bool) -> bool:
    """Out for the week but not IR/suspend (season horizon ignores these)."""
    if not is_out:
        return False
    return not _injury_long_term(status)


def _availability(
    *,
    horizon: Horizon,
    roster_status: str | None,
    injury_status: str | None,
    is_out: bool,
    is_doubtful: bool,
) -> tuple[bool, str]:
    rs = (roster_status or "").strip().upper()
    if rs in ROSTER_UNAVAILABLE:
        return False, f"roster:{rs}"

    if _injury_long_term(injury_status):
        return False, f"injury:{injury_status}"

    if horizon == "weekly":
        if is_out or _injury_long_term(injury_status):
            # is_out covers Out / IR / suspend from flags
            if is_out:
                return False, f"injury:{injury_status or 'Out'}"
        # Doubtful: soft — still on chart but flagged (caller may +1 rank)
        if is_doubtful:
            return True, "doubtful"
        return True, "active"

    # season: ignore short-term Out / Q / Doubtful
    return True, "active"


def build_effective_depth(
    depth: pl.DataFrame,
    *,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame | None = None,
    ids: pl.DataFrame | None = None,
    horizon: Horizon = "season",
    force_injuries: bool = False,
) -> pl.DataFrame:
    """Build effective depth ranks for skill positions.

    Parameters
    ----------
    depth:
        Raw nflverse depth charts (weekly or snapshot).
    rosters:
        Season rosters with gsis_id, team, status.
    injuries:
        Optional pre-fetched ESPN injuries; fetched if None.
    ids:
        Optional ff_playerids for espn_id↔gsis_id; loaded if None.
    horizon:
        ``season`` (draft) or ``weekly`` (slate week).
    """
    skill = _skill_depth_rows(depth)
    if skill.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA).with_columns(pl.lit(horizon).alias("horizon"))

    # Roster join
    if rosters.is_empty() or "gsis_id" not in rosters.columns:
        logger.warning("effective_depth: empty rosters; returning empty")
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    rost = rosters.with_columns(pl.col("gsis_id").cast(pl.Utf8))
    rost = normalize_team_column(rost, "team") if "team" in rost.columns else rost
    status_col = "status" if "status" in rost.columns else None
    rost_keep = ["gsis_id"]
    if "team" in rost.columns:
        rost_keep.append("team")
    if status_col:
        rost_keep.append(pl.col(status_col).alias("roster_status"))
    else:
        rost = rost.with_columns(pl.lit(None).cast(pl.Utf8).alias("roster_status"))
        rost_keep.append("roster_status")
    rost = rost.select(rost_keep).unique(subset=["gsis_id"], keep="first")

    # Prefer roster team over depth team
    skill = skill.drop("team") if "team" in skill.columns else skill
    df = skill.join(rost, on="gsis_id", how="inner")  # drop unsigned
    if df.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    # Injuries
    if injuries is None:
        try:
            injuries = fetch_espn_injuries(force=force_injuries)
        except Exception as exc:
            logger.warning("effective_depth: ESPN injuries unavailable: %s", exc)
            injuries = pl.DataFrame()
    if ids is None:
        try:
            ids = load_ff_playerids()
        except Exception:
            ids = pl.DataFrame()

    inj = prepare_espn_injury_features(injuries, ids) if not injuries.is_empty() else pl.DataFrame(
        schema={
            "gsis_id": pl.Utf8,
            "injury_status": pl.Utf8,
            "is_out": pl.Boolean,
            "is_doubtful": pl.Boolean,
            "is_questionable": pl.Boolean,
            "play_prob": pl.Float64,
        }
    )
    df = df.join(inj, on="gsis_id", how="left")
    df = df.with_columns(
        [
            pl.col("is_out").fill_null(False),
            pl.col("is_doubtful").fill_null(False),
            pl.col("is_questionable").fill_null(False),
            pl.col("injury_status").cast(pl.Utf8),
            pl.col("roster_status").cast(pl.Utf8),
        ]
    )

    # Availability (package-only FB never gets a skill role)
    rows = df.iter_rows(named=True)
    avail_flags: list[bool] = []
    avail_reasons: list[str] = []
    soft_penalty: list[int] = []  # weekly doubtful +1
    for row in rows:
        if row.get("_package_only"):
            avail_flags.append(False)
            avail_reasons.append("package:FB")
            soft_penalty.append(0)
            continue
        ok, reason = _availability(
            horizon=horizon,
            roster_status=row.get("roster_status"),
            injury_status=row.get("injury_status"),
            is_out=bool(row.get("is_out")),
            is_doubtful=bool(row.get("is_doubtful")),
        )
        avail_flags.append(ok)
        avail_reasons.append(reason)
        soft_penalty.append(1 if (horizon == "weekly" and reason == "doubtful") else 0)

    df = df.with_columns(
        [
            pl.Series("available", avail_flags),
            pl.Series("availability_reason", avail_reasons),
            pl.Series("_soft_penalty", soft_penalty),
        ]
    )

    # Re-rank available players; unavailable keep null effective rank
    available = df.filter(pl.col("available"))
    unavailable = df.filter(~pl.col("available"))

    if not available.is_empty():
        available = available.with_columns(
            (pl.col("raw_depth_rank").fill_null(99.0) + pl.col("_soft_penalty")).alias("_sort_rank")
        )
        available = available.sort(["team", "position", "_sort_rank", "gsis_id"])
        available = available.with_columns(
            pl.col("_sort_rank")
            .rank(method="ordinal")
            .over(["team", "position"])
            .cast(pl.Float64)
            .alias("effective_depth_rank")
        )
        available = available.with_columns(
            [
                (pl.col("effective_depth_rank") == 1.0).cast(pl.Int8).alias("is_effective_starter"),
                (
                    pl.col("position")
                    + pl.col("effective_depth_rank").cast(pl.Int64).cast(pl.Utf8)
                ).alias("role_slot"),
            ]
        )
    else:
        available = available.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("effective_depth_rank"),
                pl.lit(0).cast(pl.Int8).alias("is_effective_starter"),
                pl.lit(None).cast(pl.Utf8).alias("role_slot"),
            ]
        )

    if not unavailable.is_empty():
        unavailable = unavailable.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("effective_depth_rank"),
                pl.lit(0).cast(pl.Int8).alias("is_effective_starter"),
                pl.lit(None).cast(pl.Utf8).alias("role_slot"),
            ]
        )

    out = pl.concat([available, unavailable], how="diagonal_relaxed")
    out = out.with_columns(pl.lit(horizon).alias("horizon"))
    keep = [
        "gsis_id",
        "team",
        "position",
        "effective_depth_rank",
        "raw_depth_rank",
        "role_slot",
        "is_effective_starter",
        "available",
        "availability_reason",
        "horizon",
        "as_of_dt",
    ]
    for opt in ("player_name", "roster_status", "injury_status"):
        if opt in out.columns:
            keep.append(opt)
    logger.info(
        "Effective depth (%s): %d available / %d total skill rostered",
        horizon,
        out.filter(pl.col("available")).height,
        out.height,
    )
    return out.select(keep)


def attach_effective_depth(
    panel: pl.DataFrame,
    effective: pl.DataFrame,
    *,
    overwrite_depth_rank: bool = True,
) -> pl.DataFrame:
    """Join effective depth onto a player panel; optionally replace ``depth_rank``."""
    if effective.is_empty() or "gsis_id" not in panel.columns:
        return panel

    join_cols = [
        c
        for c in (
            "gsis_id",
            "effective_depth_rank",
            "raw_depth_rank",
            "role_slot",
            "is_effective_starter",
            "available",
            "availability_reason",
        )
        if c in effective.columns
    ]
    eff = effective.select(join_cols).unique(subset=["gsis_id"], keep="first")
    drop = [c for c in join_cols if c != "gsis_id" and c in panel.columns]
    out = panel.drop(drop) if drop else panel
    out = out.join(eff, on="gsis_id", how="left")

    if overwrite_depth_rank and "effective_depth_rank" in out.columns:
        avail = (
            pl.col("available").fill_null(True)
            if "available" in out.columns
            else pl.lit(True)
        )
        if "depth_rank" in out.columns:
            out = out.with_columns(
                pl.when(avail)
                .then(pl.coalesce([pl.col("effective_depth_rank"), pl.col("depth_rank")]))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("depth_rank")
            )
        else:
            out = out.with_columns(
                pl.when(avail)
                .then(pl.col("effective_depth_rank"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("depth_rank")
            )
        out = out.with_columns(
            (pl.col("depth_rank") == 1.0)
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_listed_starter")
        )
    return out


def clear_short_term_injuries(panel: pl.DataFrame) -> pl.DataFrame:
    """Clear ESPN short-term Out/Q/Doubtful on season outlook rows.

    Long-term IR/suspend remain demoted via effective depth availability;
    week-noise statuses must not freeze across all 17 projected weeks.
    """
    if panel.is_empty():
        return panel
    status = (
        pl.col("injury_status").cast(pl.Utf8).str.to_lowercase().fill_null("")
        if "injury_status" in panel.columns
        else pl.lit("")
    )
    long_term = (
        status.str.contains("injured reserve")
        | (status == "ir")
        | status.str.contains("suspend")
        | status.str.contains("pup")
        | status.str.contains("nfi")
    )
    exprs: list[pl.Expr] = []
    if "is_out" in panel.columns:
        exprs.append(
            pl.when(long_term).then(pl.col("is_out")).otherwise(pl.lit(False)).alias("is_out")
        )
    if "is_doubtful" in panel.columns:
        exprs.append(pl.lit(False).alias("is_doubtful"))
    if "is_questionable" in panel.columns:
        exprs.append(pl.lit(False).alias("is_questionable"))
    if "play_prob" in panel.columns:
        exprs.append(
            pl.when(long_term)
            .then(pl.col("play_prob").fill_null(0.0))
            .otherwise(pl.lit(1.0))
            .alias("play_prob")
        )
    if "injury_status" in panel.columns:
        exprs.append(
            pl.when(long_term)
            .then(pl.col("injury_status"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("injury_status")
        )
    return panel.with_columns(exprs) if exprs else panel
