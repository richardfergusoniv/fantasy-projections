"""Depth chart features with dual schema support (weekly vs timestamped)."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

SKILL_POS = {"QB", "RB", "WR", "TE", "FB", "HB"}


def _normalize_weekly_depth(depth: pl.DataFrame) -> pl.DataFrame:
    """Pre-2025 style: season/week/gsis_id + depth_team rank on offense."""
    empty_schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "gsis_id": pl.Utf8,
        "depth_rank": pl.Float64,
        "is_listed_starter": pl.Int8,
        "same_pos_depth_count": pl.Int64,
    }
    if depth.is_empty():
        return pl.DataFrame(schema=empty_schema)

    df = depth
    if "team" not in df.columns and "club_code" in df.columns:
        df = df.with_columns(pl.col("club_code").alias("team"))
    if "formation" in df.columns:
        df = df.filter(
            pl.col("formation").is_null()
            | pl.col("formation").str.to_lowercase().str.contains("off")
        )
    if "position" in df.columns:
        df = df.filter(
            pl.col("position").is_null()
            | pl.col("position").is_in(list(SKILL_POS))
            | pl.col("position").is_in(["QB", "RB", "WR", "TE"])
        )

    rank_col = "depth_team" if "depth_team" in df.columns else None
    if rank_col is None and "pos_rank" in df.columns:
        rank_col = "pos_rank"
    if rank_col is None or "team" not in df.columns:
        return pl.DataFrame(schema=empty_schema)

    pos_key = "depth_position" if "depth_position" in df.columns else "position"
    df = df.filter(pl.col("gsis_id").is_not_null() & pl.col("week").is_not_null())
    if df.is_empty():
        return pl.DataFrame(schema=empty_schema)

    df = df.with_columns(
        [
            pl.col(rank_col).cast(pl.Float64).alias("depth_rank"),
            (pl.col(rank_col).cast(pl.Float64) == 1.0).cast(pl.Int8).alias("is_listed_starter"),
        ]
    )

    if pos_key in df.columns:
        counts = df.group_by(["season", "week", "team", pos_key]).agg(
            pl.len().alias("same_pos_depth_count")
        )
        df = df.join(counts, on=["season", "week", "team", pos_key], how="left")
    else:
        df = df.with_columns(pl.lit(1).cast(pl.Int64).alias("same_pos_depth_count"))

    return (
        df.select(
            [
                "season",
                "week",
                "gsis_id",
                "depth_rank",
                "is_listed_starter",
                "same_pos_depth_count",
            ]
        )
        .unique(subset=["season", "week", "gsis_id"], keep="first")
    )


def _normalize_snapshot_depth(depth: pl.DataFrame) -> pl.DataFrame:
    """2025+ style: dt timestamps, pos_rank, no week."""
    df = depth
    if "dt" not in df.columns:
        return pl.DataFrame(
            schema={
                "gsis_id": pl.Utf8,
                "team": pl.Utf8,
                "dt": pl.Datetime,
                "depth_rank": pl.Float64,
                "is_listed_starter": pl.Int8,
                "same_pos_depth_count": pl.Int64,
            }
        )

    # Drop defense / ST package rows; keep offense packages (e.g. "3WR 1TE")
    if "pos_grp" in df.columns:
        grp = pl.col("pos_grp").str.to_lowercase()
        df = df.filter(
            pl.col("pos_grp").is_null()
            | grp.str.contains("off")
            | pl.col("pos_grp").is_in(["WR", "RB", "QB", "TE", "Offense"])
            | (
                (~grp.str.contains("base"))
                & (~grp.str.contains("special"))
                & (~grp.str.contains(" d"))
            )
        )
    if "pos_abb" in df.columns:
        # Keep skill-ish abbreviations; still keep rows with null abb
        skill_abb = [
            "QB",
            "RB",
            "FB",
            "WR",
            "TE",
            "LWR",
            "RWR",
            "SWR",
            "SLWR",
            "SRWR",
            "HB",
        ]
        df = df.filter(pl.col("pos_abb").is_null() | pl.col("pos_abb").is_in(skill_abb))

    rank_col = "pos_rank" if "pos_rank" in df.columns else "depth_team"
    pos_key = "pos_abb" if "pos_abb" in df.columns else ("position" if "position" in df.columns else None)

    if df.schema.get("dt") in (pl.Utf8, pl.String):
        dt_expr = (
            pl.col("dt")
            .str.replace(r"Z$", "+00:00")
            .str.to_datetime(time_zone="UTC", strict=False)
            .dt.replace_time_zone(None)
        )
    else:
        dt_expr = pl.col("dt").cast(pl.Datetime, strict=False)

    df = df.with_columns(
        [
            dt_expr.alias("dt"),
            pl.col(rank_col).cast(pl.Float64).alias("depth_rank"),
            (pl.col(rank_col).cast(pl.Float64) == 1.0).cast(pl.Int8).alias("is_listed_starter"),
            pl.col("gsis_id").cast(pl.Utf8),
        ]
    ).filter(pl.col("gsis_id").is_not_null() & pl.col("dt").is_not_null())

    group_keys = ["dt", "team"] + ([pos_key] if pos_key else [])
    if pos_key:
        counts = df.group_by(group_keys).agg(pl.len().alias("same_pos_depth_count"))
        df = df.join(counts, on=group_keys, how="left")
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("same_pos_depth_count"))

    return df.select(
        ["gsis_id", "team", "dt", "depth_rank", "is_listed_starter", "same_pos_depth_count"]
    )


def attach_depth_features(
    panel: pl.DataFrame,
    depth: pl.DataFrame,
    schedules: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join depth_rank / starter / committee size onto player-week panel."""
    empty = [
        pl.lit(None).cast(pl.Float64).alias("depth_rank"),
        pl.lit(None).cast(pl.Int8).alias("is_listed_starter"),
        pl.lit(None).cast(pl.Int64).alias("same_pos_depth_count"),
    ]
    if depth.is_empty() or "gsis_id" not in panel.columns:
        return panel.with_columns(empty)

    # Split by schema
    has_week = "week" in depth.columns and depth["week"].null_count() < depth.height
    has_dt = "dt" in depth.columns and depth.filter(pl.col("dt").is_not_null()).height > 0

    out = panel
    joined_ids: set[str] = set()

    if has_week:
        weekly = depth.filter(pl.col("week").is_not_null()) if "week" in depth.columns else depth
        # Drop 2025+ snapshot-only rows mixed into weekly schema
        if "season" in weekly.columns:
            weekly = weekly.filter(pl.col("season") < 2025)
        norm = _normalize_weekly_depth(weekly)
        if not norm.is_empty():
            out = out.join(norm, on=["season", "week", "gsis_id"], how="left")
            joined_ids.update(["depth_rank", "is_listed_starter", "same_pos_depth_count"])

    if has_dt and schedules is not None and not schedules.is_empty():
        # Any row with a usable dt is a snapshot candidate. Do NOT filter on
        # season — null-season snapshots must still attach via dt vs kickoff.
        snaps = depth.filter(pl.col("dt").is_not_null())
        norm_s = _normalize_snapshot_depth(snaps)
        if not norm_s.is_empty() and (
            "gameday" in schedules.columns or "gametime" in schedules.columns
        ):
            # Build kickoff timestamps per team-week
            sched = schedules
            kick_exprs = []
            if "gameday" in sched.columns:
                kick_exprs.append(pl.col("gameday").cast(pl.Utf8))
            # Prefer gameday as date; ignore timezone complexity
            home = sched.select(
                [
                    "season",
                    "week",
                    pl.col("home_team").alias("team"),
                    pl.col("gameday").alias("kick_date") if "gameday" in sched.columns else pl.lit(None).alias("kick_date"),
                ]
            )
            away = sched.select(
                [
                    "season",
                    "week",
                    pl.col("away_team").alias("team"),
                    pl.col("gameday").alias("kick_date") if "gameday" in sched.columns else pl.lit(None).alias("kick_date"),
                ]
            )
            team_kick = pl.concat([home, away], how="vertical_relaxed").with_columns(
                pl.col("kick_date").cast(pl.Utf8).str.to_date(strict=False).alias("kick_date")
            )
            panel_keys = out.select(["season", "week", "team", "gsis_id"]).unique()
            panel_keys = panel_keys.join(team_kick, on=["season", "week", "team"], how="left")

            # As-of: for each player-week, take latest depth snapshot with dt.date() <= kick_date
            norm_s = norm_s.with_columns(pl.col("dt").cast(pl.Datetime).dt.date().alias("snap_date"))
            # Join all snapshots for player then filter
            cand = panel_keys.join(
                norm_s.select(
                    ["gsis_id", "snap_date", "dt", "depth_rank", "is_listed_starter", "same_pos_depth_count"]
                ),
                on="gsis_id",
                how="left",
            )
            cand = cand.filter(
                pl.col("kick_date").is_null()
                | pl.col("snap_date").is_null()
                | (pl.col("snap_date") <= pl.col("kick_date"))
            )
            cand = cand.sort(["gsis_id", "season", "week", "dt"]).unique(
                subset=["gsis_id", "season", "week"], keep="last"
            )
            snap_feats = cand.select(
                [
                    "season",
                    "week",
                    "gsis_id",
                    "depth_rank",
                    "is_listed_starter",
                    "same_pos_depth_count",
                ]
            )
            # Coalesce onto out if weekly didn't fill
            if "depth_rank" in out.columns:
                out = out.join(snap_feats, on=["season", "week", "gsis_id"], how="left", suffix="_snap")
                out = out.with_columns(
                    [
                        pl.coalesce([pl.col("depth_rank"), pl.col("depth_rank_snap")]).alias("depth_rank"),
                        pl.coalesce([pl.col("is_listed_starter"), pl.col("is_listed_starter_snap")]).alias(
                            "is_listed_starter"
                        ),
                        pl.coalesce(
                            [pl.col("same_pos_depth_count"), pl.col("same_pos_depth_count_snap")]
                        ).alias("same_pos_depth_count"),
                    ]
                ).drop(
                    [
                        c
                        for c in ("depth_rank_snap", "is_listed_starter_snap", "same_pos_depth_count_snap")
                        if c in out.columns
                    ]
                )
            else:
                out = out.join(snap_feats, on=["season", "week", "gsis_id"], how="left")
            joined_ids.update(["depth_rank", "is_listed_starter", "same_pos_depth_count"])

    for col, expr in zip(
        ["depth_rank", "is_listed_starter", "same_pos_depth_count"],
        empty,
        strict=True,
    ):
        if col not in out.columns:
            out = out.with_columns(expr)

    logger.info(
        "Depth features attached: depth_rank non-null=%d / %d",
        out.filter(pl.col("depth_rank").is_not_null()).height,
        out.height,
    )
    return out


def clip_depth_rank_for_models(
    df: pl.DataFrame,
    *,
    col: str = "depth_rank",
    max_rank: float = 5.0,
    floor_rank: float = 6.0,
) -> pl.DataFrame:
    """Clip serve-time depth ranks to training support (1–5; 6+ floored).

    Historical weekly charts only used ranks {1,2,3}; snapshot charts can run
    1–14. Models trained on {1,2,3} treat large ranks as OOD — floor them.
    """
    if col not in df.columns or df.is_empty():
        return df
    return df.with_columns(
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(pl.col(col) > max_rank)
        .then(pl.lit(floor_rank))
        .otherwise(pl.col(col).clip(1.0, max_rank))
        .alias(col)
    )
