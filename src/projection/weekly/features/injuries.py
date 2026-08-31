"""Injury status features for volume / play probability.

Prefer nflverse weekly reports (season/week/gsis_id) for completed seasons.
ESPN is a point-in-time snapshot — apply only to the live season so completed
seasons are not hard-zeroed by a later feed.
"""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

from src.projection.weekly.data.espn_injuries import fetch_espn_injuries, injury_status_flags
from src.projection.weekly.data.ids import coerce_id_columns
from src.projection.weekly.data.nflverse_loader import load_ff_playerids, load_injuries_nflverse

logger = logging.getLogger(__name__)

INJURY_FLAG_COLS = (
    "injury_status",
    "is_out",
    "is_doubtful",
    "is_questionable",
    "play_prob",
)

_VOLUME_STAT_COLS = (
    "fantasy_points",
    "floor",
    "ceiling",
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
)


def _status_severity(status_col: str = "status") -> pl.Expr:
    s = pl.col(status_col).cast(pl.Utf8).str.to_lowercase().fill_null("")
    inactive = (
        s.str.contains("out")
        | s.str.contains("injured reserve")
        | (s == "ir")
        | s.str.contains("suspend")
    )
    return (
        pl.when(inactive)
        .then(3)
        .when(s.str.contains("doubt"))
        .then(2)
        .when(s.str.contains("question"))
        .then(1)
        .otherwise(0)
    )


def _healthy_defaults(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.lit(None).cast(pl.Utf8).alias("injury_status"),
            pl.lit(False).alias("is_out"),
            pl.lit(False).alias("is_doubtful"),
            pl.lit(False).alias("is_questionable"),
            pl.lit(1.0).alias("play_prob"),
        ]
    )


def prepare_nflverse_injury_features(injuries: pl.DataFrame) -> pl.DataFrame:
    """Collapse nflverse injuries to one row per season/week/gsis_id with flags."""
    if injuries.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "gsis_id": pl.Utf8,
                "injury_status": pl.Utf8,
                "is_out": pl.Boolean,
                "is_doubtful": pl.Boolean,
                "is_questionable": pl.Boolean,
                "play_prob": pl.Float64,
            }
        )

    df = coerce_id_columns(injuries, ("gsis_id",))
    status_col = "report_status" if "report_status" in df.columns else "status"
    if status_col not in df.columns:
        logger.warning("nflverse injuries missing status column; skipping")
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "gsis_id": pl.Utf8,
                "injury_status": pl.Utf8,
                "is_out": pl.Boolean,
                "is_doubtful": pl.Boolean,
                "is_questionable": pl.Boolean,
                "play_prob": pl.Float64,
            }
        )

    df = df.filter(
        pl.col("gsis_id").is_not_null()
        & pl.col("season").is_not_null()
        & pl.col("week").is_not_null()
    )
    df = df.with_columns(pl.col(status_col).cast(pl.Utf8).alias("status"))
    df = df.with_columns(_status_severity("status").alias("_sev"))
    sort_cols = ["season", "week", "gsis_id", "_sev"]
    descending = [False, False, False, True]
    if "date_modified" in df.columns:
        sort_cols.append("date_modified")
        descending.append(True)
    df = df.sort(sort_cols, descending=descending).unique(
        subset=["season", "week", "gsis_id"], keep="first"
    )
    df = injury_status_flags(df)
    return df.select(
        [
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("gsis_id"),
            pl.col("status").alias("injury_status"),
            "is_out",
            "is_doubtful",
            "is_questionable",
            "play_prob",
        ]
    )


def prepare_espn_injury_features(
    espn: pl.DataFrame,
    ids: pl.DataFrame,
) -> pl.DataFrame:
    """Map ESPN injuries onto gsis_id via ff_playerids crosswalk."""
    empty = pl.DataFrame(
        schema={
            "gsis_id": pl.Utf8,
            "injury_status": pl.Utf8,
            "is_out": pl.Boolean,
            "is_doubtful": pl.Boolean,
            "is_questionable": pl.Boolean,
            "play_prob": pl.Float64,
        }
    )
    if espn.is_empty() or "espn_id" not in espn.columns:
        return empty
    if ids.is_empty() or "espn_id" not in ids.columns or "gsis_id" not in ids.columns:
        logger.warning("ff_playerids missing espn_id/gsis_id; cannot attach ESPN injuries")
        return empty

    espn_df = coerce_id_columns(espn, ("espn_id",))
    espn_df = injury_status_flags(espn_df)
    id_map = (
        coerce_id_columns(ids, ("espn_id", "gsis_id"))
        .select(["espn_id", "gsis_id"])
        .filter(pl.col("espn_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .unique(subset=["espn_id"], keep="first")
    )
    joined = espn_df.join(id_map, on="espn_id", how="inner")
    if joined.is_empty():
        return empty

    joined = joined.with_columns(_status_severity("status").alias("_sev"))
    joined = joined.sort(["gsis_id", "_sev"], descending=[False, True]).unique(
        subset=["gsis_id"], keep="first"
    )
    return joined.select(
        [
            "gsis_id",
            pl.col("status").alias("injury_status"),
            "is_out",
            "is_doubtful",
            "is_questionable",
            "play_prob",
        ]
    )


def attach_injury_features(
    panel: pl.DataFrame,
    *,
    force_reload: bool = False,
    nflverse_injuries: pl.DataFrame | None = None,
    espn_injuries: pl.DataFrame | None = None,
    ids: pl.DataFrame | None = None,
    live_season: int | None = None,
) -> pl.DataFrame:
    """Join injury flags onto a player-week panel (leakage-aware by source).

    Prefer nflverse week-keyed reports whenever available. ESPN is a
    point-in-time snapshot — apply it only to ``live_season`` so completed
    seasons are not hard-zeroed by a later feed.
    """
    drop_existing = [c for c in INJURY_FLAG_COLS if c in panel.columns]
    base = panel.drop(drop_existing) if drop_existing else panel
    if base.is_empty():
        return _healthy_defaults(base)

    seasons = sorted({int(s) for s in base["season"].drop_nulls().unique().to_list()})
    live = int(live_season if live_season is not None else max(seasons))

    raw = (
        nflverse_injuries
        if nflverse_injuries is not None
        else load_injuries_nflverse(seasons, force=force_reload)
    )
    nfl_feats = (
        prepare_nflverse_injury_features(raw)
        if not raw.is_empty()
        else pl.DataFrame()
    )
    nfl_seasons: set[int] = set()
    if not nfl_feats.is_empty() and "season" in nfl_feats.columns:
        nfl_seasons = {int(s) for s in nfl_feats["season"].drop_nulls().unique().to_list()}

    parts: list[pl.DataFrame] = []

    # Completed seasons: nflverse week join only (healthy default if missing)
    completed = base.filter(pl.col("season") < live)
    if not completed.is_empty():
        if not nfl_feats.is_empty():
            completed = completed.join(
                nfl_feats, on=["season", "week", "gsis_id"], how="left"
            )
        else:
            completed = _healthy_defaults(completed)
        parts.append(completed)

    # Live season: nflverse weeks when present, else ESPN snapshot
    modern = base.filter(pl.col("season") >= live)
    if not modern.is_empty():
        if live in nfl_seasons and not nfl_feats.is_empty():
            modern = modern.join(nfl_feats, on=["season", "week", "gsis_id"], how="left")
        else:
            espn = (
                espn_injuries
                if espn_injuries is not None
                else fetch_espn_injuries(force=force_reload)
            )
            id_df = ids if ids is not None else load_ff_playerids(force=force_reload)
            feats = prepare_espn_injury_features(espn, id_df)
            modern = modern.join(feats, on="gsis_id", how="left")
        parts.append(modern)

    if not parts:
        return _healthy_defaults(base)

    out = pl.concat(parts, how="diagonal_relaxed") if len(parts) > 1 else parts[0]
    for col, default in (
        ("injury_status", None),
        ("is_out", False),
        ("is_doubtful", False),
        ("is_questionable", False),
        ("play_prob", 1.0),
    ):
        if col not in out.columns:
            if col == "injury_status":
                out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
            elif col == "play_prob":
                out = out.with_columns(pl.lit(float(default)).alias(col))
            else:
                out = out.with_columns(pl.lit(bool(default)).alias(col))

    out = out.with_columns(
        [
            pl.col("is_out").fill_null(False),
            pl.col("is_doubtful").fill_null(False),
            pl.col("is_questionable").fill_null(False),
            pl.col("play_prob").fill_null(1.0),
        ]
    )
    logger.info(
        "Attached injury features (out=%d, doubtful=%d, questionable=%d)",
        int(out["is_out"].sum()),
        int(out["is_doubtful"].sum()),
        int(out["is_questionable"].sum()),
    )
    return out


def apply_injury_haircut(
    df: pl.DataFrame,
    *,
    mode: str = "all",
) -> pl.DataFrame:
    """Scale or zero projections by injury play probability.

    mode:
      - \"shares\": scale pred_*_share only (call before team accounting)
      - \"stats\": hard-zero counting stats / FP when Out or play_prob ~ 0
      - \"all\": scale shares + volume stats by play_prob (tests / one-shot)
    """
    if df.is_empty():
        return df
    if "play_prob" not in df.columns and "is_out" not in df.columns:
        return df
    if mode not in {"shares", "stats", "all"}:
        raise ValueError(f"Unknown injury haircut mode: {mode}")

    if "play_prob" in df.columns:
        play_prob = pl.col("play_prob").cast(pl.Float64).fill_null(1.0)
    else:
        play_prob = pl.lit(1.0)

    if "is_out" in df.columns:
        play_prob = (
            pl.when(pl.col("is_out").fill_null(False)).then(pl.lit(0.0)).otherwise(play_prob)
        )

    play_prob = pl.when(play_prob <= 1e-6).then(pl.lit(0.0)).otherwise(play_prob)

    exprs: list[pl.Expr] = []

    if mode in {"shares", "all"}:
        for c in df.columns:
            if c.startswith("pred_") and c.endswith("_share"):
                exprs.append((pl.col(c) * play_prob).alias(c))

    if mode == "stats":
        inactive = play_prob <= 1e-6
        for c in _VOLUME_STAT_COLS:
            if c in df.columns:
                exprs.append(
                    pl.when(inactive).then(pl.lit(0.0)).otherwise(pl.col(c)).alias(c)
                )
    elif mode == "all":
        for c in _VOLUME_STAT_COLS:
            if c in df.columns:
                exprs.append((pl.col(c) * play_prob).alias(c))

    return df.with_columns(exprs) if exprs else df
