"""Point-in-time Sleeper status/depth overlay for the live season only."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

SLEEPER_OVERLAY_COLS = (
    "sleeper_active",
    "sleeper_status",
    "sleeper_injury_status",
    "sleeper_practice_participation",
    "sleeper_depth_rank",
    "sleeper_is_out",
    "sleeper_is_ir",
)


def _empty_columns(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    types = {
        "sleeper_active": pl.Boolean,
        "sleeper_status": pl.Utf8,
        "sleeper_injury_status": pl.Utf8,
        "sleeper_practice_participation": pl.Utf8,
        "sleeper_depth_rank": pl.Float64,
        "sleeper_is_out": pl.Boolean,
        "sleeper_is_ir": pl.Boolean,
    }
    for col, dtype in types.items():
        if col not in df.columns:
            exprs.append(pl.lit(None).cast(dtype).alias(col))
    return df.with_columns(exprs) if exprs else df


def prepare_sleeper_overlay(
    snapshot: pl.DataFrame, id_map: pl.DataFrame
) -> pl.DataFrame:
    """Map Sleeper IDs to GSIS IDs and derive conservative status flags."""
    required_snapshot = {"sleeper_id"}
    required_ids = {"sleeper_id", "gsis_id"}
    if (
        snapshot.is_empty()
        or id_map.is_empty()
        or not required_snapshot.issubset(snapshot.columns)
        or not required_ids.issubset(id_map.columns)
    ):
        return _empty_columns(pl.DataFrame(schema={"gsis_id": pl.Utf8}))
    ids = (
        id_map.select(
            [
                pl.col("sleeper_id").cast(pl.Utf8),
                pl.col("gsis_id").cast(pl.Utf8),
            ]
        )
        .drop_nulls()
        .unique(subset=["sleeper_id"], keep="first")
    )
    out = snapshot.with_columns(pl.col("sleeper_id").cast(pl.Utf8)).join(
        ids, on="sleeper_id", how="inner"
    )
    status = pl.concat_str(
        [
            pl.col("sleeper_status").cast(pl.Utf8).fill_null(""),
            pl.col("sleeper_injury_status").cast(pl.Utf8).fill_null(""),
        ],
        separator=" ",
    ).str.to_lowercase()
    out = out.with_columns(
        [
            (status.str.contains(r"\bout\b") | status.str.contains("inactive"))
            .alias("sleeper_is_out"),
            (
                status.str.contains("injured reserve")
                | status.str.contains(r"\bir\b")
                | status.str.contains("reserve")
                | status.str.contains("pup")
            ).alias("sleeper_is_ir"),
        ]
    )
    keep = ["gsis_id"] + [c for c in SLEEPER_OVERLAY_COLS if c in out.columns]
    return _empty_columns(out.select(keep).unique(subset=["gsis_id"], keep="last"))


def attach_current_sleeper_overlay(
    panel: pl.DataFrame,
    snapshot: pl.DataFrame,
    id_map: pl.DataFrame,
    *,
    live_season: int,
) -> pl.DataFrame:
    """Attach a current snapshot only to ``live_season`` rows.

    Historical rows receive nulls, not healthy defaults. This makes accidental
    use of today's status in historical training visible to feature selection.
    """
    drop = [c for c in SLEEPER_OVERLAY_COLS if c in panel.columns]
    base = panel.drop(drop) if drop else panel
    if base.is_empty() or "season" not in base.columns or "gsis_id" not in base.columns:
        return _empty_columns(base)
    overlay = prepare_sleeper_overlay(snapshot, id_map)
    historical = _empty_columns(base.filter(pl.col("season") != live_season))
    current = base.filter(pl.col("season") == live_season)
    if current.is_empty() or overlay.is_empty():
        current = _empty_columns(current)
    else:
        current = _empty_columns(current.join(overlay, on="gsis_id", how="left"))
    return pl.concat([historical, current], how="diagonal_relaxed")
