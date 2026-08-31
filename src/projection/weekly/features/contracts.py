"""OverTheCap contract features via nflverse load_contracts (rotc)."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

CONTRACT_FEATURE_COLS = [
    "contract_apy_cap_pct",
    "contract_inflated_apy",
    "contract_guaranteed_pct",
    "contract_years",
    "contract_years_remaining",
    "contract_year_signed",
]


def _null_contract_cols(panel: pl.DataFrame) -> pl.DataFrame:
    extras = []
    for c in CONTRACT_FEATURE_COLS:
        if c not in panel.columns:
            extras.append(pl.lit(None).cast(pl.Float64).alias(c))
    return panel.with_columns(extras) if extras else panel


def select_as_of_contracts(contracts: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """One contract row per (gsis_id, season): latest signing with year_signed <= season."""
    if contracts.is_empty() or "gsis_id" not in contracts.columns:
        return pl.DataFrame(
            schema={
                "gsis_id": pl.Utf8,
                "season": pl.Int64,
                "year_signed": pl.Float64,
                "years": pl.Float64,
                "value": pl.Float64,
                "apy": pl.Float64,
                "guaranteed": pl.Float64,
                "apy_cap_pct": pl.Float64,
                "inflated_apy": pl.Float64,
                "is_active": pl.Boolean,
            }
        )

    df = contracts.filter(pl.col("gsis_id").is_not_null() & pl.col("year_signed").is_not_null())
    if df.is_empty():
        return pl.DataFrame(schema={"gsis_id": pl.Utf8, "season": pl.Int64})

    # Expand each contract against seasons it can cover as-of
    season_frame = pl.DataFrame({"season": seasons}).with_columns(pl.col("season").cast(pl.Int64))
    crossed = df.join(season_frame, how="cross").filter(pl.col("year_signed") <= pl.col("season"))
    if crossed.is_empty():
        return pl.DataFrame(schema={"gsis_id": pl.Utf8, "season": pl.Int64})

    if "is_active" not in crossed.columns:
        crossed = crossed.with_columns(pl.lit(False).alias("is_active"))
    crossed = crossed.with_columns(
        pl.col("is_active").fill_null(False).cast(pl.Boolean).alias("is_active")
    )

    # Prefer newest year_signed; on ties prefer active
    crossed = crossed.sort(
        ["gsis_id", "season", "year_signed", "is_active"],
        descending=[False, False, True, True],
    )
    keep = [
        c
        for c in (
            "gsis_id",
            "season",
            "year_signed",
            "years",
            "value",
            "apy",
            "guaranteed",
            "apy_cap_pct",
            "inflated_apy",
            "is_active",
        )
        if c in crossed.columns
    ]
    return crossed.select(keep).unique(subset=["gsis_id", "season"], keep="first")


def attach_contract_features(panel: pl.DataFrame, contracts: pl.DataFrame) -> pl.DataFrame:
    """Join leak-safe as-of contract features onto player-week panel."""
    if panel.is_empty() or "gsis_id" not in panel.columns or "season" not in panel.columns:
        return _null_contract_cols(panel)

    seasons = sorted(int(s) for s in panel["season"].unique().to_list() if s is not None)
    as_of = select_as_of_contracts(contracts, seasons)
    if as_of.is_empty() or "year_signed" not in as_of.columns:
        logger.warning("No as-of contracts matched; filling nulls")
        return _null_contract_cols(panel)

    apy = (
        pl.coalesce([pl.col("inflated_apy"), pl.col("apy")])
        if "inflated_apy" in as_of.columns and "apy" in as_of.columns
        else (pl.col("inflated_apy") if "inflated_apy" in as_of.columns else pl.col("apy"))
    )
    value = pl.col("value") if "value" in as_of.columns else pl.lit(None)
    guaranteed = pl.col("guaranteed") if "guaranteed" in as_of.columns else pl.lit(None)
    years = pl.col("years") if "years" in as_of.columns else pl.lit(None)
    year_signed = pl.col("year_signed")
    apy_cap = pl.col("apy_cap_pct") if "apy_cap_pct" in as_of.columns else pl.lit(None)

    as_of = as_of.with_columns(
        [
            apy_cap.cast(pl.Float64).alias("contract_apy_cap_pct"),
            apy.cast(pl.Float64).alias("contract_inflated_apy"),
            (
                pl.when(value.is_not_null() & (value.cast(pl.Float64) > 1e-9))
                .then((guaranteed.cast(pl.Float64) / value.cast(pl.Float64)).clip(0.0, 1.0))
                .otherwise(None)
                .alias("contract_guaranteed_pct")
            ),
            years.cast(pl.Float64).alias("contract_years"),
            (
                (year_signed.cast(pl.Float64) + years.cast(pl.Float64) - pl.col("season").cast(pl.Float64))
                .clip(lower_bound=0.0)
                .alias("contract_years_remaining")
            ),
            year_signed.cast(pl.Float64).alias("contract_year_signed"),
        ]
    ).select(["gsis_id", "season"] + CONTRACT_FEATURE_COLS)

    out = panel
    drop = [c for c in CONTRACT_FEATURE_COLS if c in out.columns]
    if drop:
        out = out.drop(drop)
    out = out.join(as_of, on=["gsis_id", "season"], how="left")
    return out
