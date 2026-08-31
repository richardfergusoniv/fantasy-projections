"""Leakage guards for as-of feature construction."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl


OUTCOME_COLUMNS = frozenset(
    {
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "targets",
        "carries",
        "attempts",
        "completions",
        "fantasy_points",
        "fantasy_points_ppr",
        "fantasy_points_half_ppr",
        "air_yards",
        "target_share",
        "wopr",
        "racr",
        "passing_epa",
        "rushing_epa",
        "receiving_epa",
        "offense_snaps",
        "offense_pct",
        "fumbles_lost",
        "sacks",
        "sack_fumbles_lost",
    }
)


def filter_as_of(
    df: pl.DataFrame,
    *,
    season: int,
    week: int,
    season_col: str = "season",
    week_col: str = "week",
    include_prior_seasons: bool = True,
) -> pl.DataFrame:
    """Return rows strictly before (season, week) for feature history.

    Includes prior seasons in full and same-season weeks < week.
    Never includes the as-of week itself.
    """
    if include_prior_seasons:
        return df.filter(
            (pl.col(season_col) < season)
            | ((pl.col(season_col) == season) & (pl.col(week_col) < week))
        )
    return df.filter((pl.col(season_col) == season) & (pl.col(week_col) < week))


def assert_no_same_week_outcomes(
    features: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    keys: tuple[str, ...] = ("gsis_id", "season", "week"),
    outcome_cols: frozenset[str] | None = None,
) -> None:
    """Raise if feature frame contains same-week outcome columns joined from outcomes.

    Used in tests: feature rows for (player, season, week) must not equal that week's
    raw box-score outcomes for columns that should only appear as labels.
    """
    outcome_cols = outcome_cols or OUTCOME_COLUMNS
    shared = [c for c in features.columns if c in outcome_cols and c in outcomes.columns]
    if not shared:
        return
    # If both frames have the same keys, ensure feature values for outcome-named
    # columns are NOT identical to the label week (they should be lagged / null).
    join_keys = [k for k in keys if k in features.columns and k in outcomes.columns]
    if len(join_keys) < 3:
        return
    merged = features.select(join_keys + shared).join(
        outcomes.select(join_keys + shared),
        on=join_keys,
        how="inner",
        suffix="_actual",
    )
    if merged.is_empty():
        return
    # Check at least one lagged column differs or is null when actual is non-null
    for col in shared:
        actual = f"{col}_actual"
        if actual not in merged.columns:
            continue
        identical = merged.filter(
            pl.col(col).is_not_null()
            & pl.col(actual).is_not_null()
            & (pl.col(col) == pl.col(actual))
            & (pl.col(actual) != 0)
        )
        # Allow some coincidence; fail only if a huge fraction matches exactly
        if identical.height > 0.95 * merged.filter(pl.col(actual) != 0).height and identical.height > 50:
            raise AssertionError(
                f"Possible leakage: feature column {col!r} matches same-week outcomes "
                f"for {identical.height}/{merged.height} rows"
            )


def pregame_schedule_columns() -> tuple[str, ...]:
    """Schedule columns that are known before kickoff (safe as features)."""
    return (
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "home_rest",
        "away_rest",
        "roof",
        "surface",
        "temp",
        "wind",
        "away_team",
        "home_team",
        "weekday",
        "gametime",
    )
