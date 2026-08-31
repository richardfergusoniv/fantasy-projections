"""Rolling usage features with sample-size shrinkage."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl


def shrink_toward(
    observed: pl.Expr,
    prior: pl.Expr,
    n: pl.Expr,
    *,
    prior_strength: float = 3.0,
) -> pl.Expr:
    """Bayesian-style shrinkage: (n*obs + k*prior) / (n+k)."""
    n_safe = n.cast(pl.Float64).fill_null(0)
    return (n_safe * observed.fill_null(prior) + prior_strength * prior) / (
        n_safe + prior_strength
    )


def add_rolling_means(
    df: pl.DataFrame,
    value_cols: list[str],
    *,
    group_keys: list[str] | None = None,
    windows: tuple[int, ...] = (3, 5),
    season_col: str = "season",
    week_col: str = "week",
) -> pl.DataFrame:
    """Add lag-1 rolling means over prior games within season (no current-week leakage).

    For each window w, creates `{col}_l{w}` = mean of the previous w games
    (shifted so current week is excluded).
    Also adds `{col}_season_td` = expanding mean of prior games in-season.
    """
    group_keys = group_keys or ["gsis_id", "season"]
    out = df.sort(group_keys + [week_col])

    for col in value_cols:
        if col not in out.columns:
            continue
        # lag so current row is excluded
        lagged = pl.col(col).shift(1).over(group_keys)
        out = out.with_columns(lagged.alias(f"__lag_{col}"))
        for w in windows:
            out = out.with_columns(
                pl.col(f"__lag_{col}")
                .rolling_mean(window_size=w, min_samples=1)
                .over(group_keys)
                .alias(f"{col}_l{w}")
            )
        out = out.with_columns(
            pl.col(f"__lag_{col}")
            .cum_sum()
            .over(group_keys)
            .truediv(
                pl.col(f"__lag_{col}")
                .is_not_null()
                .cast(pl.Int64)
                .cum_sum()
                .over(group_keys)
                .cast(pl.Float64)
            )
            .alias(f"{col}_season_td")
        )
        out = out.drop(f"__lag_{col}")

    return out


def add_prior_season_means(
    df: pl.DataFrame,
    value_cols: list[str],
    *,
    player_col: str = "gsis_id",
    season_col: str = "season",
) -> pl.DataFrame:
    """Attach previous-season per-game means as `{col}_prior_season`."""
    present = [c for c in value_cols if c in df.columns]
    if not present:
        return df

    prior = (
        df.group_by([player_col, season_col])
        .agg([pl.col(c).mean().alias(c) for c in present])
        .with_columns((pl.col(season_col) + 1).alias(season_col))
        .rename({c: f"{c}_prior_season" for c in present})
    )
    return df.join(prior, on=[player_col, season_col], how="left")


def add_games_played_features(
    df: pl.DataFrame,
    *,
    group_keys: list[str] | None = None,
    week_col: str = "week",
) -> pl.DataFrame:
    """Count prior games played in-season (lagged).

    Also emits ``prior_season_games_played`` -- the same count carried forward
    from the completed prior season.  The two are deliberately separate: the
    in-season index says how far into the year a row sits, while the
    prior-season count is a durability signal.  Collapsing them makes a rookie
    look like a week-1 veteran.
    """
    group_keys = group_keys or ["gsis_id", "season"]
    out = df.sort(group_keys + [week_col])
    out = out.with_columns(
        pl.int_range(pl.len())
        .over(group_keys)
        .alias("games_played_prior")
    )

    player_col, season_col = group_keys[0], group_keys[1]
    prior = (
        out.group_by([player_col, season_col])
        .agg(pl.col(week_col).n_unique().cast(pl.Float64).alias("prior_season_games_played"))
        .with_columns((pl.col(season_col) + 1).alias(season_col))
    )
    return out.join(prior, on=[player_col, season_col], how="left")


def shrink_rolling_with_prior(
    df: pl.DataFrame,
    cols: list[str],
    *,
    windows: tuple[int, ...] = (3, 5),
    prior_strength: float = 3.0,
) -> pl.DataFrame:
    """Create shrunk versions of rolling means toward prior-season mean."""
    exprs = []
    for col in cols:
        prior_col = f"{col}_prior_season"
        if prior_col not in df.columns:
            continue
        for w in windows:
            roll_col = f"{col}_l{w}"
            if roll_col not in df.columns:
                continue
            n = pl.col("games_played_prior").clip(0, w)
            exprs.append(
                shrink_toward(
                    pl.col(roll_col),
                    pl.col(prior_col).fill_null(pl.col(roll_col)),
                    n,
                    prior_strength=prior_strength,
                ).alias(f"{col}_l{w}_shrunk")
            )
        season_col = f"{col}_season_td"
        if season_col in df.columns:
            exprs.append(
                shrink_toward(
                    pl.col(season_col),
                    pl.col(prior_col).fill_null(pl.col(season_col)),
                    pl.col("games_played_prior"),
                    prior_strength=prior_strength,
                ).alias(f"{col}_season_td_shrunk")
            )
    return df.with_columns(exprs) if exprs else df
