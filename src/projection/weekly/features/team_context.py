"""Team / game context features from schedules and team stats."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl

from src.projection.weekly.features.leakage import pregame_schedule_columns


def explode_schedules_to_team_weeks(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team-week with implied points and rest."""
    needed = {"season", "week", "home_team", "away_team", "spread_line", "total_line"}
    missing = needed - set(schedules.columns)
    if missing:
        raise ValueError(f"schedules missing columns: {missing}")

    base_cols = [c for c in pregame_schedule_columns() if c in schedules.columns]
    extra = ["season", "week", "game_id"] if "game_id" in schedules.columns else ["season", "week"]
    sched = schedules.select(list(dict.fromkeys(extra + base_cols + ["home_team", "away_team"])))

    home = sched.with_columns(
        [
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.lit(1).alias("is_home"),
            (pl.col("total_line") / 2.0 - pl.col("spread_line") / 2.0).alias("implied_team_total"),
            (pl.col("total_line") / 2.0 + pl.col("spread_line") / 2.0).alias("implied_opp_total"),
            (pl.col("home_rest") if "home_rest" in sched.columns else pl.lit(None)).alias("rest_days"),
        ]
    )
    away = sched.with_columns(
        [
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.lit(0).alias("is_home"),
            (pl.col("total_line") / 2.0 + pl.col("spread_line") / 2.0).alias("implied_team_total"),
            (pl.col("total_line") / 2.0 - pl.col("spread_line") / 2.0).alias("implied_opp_total"),
            (pl.col("away_rest") if "away_rest" in sched.columns else pl.lit(None)).alias("rest_days"),
        ]
    )
    keep = [
        "season",
        "week",
        "team",
        "opponent",
        "is_home",
        "implied_team_total",
        "implied_opp_total",
        "rest_days",
        "spread_line",
        "total_line",
    ]
    for optional in ("roof", "surface", "temp", "wind", "game_id"):
        if optional in sched.columns:
            keep.append(optional)

    return pl.concat([home.select(keep), away.select(keep)], how="vertical_relaxed")


def add_team_pass_rate(
    team_weeks: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Attach lagged team pass rate (attempts / (attempts + carries))."""
    team_col = "recent_team" if "recent_team" in player_stats.columns else "team"
    if team_col not in player_stats.columns:
        return team_weeks.with_columns(pl.lit(None).cast(pl.Float64).alias("team_pass_rate_l5"))

    attempts_col = "attempts" if "attempts" in player_stats.columns else None
    carries_col = "carries" if "carries" in player_stats.columns else None
    if not attempts_col or not carries_col:
        return team_weeks.with_columns(pl.lit(None).cast(pl.Float64).alias("team_pass_rate_l5"))

    weekly = (
        player_stats.group_by(["season", "week", team_col])
        .agg(
            [
                pl.col(attempts_col).sum().alias("team_attempts"),
                pl.col(carries_col).sum().alias("team_carries"),
            ]
        )
        .rename({team_col: "team"})
        .sort(["team", "season", "week"])
    )
    weekly = weekly.with_columns(
        (
            pl.col("team_attempts")
            / (pl.col("team_attempts") + pl.col("team_carries") + 1e-6)
        ).alias("team_pass_rate")
    )
    weekly = weekly.with_columns(
        pl.col("team_pass_rate")
        .shift(1)
        .rolling_mean(window_size=5, min_samples=1)
        .over(["team", "season"])
        .alias("team_pass_rate_l5")
    )
    return team_weeks.join(
        weekly.select(["season", "week", "team", "team_pass_rate_l5", "team_attempts", "team_carries"]),
        on=["season", "week", "team"],
        how="left",
    )


def add_prior_season_team_pass_rate(
    team_weeks: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    team_col = "recent_team" if "recent_team" in player_stats.columns else "team"
    if team_col not in player_stats.columns or "attempts" not in player_stats.columns:
        return team_weeks

    season_rates = (
        player_stats.group_by(["season", team_col])
        .agg(
            [
                pl.col("attempts").sum().alias("att"),
                pl.col("carries").sum().alias("car"),
            ]
        )
        .with_columns(
            (pl.col("att") / (pl.col("att") + pl.col("car") + 1e-6)).alias("team_pass_rate_prior_season")
        )
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({team_col: "team"})
        .select(["season", "team", "team_pass_rate_prior_season"])
    )
    return team_weeks.join(season_rates, on=["season", "team"], how="left")


def add_opponent_defense_features(
    team_weeks: pl.DataFrame,
    team_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Attach lagged defensive 'allowed' rates for the upcoming opponent.

    team_stats rows are offensive production. The defending team is
    ``opponent_team`` on that row, so we re-key and lag before joining
    panel rows on ``opponent``.
    """
    empty_cols = [
        "opp_ypa_allowed_l5",
        "opp_ypc_allowed_l5",
        "opp_ypr_allowed_l5",
        "opp_pass_epa_allowed_l5",
        "opp_rush_epa_allowed_l5",
        "opp_pass_rate_allowed_l5",
    ]
    if team_stats.is_empty() or "opponent_team" not in team_stats.columns:
        return team_weeks.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in empty_cols])

    ts = team_stats
    if "season_type" in ts.columns:
        ts = ts.filter(pl.col("season_type").is_in(["REG", "Regular", "regular"]))

    needed = {"season", "week", "opponent_team", "attempts", "passing_yards", "carries", "rushing_yards"}
    if not needed.issubset(set(ts.columns)):
        return team_weeks.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in empty_cols])

    exprs = [
        pl.col("opponent_team").alias("team"),
        pl.when(pl.col("attempts") > 0)
        .then(pl.col("passing_yards") / pl.col("attempts"))
        .otherwise(None)
        .alias("ypa_allowed"),
        pl.when(pl.col("carries") > 0)
        .then(pl.col("rushing_yards") / pl.col("carries"))
        .otherwise(None)
        .alias("ypc_allowed"),
        (
            pl.col("attempts")
            / (pl.col("attempts") + pl.col("carries") + 1e-6)
        ).alias("pass_rate_allowed"),
    ]
    if "receptions" in ts.columns and "receiving_yards" in ts.columns:
        exprs.append(
            pl.when(pl.col("receptions") > 0)
            .then(pl.col("receiving_yards") / pl.col("receptions"))
            .otherwise(None)
            .alias("ypr_allowed")
        )
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("ypr_allowed"))
    if "passing_epa" in ts.columns:
        exprs.append(pl.col("passing_epa").alias("pass_epa_allowed"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("pass_epa_allowed"))
    if "rushing_epa" in ts.columns:
        exprs.append(pl.col("rushing_epa").alias("rush_epa_allowed"))
    else:
        exprs.append(pl.lit(None).cast(pl.Float64).alias("rush_epa_allowed"))

    allowed = ts.with_columns(exprs).select(
        [
            "season",
            "week",
            "team",
            "ypa_allowed",
            "ypc_allowed",
            "ypr_allowed",
            "pass_epa_allowed",
            "rush_epa_allowed",
            "pass_rate_allowed",
        ]
    )

    # One row per defending team-week (offense of their opponent that week)
    allowed = (
        allowed.group_by(["season", "week", "team"])
        .agg(
            [
                pl.col("ypa_allowed").mean(),
                pl.col("ypc_allowed").mean(),
                pl.col("ypr_allowed").mean(),
                pl.col("pass_epa_allowed").mean(),
                pl.col("rush_epa_allowed").mean(),
                pl.col("pass_rate_allowed").mean(),
            ]
        )
        .sort(["team", "season", "week"])
    )

    lag_map = {
        "ypa_allowed": "opp_ypa_allowed_l5",
        "ypc_allowed": "opp_ypc_allowed_l5",
        "ypr_allowed": "opp_ypr_allowed_l5",
        "pass_epa_allowed": "opp_pass_epa_allowed_l5",
        "rush_epa_allowed": "opp_rush_epa_allowed_l5",
        "pass_rate_allowed": "opp_pass_rate_allowed_l5",
    }
    lag_exprs = [
        pl.col(src)
        .shift(1)
        .rolling_mean(window_size=5, min_samples=1)
        .over(["team", "season"])
        .alias(dst)
        for src, dst in lag_map.items()
    ]
    # The normal weekly features exclude the current game.  Also compute the
    # final five completed games and carry that defensive prior into the next
    # season, which is the only information available for a preseason slate.
    prior_exprs = [
        pl.col(src).drop_nulls().tail(5).mean().alias(dst)
        for src, dst in lag_map.items()
    ]
    prior_allowed = (
        allowed.group_by(["season", "team"])
        .agg(prior_exprs)
        .with_columns((pl.col("season") + 1).alias("season"))
        .rename({dst: f"__prior_{dst}" for dst in lag_map.values()})
    )

    allowed = allowed.with_columns(lag_exprs).select(
        ["season", "week", "team"] + list(lag_map.values())
    )

    # Join defense of the opponent onto each team-week
    out = team_weeks.join(
        allowed.rename({"team": "opponent"}),
        on=["season", "week", "opponent"],
        how="left",
    )
    out = out.join(
        prior_allowed.rename({"team": "opponent"}),
        on=["season", "opponent"],
        how="left",
    )
    out = out.with_columns(
        [
            pl.coalesce([pl.col(dst), pl.col(f"__prior_{dst}")]).alias(dst)
            for dst in lag_map.values()
        ]
    )
    return out.drop([f"__prior_{dst}" for dst in lag_map.values()])
