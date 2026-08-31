"""Preseason player availability estimates.

The weekly projection is conditional on a player being available.  Draft value
needs an additional estimate of games played; multiplying every player by 17
systematically overvalues fragile veterans and players with unresolved roster
statuses.  This module provides a leakage-safe empirical-Bayes estimate using
only seasons before the target year.
"""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import polars as pl


DEFAULT_GAMES_PRIOR = 15.0
STARTING_QB_GAMES_FLOOR = 15.0
RECENCY_WEIGHTS = {1: 0.60, 2: 0.30, 3: 0.10}
STATUS_GAME_CAPS = {
    "RES": 12.0,
    "PUP": 12.0,
    "NFI": 12.0,
    "SUS": 13.0,
    "CUT": 0.0,
    "RET": 0.0,
}


def condition_season_outlook_on_playing(
    outlook: pl.DataFrame,
    *,
    long_term_cap: float = 12.0,
    starting_qb_floor: float = STARTING_QB_GAMES_FLOOR,
) -> pl.DataFrame:
    """Use current absence news once, through projected games.

    Weekly forecasts are conditional on playing. Freezing today's IR/PUP/Out
    flag over every future week would hard-zero the player and then apply the
    games-played haircut a second time.
    """
    if outlook.is_empty():
        return outlook
    out = outlook
    long_term = pl.lit(False)
    if "sleeper_is_ir" in out.columns:
        long_term = long_term | pl.col("sleeper_is_ir").fill_null(False)
    if "injury_status" in out.columns:
        long_term = long_term | (
            pl.col("injury_status")
            .cast(pl.Utf8)
            .fill_null("")
            .str.to_lowercase()
            .str.contains("injured reserve|pup|physically unable|nfi")
        )
    if "projected_games_estimate" in out.columns:
        current_qb_starter = pl.lit(False)
        if "position" in out.columns and "depth_rank" in out.columns:
            current_qb_starter = (
                (pl.col("position") == "QB")
                & (pl.col("depth_rank").cast(pl.Float64).fill_null(99.0) <= 1.0)
            )
        out = out.with_columns(
            pl.when(long_term)
            .then(
                pl.min_horizontal(
                    pl.col("projected_games_estimate").fill_null(long_term_cap),
                    pl.lit(float(long_term_cap)),
                )
            )
            # QB appearance history is role-confounded: a healthy backup can
            # record zero panel games for years, then become the current QB1.
            # Once the depth chart names him starter, use the normal healthy
            # prior as a floor instead of treating backup DNPs as injuries.
            .when(current_qb_starter)
            .then(
                pl.max_horizontal(
                    pl.col("projected_games_estimate").fill_null(starting_qb_floor),
                    pl.lit(float(starting_qb_floor)),
                )
            )
            .otherwise(pl.col("projected_games_estimate"))
            .alias("projected_games_estimate")
        )
    resets: list[pl.Expr] = []
    if "is_out" in out.columns:
        resets.append(pl.lit(False).alias("is_out"))
    if "play_prob" in out.columns:
        resets.append(pl.lit(1.0).alias("play_prob"))
    if "available" in out.columns:
        resets.append(pl.lit(True).alias("available"))
    return out.with_columns(resets) if resets else out


def estimate_projected_games(
    panel: pl.DataFrame,
    players: pl.DataFrame,
    *,
    target_season: int,
    roster: pl.DataFrame | None = None,
    prior_games: float = DEFAULT_GAMES_PRIOR,
) -> pl.DataFrame:
    """Return one projected-games estimate per player.

    Appearance counts come from the three completed seasons preceding
    ``target_season``.  A season the player spent entirely on the sideline
    leaves no panel rows, so missed seasons after a player's debut are imputed
    as zero-game seasons carrying full recency weight; without that the
    estimate is non-monotonic in games missed and a lost season scores better
    than a half-played one.  Seasons before the debut stay excluded so a
    second-year player is not charged for a year they could not have played.
    The weighted rate is shrunk by one season of evidence toward a conservative
    league prior.  A current long-term roster status may cap the result, while
    short-term questionable designations are deliberately excluded from a
    full-season estimate.
    """
    if players.is_empty() or "gsis_id" not in players.columns:
        return pl.DataFrame(
            schema={"gsis_id": pl.Utf8, "projected_games_estimate": pl.Float64}
        )

    ids = players.select(pl.col("gsis_id").cast(pl.Utf8)).unique()
    completed = panel.filter(
        (pl.col("season") < target_season) & pl.col("gsis_id").is_not_null()
    )
    hist = completed.filter(pl.col("season") >= target_season - 3)
    if hist.is_empty():
        estimates = ids.with_columns(
            pl.lit(float(prior_games)).alias("projected_games_estimate")
        )
    else:
        # A season missed end to end leaves no panel rows at all.  Enumerate
        # the window explicitly so those seasons enter as zero-game evidence
        # instead of silently shrinking evidence_weight toward the prior.
        played = hist.group_by(["gsis_id", "season"]).agg(
            pl.col("week").n_unique().cast(pl.Float64).alias("games")
        )
        debut = completed.group_by("gsis_id").agg(
            pl.col("season").min().cast(pl.Int64).alias("debut_season")
        )
        window = pl.DataFrame(
            {"lag": pl.Series(sorted(RECENCY_WEIGHTS), dtype=pl.Int64)}
        ).with_columns(
            (pl.lit(int(target_season)) - pl.col("lag")).alias("season")
        )
        appearances = (
            ids.join(debut, on="gsis_id", how="inner")
            .join(window, how="cross")
            # Seasons before a player's debut are absence of opportunity, not
            # absence of durability, so they stay out of the evidence pool.
            .filter(pl.col("season") >= pl.col("debut_season"))
            .join(played, on=["gsis_id", "season"], how="left")
            .with_columns(
                [
                    pl.col("games").fill_null(0.0),
                    pl.col("lag")
                    .replace_strict(RECENCY_WEIGHTS, default=0.0)
                    .cast(pl.Float64)
                    .alias("weight"),
                ]
            )
        )
        rates = appearances.group_by("gsis_id").agg(
            [
                (pl.col("games") * pl.col("weight")).sum().alias("weighted_games"),
                pl.col("weight").sum().alias("evidence_weight"),
            ]
        )
        estimates = ids.join(rates, on="gsis_id", how="left").with_columns(
            (
                (
                    pl.col("weighted_games").fill_null(0.0)
                    + float(prior_games)
                )
                / (pl.col("evidence_weight").fill_null(0.0) + 1.0)
            )
            .clip(0.0, 17.0)
            .alias("projected_games_estimate")
        )

    if roster is not None and not roster.is_empty() and "status" in roster.columns:
        status = (
            roster.select(
                [
                    pl.col("gsis_id").cast(pl.Utf8),
                    pl.col("status").cast(pl.Utf8).str.to_uppercase().alias("roster_status"),
                ]
            )
            .filter(pl.col("gsis_id").is_not_null())
            .unique(subset=["gsis_id"], keep="first")
            .with_columns(
                pl.col("roster_status")
                .replace_strict(STATUS_GAME_CAPS, default=17.0)
                .cast(pl.Float64)
                .alias("status_games_cap")
            )
        )
        estimates = estimates.join(status, on="gsis_id", how="left").with_columns(
            pl.min_horizontal(
                pl.col("projected_games_estimate"),
                pl.col("status_games_cap").fill_null(17.0),
            ).alias("projected_games_estimate")
        )

    return estimates.select(
        ["gsis_id", pl.col("projected_games_estimate").round(2)]
    )
