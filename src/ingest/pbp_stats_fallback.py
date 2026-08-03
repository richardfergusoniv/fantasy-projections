"""Aggregate weekly passing/rushing/receiving stat lines directly from
play-by-play, for use when nflverse hasn't yet published the pre-aggregated
player_stats file for a season (e.g. 2025 as of this writing - pbp and
participation are published for the full season, but player_stats 404s).

This intentionally covers only the stats named in the project goal
(attempts, yards, TDs, receptions, targets, interceptions) - it is NOT a
full replica of nflverse's player_stats schema (no 2pt conversions, no
fumbles, no fantasy points). Anything computed here should be clearly
distinguishable from the official release, since methodology could differ
at the margins (stat corrections, lateral plays, etc).
"""
import pandas as pd


def aggregate_weekly_stats_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    passing = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["season", "week", "passer_player_id", "passer_player_name"], dropna=True)
        .agg(
            completions=("complete_pass", "sum"),
            attempts=("pass_attempt", "sum"),
            passing_yards=("passing_yards", "sum"),
            passing_tds=("pass_touchdown", "sum"),
            interceptions=("interception", "sum"),
        )
        .reset_index()
        .rename(columns={"passer_player_id": "player_id", "passer_player_name": "player_name"})
    )

    rushing = (
        pbp[pbp["rush_attempt"] == 1]
        .groupby(["season", "week", "rusher_player_id", "rusher_player_name"], dropna=True)
        .agg(
            carries=("rush_attempt", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rush_touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name"})
    )

    receiving = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["season", "week", "receiver_player_id", "receiver_player_name"], dropna=True)
        .agg(
            targets=("pass_attempt", "sum"),
            receptions=("complete_pass", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("pass_touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name"})
    )

    merged = passing.merge(rushing, on=["season", "week", "player_id", "player_name"], how="outer").merge(
        receiving, on=["season", "week", "player_id", "player_name"], how="outer"
    )

    stat_cols = [c for c in merged.columns if c not in ("season", "week", "player_id", "player_name")]
    merged[stat_cols] = merged[stat_cols].fillna(0)
    merged["stat_source"] = "pbp_fallback"
    return merged
