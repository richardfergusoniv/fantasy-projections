"""Leakage-safe features from optional public nflverse datasets.

Most nflverse data is CC-BY 4.0. Participation from 2023 onward and all FTN
charting are CC-BY-SA 4.0; derived artifacts must attribute FTN Data via
nflverse. These functions deliberately retain current-week aggregates only as
labels for the common rolling-feature step, or shift them here before joining.
"""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

SOURCE_METADATA = {
    "participation": {
        "source": "NFL Next Gen Stats / FTN Data via nflverse",
        "coverage": "2016+; FTN from 2023 onward",
        "license": "CC-BY-SA-4.0",
    },
    "nextgen_stats": {
        "source": "NFL Next Gen Stats via nflverse",
        "coverage": "2016+",
        "license": "CC-BY-4.0",
    },
    "rosters_weekly": {
        "source": "nflverse weekly rosters",
        "coverage": "2002+",
        "license": "CC-BY-4.0",
    },
    "ftn_charting": {
        "source": "FTN Data via nflverse",
        "coverage": "2022+",
        "license": "CC-BY-SA-4.0",
    },
}

PARTICIPATION_VALUE_COLS = [
    "offense_play_participation",
    "pass_play_participation",
]

NGS_VALUE_COLS = [
    "ngs_cpoe",
    "ngs_expected_completion_pct",
    "ngs_time_to_throw",
    "ngs_air_yards_to_sticks",
    "ngs_avg_cushion",
    "ngs_avg_separation",
    "ngs_yac_above_expectation",
    "ngs_rush_efficiency",
    "ngs_stacked_box_rate",
    "ngs_time_to_los",
    "ngs_ryoe_per_attempt",
    "ngs_rush_pct_over_expected",
]

FTN_TEAM_FEATURE_COLS = [
    "team_motion_rate_l5",
    "team_play_action_rate_l5",
    "team_screen_rate_l5",
    "team_rpo_rate_l5",
    "team_no_huddle_rate_l5",
    "team_catchable_rate_l5",
    "team_drop_rate_l5",
    "team_int_worthy_rate_l5",
    "team_qb_fault_sack_rate_l5",
]

ROSTER_FEATURE_COLS = [
    "roster_active_prev_week",
    "roster_reserve_prev_week",
    "roster_active_rate_prior",
    "roster_team_changed_prev_week",
]


def _empty_columns(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    return df.with_columns(
        [pl.lit(None).cast(pl.Float64).alias(c) for c in names if c not in df.columns]
    )


def _player_list_expr(col: str) -> pl.Expr:
    # nflverse has used semicolon-separated IDs; accepting commas/pipes makes
    # cached older variants harmless. GSIS IDs themselves contain hyphens only.
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.replace_all(r"[,|]", ";")
        .str.split(";")
    )


def attach_participation_features(
    panel: pl.DataFrame, participation: pl.DataFrame
) -> pl.DataFrame:
    """Attach player-week on-field participation aggregates.

    Values describe the current week and must be passed through the panel's
    lagged rolling/prior-season transformer before becoming model inputs.
    """
    if (
        participation.is_empty()
        or "offense_players" not in participation.columns
        or "gsis_id" not in panel.columns
    ):
        return _empty_columns(panel, PARTICIPATION_VALUE_COLS)

    p = participation
    game_col = "nflverse_game_id" if "nflverse_game_id" in p.columns else "game_id"
    if game_col not in p.columns or "play_id" not in p.columns:
        return _empty_columns(panel, PARTICIPATION_VALUE_COLS)

    if "season" not in p.columns or "week" not in p.columns:
        # nflverse game ids begin YEAR_WEEK_...
        p = p.with_columns(
            [
                pl.col(game_col).str.slice(0, 4).cast(pl.Int64, strict=False).alias("season"),
                pl.col(game_col).str.slice(5, 2).cast(pl.Int64, strict=False).alias("week"),
            ]
        )
    team_col = "possession_team" if "possession_team" in p.columns else None
    if team_col is None:
        return _empty_columns(panel, PARTICIPATION_VALUE_COLS)

    pass_expr = pl.lit(False)
    for col in ("time_to_throw", "route", "ngs_air_yards"):
        if col in p.columns:
            pass_expr = pass_expr | pl.col(col).is_not_null()
    p = p.with_columns(pass_expr.alias("_is_pass_play"))
    plays = (
        p.filter(
            pl.col("season").is_not_null()
            & pl.col("week").is_not_null()
            & pl.col(team_col).is_not_null()
        )
        .select(
            [
                "season",
                "week",
                pl.col(team_col).alias("team"),
                game_col,
                "play_id",
                "_is_pass_play",
                _player_list_expr("offense_players").alias("_players"),
            ]
        )
        .unique(subset=[game_col, "play_id"])
    )
    totals = plays.group_by(["season", "week", "team"]).agg(
        [
            pl.len().alias("_team_off_plays"),
            pl.col("_is_pass_play").sum().alias("_team_pass_plays"),
        ]
    )
    players = (
        plays.explode("_players", empty_as_null=True)
        .with_columns(pl.col("_players").str.strip_chars().alias("gsis_id"))
        .filter(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
        .group_by(["season", "week", "team", "gsis_id"])
        .agg(
            [
                pl.len().alias("_player_off_plays"),
                pl.col("_is_pass_play").sum().alias("_player_pass_plays"),
            ]
        )
        .join(totals, on=["season", "week", "team"], how="left")
        .with_columns(
            [
                (pl.col("_player_off_plays") / pl.col("_team_off_plays"))
                .cast(pl.Float64)
                .alias("offense_play_participation"),
                pl.when(pl.col("_team_pass_plays") > 0)
                .then(pl.col("_player_pass_plays") / pl.col("_team_pass_plays"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("pass_play_participation"),
            ]
        )
        .select(["season", "week", "team", "gsis_id"] + PARTICIPATION_VALUE_COLS)
    )
    return panel.join(players, on=["season", "week", "team", "gsis_id"], how="left")


def attach_nextgen_features(
    panel: pl.DataFrame, frames: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Attach normalized current-week NGS metrics for later lagging."""
    mappings = {
        "passing": {
            "completion_percentage_above_expectation": "ngs_cpoe",
            "expected_completion_percentage": "ngs_expected_completion_pct",
            "avg_time_to_throw": "ngs_time_to_throw",
            "time_to_throw": "ngs_time_to_throw",
            "avg_air_yards_to_sticks": "ngs_air_yards_to_sticks",
        },
        "receiving": {
            "avg_cushion": "ngs_avg_cushion",
            "avg_separation": "ngs_avg_separation",
            "avg_yac_above_expectation": "ngs_yac_above_expectation",
        },
        "rushing": {
            "efficiency": "ngs_rush_efficiency",
            "percent_attempts_gte_eight_defenders": "ngs_stacked_box_rate",
            "avg_time_to_los": "ngs_time_to_los",
            "rush_yards_over_expected_per_att": "ngs_ryoe_per_attempt",
            "rush_pct_over_expected": "ngs_rush_pct_over_expected",
        },
    }
    out = panel
    for stat_type, mapping in mappings.items():
        raw = frames.get(stat_type, pl.DataFrame())
        if raw.is_empty():
            continue
        df = raw
        if "player_gsis_id" in df.columns and "gsis_id" not in df.columns:
            df = df.rename({"player_gsis_id": "gsis_id"})
        if not {"season", "week", "gsis_id"}.issubset(df.columns):
            continue
        df = df.filter(pl.col("week") > 0)  # week 0 is the NGS season summary
        present = {src: dst for src, dst in mapping.items() if src in df.columns}
        # Prefer the first alias if a release contains both time-to-throw names.
        destinations: set[str] = set()
        select = ["season", "week", "gsis_id"]
        for src, dst in present.items():
            if dst not in destinations:
                select.append(pl.col(src).cast(pl.Float64, strict=False).alias(dst))
                destinations.add(dst)
        if len(select) == 3:
            continue
        feats = df.select(select).unique(
            subset=["season", "week", "gsis_id"], keep="last"
        )
        overlap = [c for c in destinations if c in out.columns]
        if overlap:
            out = out.drop(overlap)
        out = out.join(feats, on=["season", "week", "gsis_id"], how="left")
    return _empty_columns(out, NGS_VALUE_COLS)


def attach_weekly_roster_features(
    panel: pl.DataFrame, rosters: pl.DataFrame
) -> pl.DataFrame:
    """Attach status features using only roster records before the target week."""
    if rosters.is_empty() or not {"season", "week", "gsis_id"}.issubset(rosters.columns):
        return _empty_columns(panel, ROSTER_FEATURE_COLS)
    status_col = "status" if "status" in rosters.columns else "status_description_abbr"
    if status_col not in rosters.columns:
        return _empty_columns(panel, ROSTER_FEATURE_COLS)
    team_col = "team" if "team" in rosters.columns else None
    if team_col is None:
        return _empty_columns(panel, ROSTER_FEATURE_COLS)

    s = pl.col(status_col).cast(pl.Utf8).str.to_uppercase().fill_null("")
    weekly = (
        rosters.filter(pl.col("gsis_id").is_not_null() & pl.col("week").is_not_null())
        .with_columns(
            [
                s.is_in(["ACT", "ACTIVE"]).cast(pl.Float64).alias("_active"),
                (
                    s.str.contains("RES")
                    | s.str.contains("IR")
                    | s.str.contains("INJ")
                    | s.str.contains("PUP")
                )
                .cast(pl.Float64)
                .alias("_reserve"),
            ]
        )
        .sort(["gsis_id", "season", "week"])
        .unique(subset=["gsis_id", "season", "week"], keep="last")
        .sort(["gsis_id", "season", "week"])
        .with_columns(
            [
                pl.col("_active").shift(1).over(["gsis_id", "season"]).alias("roster_active_prev_week"),
                pl.col("_reserve").shift(1).over(["gsis_id", "season"]).alias("roster_reserve_prev_week"),
                (
                    pl.col(team_col).shift(1).over("gsis_id")
                    != pl.col(team_col).shift(2).over("gsis_id")
                )
                .fill_null(False)
                .cast(pl.Float64)
                .alias("roster_team_changed_prev_week"),
            ]
        )
        .with_columns(
            (
                pl.col("roster_active_prev_week")
                .cum_sum()
                .over(["gsis_id", "season"])
                / pl.col("roster_active_prev_week")
                .is_not_null()
                .cast(pl.Int64)
                .cum_sum()
                .over(["gsis_id", "season"])
            ).alias("roster_active_rate_prior")
        )
        .select(["season", "week", "gsis_id"] + ROSTER_FEATURE_COLS)
    )
    return panel.join(weekly, on=["season", "week", "gsis_id"], how="left")


def attach_ftn_team_features(
    panel: pl.DataFrame,
    charting: pl.DataFrame,
    *,
    participation: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach prior-five-game team scheme/charting rates (never current week)."""
    if charting.is_empty() or not {"season", "week"}.issubset(charting.columns):
        return _empty_columns(panel, FTN_TEAM_FEATURE_COLS)
    team_col = next(
        (c for c in ("possession_team", "posteam", "offense_team", "team") if c in charting.columns),
        None,
    )
    if team_col is None:
        # FTN releases omit offense team. Participation has the same nflverse
        # game/play keys and possession team, so use that public table as the
        # deterministic crosswalk without downloading the much larger PBP set.
        part = participation if participation is not None else pl.DataFrame()
        part_play = "play_id" if "play_id" in part.columns else "nflverse_play_id"
        chart_play = (
            "nflverse_play_id" if "nflverse_play_id" in charting.columns else "play_id"
        )
        if (
            part.is_empty()
            or "nflverse_game_id" not in part.columns
            or "nflverse_game_id" not in charting.columns
            or part_play not in part.columns
            or chart_play not in charting.columns
            or "possession_team" not in part.columns
        ):
            logger.info("FTN charting lacks offense team and participation crosswalk")
            return _empty_columns(panel, FTN_TEAM_FEATURE_COLS)
        crosswalk = (
            part.select(
                [
                    "nflverse_game_id",
                    pl.col(part_play)
                    .cast(pl.Int64, strict=False)
                    .alias("_join_play_id"),
                    pl.col("possession_team").alias("_ftn_team"),
                ]
            )
            .filter(pl.col("_ftn_team").is_not_null())
            .unique(subset=["nflverse_game_id", "_join_play_id"], keep="last")
        )
        charting = charting.with_columns(
            pl.col(chart_play).cast(pl.Int64, strict=False).alias("_join_play_id")
        ).join(crosswalk, on=["nflverse_game_id", "_join_play_id"], how="left")
        team_col = "_ftn_team"

    field_map = {
        "is_motion": "team_motion_rate",
        "is_play_action": "team_play_action_rate",
        "is_screen_pass": "team_screen_rate",
        "is_rpo": "team_rpo_rate",
        "is_no_huddle": "team_no_huddle_rate",
        "is_catchable_ball": "team_catchable_rate",
        "is_drop": "team_drop_rate",
        "is_interception_worthy": "team_int_worthy_rate",
        "is_qb_fault_sack": "team_qb_fault_sack_rate",
    }
    present = {src: dst for src, dst in field_map.items() if src in charting.columns}
    if not present:
        return _empty_columns(panel, FTN_TEAM_FEATURE_COLS)
    weekly = (
        charting.group_by(["season", "week", team_col])
        .agg(
            [
                pl.col(src).cast(pl.Float64, strict=False).mean().alias(dst)
                for src, dst in present.items()
            ]
        )
        .rename({team_col: "team"})
        .sort(["team", "season", "week"])
    )
    weekly = weekly.with_columns(
        [
            pl.col(dst)
            .shift(1)
            .rolling_mean(window_size=5, min_samples=1)
            .over(["team", "season"])
            .alias(f"{dst}_l5")
            for dst in present.values()
        ]
    ).select(["season", "week", "team"] + [f"{d}_l5" for d in present.values()])
    return _empty_columns(
        panel.join(weekly, on=["season", "week", "team"], how="left"),
        FTN_TEAM_FEATURE_COLS,
    )
