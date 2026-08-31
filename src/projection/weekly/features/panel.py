"""Build the player-week training / inference panel."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.projection.weekly.config.paths import (
    DATA_DIR,
    SKILL_POSITIONS,
    TRAIN_START_SEASON,
    VALIDATE_SEASON,
    ensure_dirs,
)
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.data.ids import coerce_id_columns, coalesce_player_id, normalize_position
from src.projection.weekly.data.nflverse_loader import (
    load_contracts,
    load_depth_charts,
    load_draft_picks,
    load_ff_opportunity,
    load_ff_playerids,
    load_ftn_charting,
    load_nextgen_stats,
    load_participation,
    load_players,
    load_player_stats,
    load_rosters,
    load_rosters_weekly,
    load_schedules,
    load_snap_counts,
    load_team_stats,
)
from src.projection.weekly.features.advanced_public import (
    NGS_VALUE_COLS,
    PARTICIPATION_VALUE_COLS,
    attach_ftn_team_features,
    attach_nextgen_features,
    attach_participation_features,
    attach_weekly_roster_features,
)
from src.projection.weekly.features.contracts import CONTRACT_FEATURE_COLS, attach_contract_features
from src.projection.weekly.features.depth import attach_depth_features
from src.projection.weekly.features.injuries import attach_injury_features
from src.projection.weekly.features.rolling import (
    add_games_played_features,
    add_prior_season_means,
    add_rolling_means,
    shrink_rolling_with_prior,
)
from src.projection.weekly.features.team_context import (
    add_opponent_defense_features,
    add_prior_season_team_pass_rate,
    add_team_pass_rate,
    explode_schedules_to_team_weeks,
)
from src.projection.weekly.features.xfp import attach_xfp_features
from src.projection.weekly.scoring.fantasy_points import compute_fantasy_points

logger = logging.getLogger(__name__)

USAGE_COLS = [
    "attempts",
    "completions",
    "carries",
    "targets",
    "receptions",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "interceptions",
    "air_yards",
    "target_share",
    "wopr",
    "fantasy_points",
]

SHARE_COLS = [
    "target_share",
    "carry_share",
    "snap_share",
    "air_yards_share",
    "redzone_target_share",
]


def _team_column(df: pl.DataFrame) -> str:
    for c in ("recent_team", "team", "posteam"):
        if c in df.columns:
            return c
    raise ValueError("No team column found in player stats")


def _ensure_stat_columns(df: pl.DataFrame) -> pl.DataFrame:
    # nflverse player stats use passing_interceptions; normalize to interceptions
    if "interceptions" not in df.columns and "passing_interceptions" in df.columns:
        df = df.with_columns(pl.col("passing_interceptions").alias("interceptions"))
    defaults = {
        "attempts": 0.0,
        "completions": 0.0,
        "carries": 0.0,
        "targets": 0.0,
        "receptions": 0.0,
        "passing_yards": 0.0,
        "rushing_yards": 0.0,
        "receiving_yards": 0.0,
        "passing_tds": 0.0,
        "rushing_tds": 0.0,
        "receiving_tds": 0.0,
        "interceptions": 0.0,
        "air_yards": None,
        "target_share": None,
        "wopr": None,
        "racr": None,
        "passing_epa": None,
        "rushing_epa": None,
        "receiving_epa": None,
        "fumbles_lost": 0.0,
        "sacks": 0.0,
        "sack_yards": 0.0,
        "passing_air_yards": None,
        "receiving_air_yards": None,
    }
    exprs = []
    for col, default in defaults.items():
        if col not in df.columns:
            if default is None:
                exprs.append(pl.lit(None).cast(pl.Float64).alias(col))
            else:
                exprs.append(pl.lit(float(default)).alias(col))
    return df.with_columns(exprs) if exprs else df


def _attach_snaps(player_stats: pl.DataFrame, snaps: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    if snaps.is_empty():
        return player_stats.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("offense_snaps"),
                pl.lit(None).cast(pl.Float64).alias("snap_share"),
            ]
        )

    snap_df = snaps
    # Normalize snap percent column
    pct_col = next(
        (c for c in ("offense_pct", "offense_snaps_pct", "otot_pct") if c in snap_df.columns),
        None,
    )
    snaps_col = next((c for c in ("offense_snaps", "offense") if c in snap_df.columns), None)

    # Join path: prefer gsis_id, else pfr_id via ff ids
    if "gsis_id" in snap_df.columns and snap_df["gsis_id"].null_count() < snap_df.height:
        join_on = ["season", "week", "gsis_id"]
        snap_sel = snap_df
    else:
        pfr_col = "pfr_id" if "pfr_id" in snap_df.columns else "pfr_player_id"
        if pfr_col not in snap_df.columns or ids.is_empty() or "pfr_id" not in ids.columns:
            return player_stats.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("offense_snaps"),
                    pl.lit(None).cast(pl.Float64).alias("snap_share"),
                ]
            )
        id_map = ids.select(
            [c for c in ("gsis_id", "pfr_id") if c in ids.columns]
        ).unique(subset=["pfr_id"], keep="first")
        snap_sel = snap_df.rename({pfr_col: "pfr_id"}).join(id_map, on="pfr_id", how="left")
        join_on = ["season", "week", "gsis_id"]

    cols = ["season", "week", "gsis_id"]
    if snaps_col:
        cols.append(snaps_col)
    if pct_col:
        cols.append(pct_col)
    snap_sel = snap_sel.select([c for c in cols if c in snap_sel.columns]).unique(
        subset=["season", "week", "gsis_id"], keep="first"
    )
    rename = {}
    if snaps_col and snaps_col != "offense_snaps":
        rename[snaps_col] = "offense_snaps"
    if pct_col:
        rename[pct_col] = "snap_share"
    snap_sel = snap_sel.rename(rename) if rename else snap_sel

    out = player_stats.join(snap_sel, on=join_on, how="left")
    if "snap_share" in out.columns:
        # offense_pct is often 0-100
        out = out.with_columns(
            pl.when(pl.col("snap_share") > 1.5)
            .then(pl.col("snap_share") / 100.0)
            .otherwise(pl.col("snap_share"))
            .alias("snap_share")
        )
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("snap_share"))
    if "offense_snaps" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("offense_snaps"))
    return out


def _add_team_shares(df: pl.DataFrame) -> pl.DataFrame:
    team_col = _team_column(df)
    # Team totals for share denominators
    team_tot = df.group_by(["season", "week", team_col]).agg(
        [
            pl.col("carries").sum().alias("team_carries"),
            pl.col("targets").sum().alias("team_targets"),
            pl.col("air_yards").sum().alias("team_air_yards"),
            pl.col("attempts").sum().alias("team_attempts"),
        ]
    )
    out = df.join(team_tot, on=["season", "week", team_col], how="left")
    out = out.with_columns(
        [
            (pl.col("carries") / (pl.col("team_carries") + 1e-6)).alias("carry_share"),
            pl.when(pl.col("target_share").is_not_null())
            .then(pl.col("target_share"))
            .otherwise(pl.col("targets") / (pl.col("team_targets") + 1e-6))
            .alias("target_share"),
            (pl.col("air_yards") / (pl.col("team_air_yards").abs() + 1e-6)).alias("air_yards_share"),
            (pl.col("attempts") / (pl.col("team_attempts") + 1e-6)).alias("dropback_share"),
        ]
    )
    # Red-zone proxy: receiving_tds / team receiving context is weak; use target share * RZ flag later.
    # Approximate redzone target share with receiving_tds share when RZ targets unavailable.
    rz = df.group_by(["season", "week", team_col]).agg(
        pl.col("receiving_tds").sum().alias("team_rec_tds")
    )
    out = out.join(rz, on=["season", "week", team_col], how="left")
    out = out.with_columns(
        (pl.col("receiving_tds") / (pl.col("team_rec_tds") + 1e-6)).alias("redzone_target_share")
    )
    return out


def _add_efficiency_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Per-play rates; null when the denominator is 0 to avoid 1e-6 explosions."""

    def _rate(num: str, den: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > 0)
            .then(pl.col(num) / pl.col(den))
            .otherwise(None)
            .cast(pl.Float64)
        )

    return df.with_columns(
        [
            _rate("passing_yards", "attempts").alias("ypa"),
            _rate("rushing_yards", "carries").alias("ypc"),
            _rate("receiving_yards", "receptions").alias("ypr"),
            _rate("receptions", "targets").alias("catch_rate"),
            _rate("passing_tds", "attempts").alias("pass_td_rate"),
            _rate("interceptions", "attempts").alias("int_rate"),
            _rate("rushing_tds", "carries").alias("rush_td_rate"),
            _rate("receiving_tds", "targets").alias("rec_td_rate"),
            _rate("rushing_yards", "carries").alias("rush_ypa"),
            _rate("passing_epa", "attempts").alias("passing_epa_per_play"),
        ]
    )


def _add_draft_rookie_flags(df: pl.DataFrame, draft: pl.DataFrame) -> pl.DataFrame:
    if draft.is_empty():
        return df.with_columns(
            [
                pl.lit(0).alias("is_rookie"),
                pl.lit(None).cast(pl.Int64).alias("draft_pick"),
                pl.lit(None).cast(pl.Int64).alias("draft_round"),
            ]
        )
    d = draft
    if "pick" in d.columns and "draft_pick" not in d.columns:
        d = d.rename({"pick": "draft_pick"}) if "draft_pick" not in d.columns else d
    pick_col = "draft_pick" if "draft_pick" in d.columns else ("pick" if "pick" in d.columns else None)
    round_col = "round" if "round" in d.columns else None
    season_col = "season" if "season" in d.columns else None
    id_col = "gsis_id" if "gsis_id" in d.columns else None
    if not id_col or not season_col:
        return df.with_columns(
            [
                pl.lit(0).alias("is_rookie"),
                pl.lit(None).cast(pl.Int64).alias("draft_pick"),
                pl.lit(None).cast(pl.Int64).alias("draft_round"),
            ]
        )
    cols = [id_col, season_col]
    if pick_col:
        cols.append(pick_col)
    if round_col:
        cols.append(round_col)
    d = d.select(cols).unique(subset=[id_col, season_col], keep="first")
    rename = {season_col: "rookie_season"}
    if pick_col and pick_col != "draft_pick":
        rename[pick_col] = "draft_pick"
    if round_col:
        rename[round_col] = "draft_round"
    d = d.rename(rename)
    out = df.join(d, left_on=["gsis_id"], right_on=[id_col], how="left")
    out = out.with_columns(
        (pl.col("season") == pl.col("rookie_season")).fill_null(False).cast(pl.Int8).alias("is_rookie")
    )
    if "draft_pick" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Int64).alias("draft_pick"))
    if "draft_round" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Int64).alias("draft_round"))
    return out


def build_player_week_panel(
    *,
    seasons: list[int] | None = None,
    scoring: ScoringConfig | None = None,
    force_reload: bool = False,
) -> pl.DataFrame:
    """Construct leakage-safe player-week panel with labels and lagged features."""
    scoring = scoring or ScoringConfig()
    seasons = seasons or list(range(TRAIN_START_SEASON, VALIDATE_SEASON + 1))
    ensure_dirs()

    logger.info("Loading player stats for seasons %s", seasons)
    stats = load_player_stats(seasons, force=force_reload)
    stats = coerce_id_columns(stats)
    stats = coalesce_player_id(stats)
    stats = normalize_position(stats)
    stats = _ensure_stat_columns(stats)

    # air_yards: prefer receiving_air_yards for skill, passing_air_yards for QB
    air_parts = []
    if "receiving_air_yards" in stats.columns:
        air_parts.append(pl.col("receiving_air_yards"))
    if "passing_air_yards" in stats.columns:
        air_parts.append(pl.col("passing_air_yards"))
    if "air_yards" in stats.columns:
        air_parts.append(pl.col("air_yards"))
    if air_parts:
        stats = stats.with_columns(pl.coalesce(air_parts).alias("air_yards"))

    stats = compute_fantasy_points(stats, scoring, alias="fantasy_points")
    stats = stats.filter(pl.col("position").is_in(list(SKILL_POSITIONS)))

    # Player name
    name_col = next(
        (c for c in ("player_display_name", "player_name", "player") if c in stats.columns),
        None,
    )
    if name_col and name_col != "player_name":
        stats = stats.with_columns(pl.col(name_col).alias("player_name"))
    elif "player_name" not in stats.columns:
        stats = stats.with_columns(pl.lit("Unknown").alias("player_name"))

    team_col = _team_column(stats)
    if team_col != "team":
        stats = stats.with_columns(pl.col(team_col).alias("team"))

    snaps = load_snap_counts(seasons, force=force_reload)
    ids = load_ff_playerids(force=force_reload)
    stats = _attach_snaps(stats, snaps, ids)
    stats = _add_team_shares(stats)
    stats = _add_efficiency_labels(stats)

    draft = load_draft_picks(seasons, force=force_reload)
    stats = _add_draft_rookie_flags(stats, draft)

    # Optional public advanced feeds. Current-week player aggregates are only
    # labels here; they become usable predictors through the lagged rolling
    # transforms below. Roster and FTN functions shift internally.
    participation = pl.DataFrame()
    try:
        participation = load_participation(seasons, force=force_reload)
        stats = attach_participation_features(stats, participation)
    except Exception as exc:
        logger.warning("Participation features skipped: %s", exc)
        stats = attach_participation_features(stats, pl.DataFrame())

    try:
        ngs = {
            kind: load_nextgen_stats(seasons, stat_type=kind, force=force_reload)
            for kind in ("passing", "receiving", "rushing")
        }
        stats = attach_nextgen_features(stats, ngs)
    except Exception as exc:
        logger.warning("Next Gen Stats features skipped: %s", exc)
        stats = attach_nextgen_features(stats, {})

    try:
        weekly_rosters = load_rosters_weekly(seasons, force=force_reload)
        stats = attach_weekly_roster_features(stats, weekly_rosters)
    except Exception as exc:
        logger.warning("Weekly roster features skipped: %s", exc)
        stats = attach_weekly_roster_features(stats, pl.DataFrame())

    try:
        ftn = load_ftn_charting(seasons, force=force_reload)
        stats = attach_ftn_team_features(stats, ftn, participation=participation)
    except Exception as exc:
        logger.warning("FTN charting features skipped: %s", exc)
        stats = attach_ftn_team_features(stats, pl.DataFrame())

    # OverTheCap contracts (nflverse) — as-of by season
    try:
        contracts = load_contracts(force=force_reload)
        stats = attach_contract_features(stats, contracts)
    except Exception as exc:
        logger.warning("Contract features skipped: %s", exc)
        stats = stats.with_columns(
            [pl.lit(None).cast(pl.Float64).alias(c) for c in CONTRACT_FEATURE_COLS if c not in stats.columns]
        )

    # Injury status — nflverse for completed seasons; ESPN only for live season
    from datetime import date

    # Upcoming/current NFL season year (Aug 2026 → 2026 so 2025 stays historical)
    calendar_live = date.today().year
    stats = attach_injury_features(
        stats,
        force_reload=force_reload,
        ids=ids,
        live_season=calendar_live,
    )

    # Team context from schedules + opponent defense (lagged)
    schedules = load_schedules(seasons, force=force_reload)
    team_weeks = explode_schedules_to_team_weeks(schedules)
    team_weeks = add_team_pass_rate(team_weeks, stats)
    team_weeks = add_prior_season_team_pass_rate(team_weeks, stats)
    try:
        team_stats = load_team_stats(seasons, force=force_reload)
        team_weeks = add_opponent_defense_features(team_weeks, team_stats)
    except Exception as exc:
        logger.warning("Opponent defense features skipped: %s", exc)

    stats = stats.join(team_weeks, on=["season", "week", "team"], how="left")

    # Depth charts (weekly <=2024; as-of snapshots 2025+)
    try:
        depth = load_depth_charts(seasons, force=force_reload)
        stats = attach_depth_features(stats, depth, schedules=schedules)
    except Exception as exc:
        logger.warning("Depth chart features skipped: %s", exc)
        stats = stats.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("depth_rank"),
                pl.lit(None).cast(pl.Int8).alias("is_listed_starter"),
                pl.lit(None).cast(pl.Int64).alias("same_pos_depth_count"),
            ]
        )

    # ffopportunity xFP (joined then lagged below)
    try:
        xfp = load_ff_opportunity(seasons, force=force_reload)
        stats = attach_xfp_features(stats, xfp)
    except Exception as exc:
        logger.warning("xFP features skipped: %s", exc)

    # Rolling + prior-season features (labels remain current-week values)
    value_cols = list(
        dict.fromkeys(
            c
            for c in USAGE_COLS
            + [
                "carry_share",
                "snap_share",
                "air_yards_share",
                "dropback_share",
                "redzone_target_share",
                "ypa",
                "ypc",
                "ypr",
                "catch_rate",
                "pass_td_rate",
                "int_rate",
                "rush_td_rate",
                "rec_td_rate",
                "offense_snaps",
                "wopr",
                "racr",
                "passing_epa",
                "receiving_epa",
                "rushing_epa",
                "passing_epa_per_play",
                "xfp",
                "fp_minus_xfp",
                "rec_yards_oe",
                "rush_yards_oe",
                *PARTICIPATION_VALUE_COLS,
                *NGS_VALUE_COLS,
            ]
            if c in stats.columns
        )
    )

    stats = add_prior_season_means(stats, value_cols)
    stats = add_games_played_features(stats)
    stats = add_rolling_means(stats, value_cols)
    stats = shrink_rolling_with_prior(stats, value_cols)

    # Age from birth_date (rosters / players)
    stats = _attach_age(stats, seasons=seasons, force_reload=force_reload)

    # Sort and return
    stats = stats.sort(["season", "week", "position", "fantasy_points"], descending=[False, False, False, True])
    logger.info("Built player-week panel: %d rows, %d columns", stats.height, len(stats.columns))
    return stats


def _attach_age(
    stats: pl.DataFrame,
    *,
    seasons: list[int],
    force_reload: bool = False,
) -> pl.DataFrame:
    """Populate age from roster/player birth_date at season start (approx Sept 1)."""
    birth = None
    try:
        rost = load_rosters(seasons, force=force_reload)
        if not rost.is_empty() and "birth_date" in rost.columns and "gsis_id" in rost.columns:
            birth = (
                rost.select(["gsis_id", "birth_date"])
                .filter(pl.col("gsis_id").is_not_null() & pl.col("birth_date").is_not_null())
                .unique(subset=["gsis_id"], keep="last")
            )
    except Exception as exc:
        logger.warning("Roster birth_date unavailable: %s", exc)
    if birth is None or birth.is_empty():
        try:
            players = load_players(force=force_reload)
            if not players.is_empty() and "birth_date" in players.columns and "gsis_id" in players.columns:
                birth = (
                    players.select(["gsis_id", "birth_date"])
                    .filter(pl.col("gsis_id").is_not_null() & pl.col("birth_date").is_not_null())
                    .unique(subset=["gsis_id"], keep="last")
                )
        except Exception as exc:
            logger.warning("Players birth_date unavailable: %s", exc)

    if birth is None or birth.is_empty():
        if "age" not in stats.columns:
            return stats.with_columns(pl.lit(None).cast(pl.Float64).alias("age"))
        return stats

    out = stats.join(birth, on="gsis_id", how="left")
    # Age at approx season start: Sept 1 of season year
    bd = pl.col("birth_date")
    if out.schema.get("birth_date") in (pl.Utf8, pl.String):
        bd = pl.col("birth_date").str.to_date(strict=False)
    age_expr = (
        (pl.date(pl.col("season"), 9, 1) - bd).dt.total_days() / 365.25
    ).cast(pl.Float64)
    if "age" in out.columns:
        age_expr = pl.coalesce([pl.col("age").cast(pl.Float64, strict=False), age_expr])
    out = out.with_columns(age_expr.alias("age"))
    if "birth_date" in out.columns:
        out = out.drop("birth_date")
    return out


def save_panel(df: pl.DataFrame, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (DATA_DIR / "processed" / "player_week_panel.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Wrote panel -> %s", path)
    return path


def load_panel(path: Path | None = None) -> pl.DataFrame:
    path = path or (DATA_DIR / "processed" / "player_week_panel.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Panel not found at {path}. Run scripts/build_features.py first.")
    return pl.read_parquet(path)
