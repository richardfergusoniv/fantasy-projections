"""Season-level projection aggregation for draft boards."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from src.projection.weekly.config.paths import OUTPUTS_DIR, TRAIN_END_SEASON, TRAIN_START_SEASON, ensure_dirs
from src.projection.weekly.config.scoring import ScoringConfig
from src.projection.weekly.data.nflverse_loader import (
    load_combine,
    load_depth_charts,
    load_draft_picks,
    load_rosters,
    load_schedules,
)
from src.projection.weekly.data.teams import normalize_team_column
from src.projection.weekly.features.depth import attach_depth_features
from src.projection.weekly.features.effective_depth import (
    attach_effective_depth,
    build_effective_depth,
    clear_short_term_injuries,
)
from src.projection.weekly.features.team_context import (
    add_opponent_defense_features,
    add_prior_season_team_pass_rate,
    add_team_pass_rate,
    explode_schedules_to_team_weeks,
)
from src.projection.weekly.pipeline.accounting import depth_share_weight
from src.projection.weekly.pipeline.availability import (
    condition_season_outlook_on_playing,
    estimate_projected_games,
)
from src.projection.weekly.pipeline.rookie_projector import project_week_with_rookies
from src.projection.weekly.models.rookie import (
    attach_target_rookie_features,
    build_target_vacated_opportunity,
)

logger = logging.getLogger(__name__)

STAT_MEAN_COLS = [
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
]

SCHEDULE_CTX_COLS = [
    "opponent",
    "is_home",
    "implied_team_total",
    "implied_opp_total",
    "rest_days",
    "spread_line",
    "total_line",
    "team_pass_rate_l5",
    "team_pass_rate_prior_season",
    "opp_ypa_allowed_l5",
    "opp_ypc_allowed_l5",
    "opp_ypr_allowed_l5",
    "opp_pass_epa_allowed_l5",
    "opp_rush_epa_allowed_l5",
    "opp_pass_rate_allowed_l5",
]

DEPTH_COLS = ["depth_rank", "is_listed_starter", "same_pos_depth_count"]
CONTRACT_COLS = [
    "contract_apy_cap_pct",
    "contract_inflated_apy",
    "contract_guaranteed_pct",
    "contract_years",
    "contract_years_remaining",
    "contract_year_signed",
]

# Prefer active roster rows when a player appears multiple times
_ROSTER_STATUS_RANK = {
    "ACT": 0,
    "RES": 1,
    "E14": 2,
    "PUP": 2,
    "NFI": 2,
    "SUS": 3,
    "CUT": 9,
    "RET": 9,
}


def available_weeks(panel: pl.DataFrame, season: int) -> list[int]:
    if panel.is_empty() or "season" not in panel.columns:
        return []
    weeks = (
        panel.filter(pl.col("season") == season)["week"]
        .unique()
        .sort()
        .to_list()
    )
    return [int(w) for w in weeks if w is not None]


def _drop_existing(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df.drop(present) if present else df


def current_roster_teams(
    target_season: int,
    *,
    force: bool = False,
    roster_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per gsis_id with current team (and optional position) for a season."""
    rosters = roster_data if roster_data is not None else load_rosters([target_season], force=force)
    if rosters.is_empty() or "gsis_id" not in rosters.columns:
        return pl.DataFrame(schema={"gsis_id": pl.Utf8, "team": pl.Utf8})

    team_col = "team" if "team" in rosters.columns else None
    if team_col is None:
        return pl.DataFrame(schema={"gsis_id": pl.Utf8, "team": pl.Utf8})

    df = rosters.with_columns(pl.col("gsis_id").cast(pl.Utf8)).filter(
        pl.col("gsis_id").is_not_null() & pl.col(team_col).is_not_null()
    )
    if "status" in df.columns:
        df = df.filter(
            pl.col("status").is_in(["ACT", "RES", "INA", "PUP", "NFI", "SUS"])
            | pl.col("status").is_null()
        )
        rank_expr = pl.col("status").cast(pl.Utf8).replace_strict(
            _ROSTER_STATUS_RANK, default=5
        )
        df = df.with_columns(rank_expr.alias("_status_rank"))
    else:
        df = df.with_columns(pl.lit(0).alias("_status_rank"))

    sort_cols = ["_status_rank"]
    if "week" in df.columns:
        sort_cols = ["week"] + sort_cols
        df = df.sort(sort_cols, descending=[True, False])
    else:
        df = df.sort(sort_cols)

    keep = ["gsis_id", pl.col(team_col).alias("team")]
    keep.extend(
        c
        for c in (
            "position",
            "full_name",
            "rookie_year",
            "entry_year",
            "birth_date",
            "draft_number",
        )
        if c in df.columns
    )
    out = df.select(keep).unique(subset=["gsis_id"], keep="first")
    return normalize_team_column(out, "team")


def last_available_player_snapshot(prior: pl.DataFrame) -> pl.DataFrame:
    """One feature row per player: their latest week in the feature season.

    Using a single global week (e.g. week 18) drops injured/rested starters and
    leaves backups to absorb team volume in accounting.
    """
    if prior.is_empty() or "gsis_id" not in prior.columns:
        return prior
    return prior.sort(["gsis_id", "week"]).unique(subset=["gsis_id"], keep="last")


def end_of_season_player_snapshot(prior: pl.DataFrame) -> pl.DataFrame:
    """Build an as-of snapshot that includes every completed game.

    Feature rows are intentionally pregame/lagged, so copying the final row
    omits that player's final game and leaves ``*_prior_season`` values one
    year stale.  For a preseason outlook, recompute supported rolling and
    prior-season summaries through the last completed game.
    """
    snapshot = last_available_player_snapshot(prior)
    if snapshot.is_empty() or "gsis_id" not in snapshot.columns:
        return snapshot

    updates: list[pl.DataFrame] = []
    roll_specs: list[tuple[str, str, int | None]] = []
    for feature in snapshot.columns:
        if feature.endswith("_l3") and not feature.endswith("_shrunk"):
            roll_specs.append((feature, feature[:-3], 3))
        elif feature.endswith("_l5") and not feature.endswith("_shrunk"):
            roll_specs.append((feature, feature[:-3], 5))
        elif feature.endswith("_season_td") and not feature.endswith("_shrunk"):
            roll_specs.append((feature, feature[: -len("_season_td")], None))
        elif feature.endswith("_prior_season"):
            roll_specs.append((feature, feature[: -len("_prior_season")], None))

    agg_exprs: list[pl.Expr] = [pl.len().alias("_games_completed")]
    seen: set[str] = set()
    for feature, base, window in roll_specs:
        if feature in seen or base not in prior.columns:
            continue
        seen.add(feature)
        values = pl.col(base).drop_nulls()
        expr = values.mean() if window is None else values.tail(window).mean()
        agg_exprs.append(expr.alias(f"__eos_{feature}"))

    summary = prior.group_by("gsis_id").agg(agg_exprs)
    snapshot = snapshot.join(summary, on="gsis_id", how="left")
    replacements: list[pl.Expr] = []
    for feature in seen:
        helper = f"__eos_{feature}"
        replacements.append(pl.coalesce([pl.col(helper), pl.col(feature)]).alias(feature))
    # ``games_played_prior`` is an in-season week index everywhere the models
    # were trained (rolling.add_games_played_features), so prior-season game
    # counts must not be written into it -- a durable veteran and a week-17 row
    # would be indistinguishable, and a rookie's 0 would read as "week 1"
    # rather than "no NFL sample".  Durability gets its own column instead.
    replacements.append(
        pl.col("_games_completed").cast(pl.Float64).alias("prior_season_games_played")
    )
    if replacements:
        snapshot = snapshot.with_columns(replacements)

    # Recompute shrunk variants using the newly completed rolling summaries.
    shrink_exprs: list[pl.Expr] = []
    for feature in snapshot.columns:
        suffix = next(
            (s for s in ("_l3_shrunk", "_l5_shrunk", "_season_td_shrunk") if feature.endswith(s)),
            None,
        )
        if suffix is None:
            continue
        base = feature[: -len(suffix)]
        observed = f"{base}{suffix.replace('_shrunk', '')}"
        prior_col = f"{base}_prior_season"
        if observed not in snapshot.columns or prior_col not in snapshot.columns:
            continue
        max_n = 3.0 if suffix.startswith("_l3") else 5.0 if suffix.startswith("_l5") else None
        n = pl.col("_games_completed").cast(pl.Float64)
        if max_n is not None:
            n = n.clip(0.0, max_n)
        prior_expr = pl.col(prior_col).fill_null(pl.col(observed))
        shrink_exprs.append(
            (
                (n * pl.col(observed).fill_null(prior_expr) + 3.0 * prior_expr)
                / (n + 3.0)
            ).alias(feature)
        )
    if shrink_exprs:
        snapshot = snapshot.with_columns(shrink_exprs)

    helpers = [c for c in snapshot.columns if c.startswith("__eos_")]
    helpers.append("_games_completed")
    return snapshot.drop([c for c in helpers if c in snapshot.columns])


def latest_skill_depth_board(depth: pl.DataFrame) -> pl.DataFrame:
    """Latest offense skill depth rows with usable ranks (QB/RB/TE ≤2, WR ≤3)."""
    if depth.is_empty():
        return depth.head(0)

    df = depth
    if "team" in df.columns:
        df = normalize_team_column(df, "team")
    if "dt" in df.columns and df["dt"].null_count() < df.height:
        dt_max = df["dt"].drop_nulls().max()
        df = df.filter(pl.col("dt") == dt_max)
    elif "week" in df.columns and df["week"].null_count() < df.height:
        # Some nflverse depth archives have a null date but a valid week.
        # Fall back to the latest supplied week; callers can pre-filter to
        # Week 1 for a strict preseason snapshot.
        week_max = df["week"].drop_nulls().max()
        df = df.filter(pl.col("week") == week_max)
    else:
        return df.head(0)
    if "pos_grp" in df.columns:
        grp = pl.col("pos_grp").str.to_lowercase()
        df = df.filter(~grp.str.contains("base") & ~grp.str.contains("special"))
    if "pos_abb" not in df.columns or "pos_rank" not in df.columns:
        return df.head(0)

    pos = pl.col("pos_abb").cast(pl.Utf8)
    rank = pl.col("pos_rank").cast(pl.Float64)
    df = df.filter(
        ((pos.is_in(["QB", "RB", "TE", "HB"]) & (rank <= 2.0))
         | (pos.is_in(["WR", "LWR", "RWR", "SWR"]) & (rank <= 3.0)))
        & pl.col("gsis_id").is_not_null()
    )
    # Map depth abbrev → fantasy position
    df = df.with_columns(
        pl.when(pos.is_in(["HB", "FB"]))
        .then(pl.lit("RB"))
        .when(pos.is_in(["LWR", "RWR", "SWR", "SLWR", "SRWR"]))
        .then(pl.lit("WR"))
        .otherwise(pos)
        .alias("position")
    )
    return (
        df.select(
            [
                pl.col("gsis_id").cast(pl.Utf8),
                "team",
                "position",
                pl.col("pos_rank").cast(pl.Float64).alias("depth_rank"),
                pl.col("player_name") if "player_name" in df.columns else pl.lit(None).alias("player_name"),
            ]
        )
        .unique(subset=["gsis_id"], keep="first")
    )


def _position_feature_templates(snap: pl.DataFrame) -> pl.DataFrame:
    """Median non-usage feature row per position (efficiency / context only).

    Usage share / rolling volume columns are excluded so rookie stubs do not
    inherit WR1-level target/carry priors from the position median.
    """
    if snap.is_empty() or "position" not in snap.columns:
        return snap.head(0)
    skip = {
        "gsis_id",
        "player_id",
        "player_name",
        "display_name",
        "team",
        "season",
        "week",
        "is_rookie",
        "rookie_season",
        "depth_rank",
        "is_listed_starter",
        "same_pos_depth_count",
        "age",
        "draft_pick",
        "draft_round",
        "draft_value",
        "games_played_prior",
        "prior_season_games_played",
        "combine_forty",
        "combine_height",
        "combine_weight",
        "forty_missing",
        "college_ppa_all",
        "college_ppa_last_season",
        "college_rec_yards",
        "college_rush_yards",
        "college_recruiting_stars",
        "college_recruiting_rating",
        "vacated_target_share",
        "vacated_carry_share",
    }
    usage_substrings = (
        "share",
        "targets",
        "carries",
        "attempts",
        "receptions",
        "yards",
        "tds",
        "snap",
        "wopr",
        "racr",
        "air_yards",
        "fantasy_points",
        "xfp",
        "fp_minus",
    )
    num_cols = []
    for c, dtype in snap.schema.items():
        if c in skip or not dtype.is_numeric():
            continue
        cl = c.lower()
        if any(s in cl for s in usage_substrings):
            continue
        num_cols.append(c)
    if not num_cols:
        return snap.select(["position"]).unique()
    return snap.group_by("position").agg([pl.col(c).median().alias(c) for c in num_cols])


def _stub_share_priors(
    depth_rank: float | None,
    *,
    position: str | None = None,
) -> dict[str, float]:
    """Usage priors for roster stubs with no NFL sample.

    Listed starters get role-appropriate floors so depth-chart RB1/WR1 rookies
    are not anchored to near-zero "history" that blocks volume seeding and
    invents FP-only projections. Deeper stubs stay tiny.
    """
    w = depth_share_weight(depth_rank)
    rank = int(round(float(depth_rank))) if depth_rank is not None else 99
    pos = (position or "").upper()
    if rank == 1:
        carry = 0.42 if pos in ("RB", "HB", "FB") else (0.16 if pos == "QB" else 0.02 * w)
        if pos in ("WR", "LWR", "RWR", "SWR"):
            target = 0.18
        elif pos == "TE":
            target = 0.14
        elif pos in ("RB", "HB", "FB"):
            target = 0.06
        else:
            target = 0.08 * w
        dropback = 0.95 if pos == "QB" else 0.02 * w
        snap = 0.65 if pos in ("RB", "WR", "TE", "QB") else 0.15 * w
        return {
            "target_share": target,
            "carry_share": carry,
            "dropback_share": dropback,
            "snap_share": snap,
            "air_yards_share": 0.16 if pos.startswith("W") else 0.06 * w,
            "redzone_target_share": 0.10 if pos in ("WR", "TE", "RB") else 0.05 * w,
        }
    return {
        "target_share": 0.08 * w,
        "carry_share": 0.12 * w,
        "dropback_share": 0.05 * w,
        "snap_share": 0.15 * w,
        "air_yards_share": 0.06 * w,
        "redzone_target_share": 0.05 * w,
    }


def seed_missing_depth_players(
    snap: pl.DataFrame,
    *,
    panel: pl.DataFrame,
    depth: pl.DataFrame,
    roster: pl.DataFrame,
    target_season: int,
) -> pl.DataFrame:
    """Add rostered skill players missing from the feature snapshot.

    The depth chart supplies role rank when available, but the Week 1 roster is
    the coverage authority. Pulls a player's last NFL sample when available;
    otherwise builds a low-usage position-median stub.
    """
    if roster.is_empty() or "gsis_id" not in snap.columns:
        return snap

    roster_board = roster
    if "position" not in roster_board.columns:
        return snap
    roster_board = roster_board.with_columns(
        pl.when(pl.col("position").is_in(["HB", "FB"]))
        .then(pl.lit("RB"))
        .when(pl.col("position").is_in(["LWR", "RWR", "SWR", "SLWR", "SRWR"]))
        .then(pl.lit("WR"))
        .otherwise(pl.col("position"))
        .alias("position")
    ).filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    name_expr = (
        pl.col("player_name")
        if "player_name" in roster_board.columns
        else (
            pl.col("full_name").alias("player_name")
            if "full_name" in roster_board.columns
            else pl.lit(None).cast(pl.Utf8).alias("player_name")
        )
    )
    board = roster_board.select(
        [
            pl.col("gsis_id").cast(pl.Utf8),
            "team",
            "position",
            name_expr,
        ]
    ).unique(subset=["gsis_id"], keep="first")
    depth_board = latest_skill_depth_board(depth)
    if not depth_board.is_empty():
        depth_meta = depth_board.select(
            ["gsis_id", "depth_rank"]
        ).unique(subset=["gsis_id"], keep="first")
        board = board.join(depth_meta, on="gsis_id", how="left")

    roster_ids = set(roster["gsis_id"].cast(pl.Utf8).to_list()) if not roster.is_empty() else set()
    have = set(snap["gsis_id"].cast(pl.Utf8).to_list())
    need = board.filter(
        pl.col("gsis_id").is_in(list(roster_ids)) & ~pl.col("gsis_id").is_in(list(have))
    )
    roster_meta = roster.select(
        [c for c in roster.columns if c not in ("team", "position")]
    ).unique(subset=["gsis_id"], keep="first")
    if roster_meta.width > 1:
        need = need.join(roster_meta, on="gsis_id", how="left")
    if need.is_empty():
        return snap

    # Prefer historical panel rows (any season)
    hist = panel.filter(pl.col("gsis_id").cast(pl.Utf8).is_in(need["gsis_id"].to_list()))
    from_hist = (
        hist.sort(["gsis_id", "season", "week"])
        .unique(subset=["gsis_id"], keep="last")
        if not hist.is_empty()
        else hist
    )
    from_hist = _drop_existing(from_hist, ["season", "week"] + SCHEDULE_CTX_COLS + DEPTH_COLS)
    if not from_hist.is_empty():
        meta = need.select(["gsis_id", "team", "position"]).rename(
            {"team": "_seed_team", "position": "_seed_pos"}
        )
        from_hist = _drop_existing(from_hist, ["team", "position"]).join(meta, on="gsis_id", how="left")
        from_hist = from_hist.rename({"_seed_team": "team", "_seed_pos": "position"})

    seeded_ids = set(from_hist["gsis_id"].cast(pl.Utf8).to_list()) if not from_hist.is_empty() else set()
    still = need.filter(~pl.col("gsis_id").is_in(list(seeded_ids)))

    stubs = still.head(0)
    if not still.is_empty():
        templates = _position_feature_templates(snap)
        stubs = still.join(templates, on="position", how="left")
        if "player_name" not in stubs.columns and "player_name" in snap.columns:
            stubs = stubs.with_columns(pl.lit(None).cast(pl.Utf8).alias("player_name"))
        rookie_exprs = []
        if "rookie_year" in stubs.columns:
            rookie_exprs.append(pl.col("rookie_year").cast(pl.Int64, strict=False) == target_season)
        if "entry_year" in stubs.columns:
            rookie_exprs.append(pl.col("entry_year").cast(pl.Int64, strict=False) == target_season)
        is_rookie = rookie_exprs[0] if rookie_exprs else pl.lit(False)
        for expr in rookie_exprs[1:]:
            is_rookie = is_rookie | expr
        stub_exprs = [
            is_rookie.fill_null(False).cast(pl.Int8).alias("is_rookie"),
            pl.when(is_rookie.fill_null(False))
            .then(pl.lit(int(target_season)))
            .otherwise(pl.lit(None))
            .cast(pl.Int64)
            .alias("rookie_season"),
            pl.lit(0.0).cast(pl.Float64).alias("games_played_prior"),
            pl.lit(0.0).cast(pl.Float64).alias("prior_season_games_played"),
        ]
        if "birth_date" in stubs.columns:
            stub_exprs.append(
                (pl.lit(float(target_season)) - pl.col("birth_date").cast(pl.Utf8).str.slice(0, 4).cast(pl.Float64, strict=False))
                .alias("age")
            )
        if "draft_pick" not in stubs.columns and "draft_number" in stubs.columns:
            stub_exprs.append(pl.col("draft_number").cast(pl.Float64, strict=False).alias("draft_pick"))
        stubs = stubs.with_columns(stub_exprs)
        # Depth-scaled usage priors. Only seed ``*_prior_season`` anchors — not
        # l5 / shrunk / season_td — so we do not invent recent form for rookies.
        share_cols = ["gsis_id"]
        if "depth_rank" in stubs.columns:
            share_cols.append("depth_rank")
        if "position" in stubs.columns:
            share_cols.append("position")
        share_rows = [
            {
                "gsis_id": row["gsis_id"],
                **_stub_share_priors(row.get("depth_rank"), position=row.get("position")),
            }
            for row in stubs.select(share_cols).iter_rows(named=True)
        ]
        share_df = pl.DataFrame(share_rows)
        stubs = stubs.join(share_df, on="gsis_id", how="left")
        base_to_prior = {
            "target_share": "target_share",
            "carry_share": "carry_share",
            "dropback_share": "dropback_share",
            "snap_share": "snap_share",
            "air_yards_share": "air_yards_share",
            "redzone_target_share": "redzone_target_share",
        }
        for c in snap.columns:
            cl = c.lower()
            if not cl.endswith("_prior_season"):
                continue
            for base, prior_col in base_to_prior.items():
                if cl == f"{base}_prior_season" and prior_col in stubs.columns:
                    stubs = stubs.with_columns(pl.col(prior_col).cast(pl.Float64).alias(c))
                    break
        # Drop helper prior cols that aren't on the panel schema
        drop_helpers = [
            c
            for c in base_to_prior
            if c in stubs.columns and c not in snap.columns
        ]
        if drop_helpers:
            stubs = stubs.drop(drop_helpers)

    pieces = [snap]
    if not from_hist.is_empty():
        pieces.append(_drop_existing(from_hist, DEPTH_COLS))
    if not stubs.is_empty():
        pieces.append(_drop_existing(stubs, DEPTH_COLS))
    out = pl.concat(pieces, how="diagonal_relaxed")
    if "position" in out.columns:
        out = out.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    logger.info(
        "Seeded %d roster players into outlook (%d from history, %d stubs)",
        need.height,
        0 if from_hist.is_empty() else from_hist.height,
        0 if stubs.is_empty() else stubs.height,
    )
    return out


def apply_current_roster_teams(
    snap: pl.DataFrame,
    target_season: int,
    *,
    force_rosters: bool = False,
    drop_missing: bool = False,
    roster_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Overwrite snapshot ``team`` from target-season rosters (free-agency aware)."""
    if "gsis_id" not in snap.columns:
        logger.warning("Outlook snapshot missing gsis_id; cannot remap teams")
        return snap

    roster = current_roster_teams(
        target_season, force=force_rosters, roster_data=roster_data
    )
    if roster.is_empty():
        logger.warning("No %s rosters found; leaving prior-season teams", target_season)
        return snap

    before = snap.select(["gsis_id", "team"]).unique() if "team" in snap.columns else None
    out = _drop_existing(snap, ["team"])
    out = out.join(roster.select(["gsis_id", "team"]), on="gsis_id", how="left")

    remapped = 0
    if before is not None:
        cmp = before.join(
            out.select(["gsis_id", "team"]).unique().rename({"team": "team_new"}),
            on="gsis_id",
            how="left",
        )
        remapped = cmp.filter(
            pl.col("team_new").is_not_null() & (pl.col("team") != pl.col("team_new"))
        ).height

    missing = out.filter(pl.col("team").is_null()).height
    if drop_missing:
        out = out.filter(pl.col("team").is_not_null())
    elif missing and before is not None:
        # Fall back to prior-season team when player not on target roster
        prior_team = before.rename({"team": "_prior_team"})
        out = out.join(prior_team, on="gsis_id", how="left").with_columns(
            pl.coalesce([pl.col("team"), pl.col("_prior_team")]).alias("team")
        ).drop("_prior_team")

    out = normalize_team_column(out, "team")

    logger.info(
        "Roster remap for %s: %d team changes, %d missing on roster (kept %d)",
        target_season,
        remapped,
        missing,
        out.height,
    )
    return out


def build_outlook_panel(
    panel: pl.DataFrame,
    *,
    target_season: int,
    feature_season: int | None = None,
    force_rosters: bool = False,
    roster_data: pl.DataFrame | None = None,
    depth_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Clone prior-season feature rows onto target-season schedule weeks.

    Used when the panel has no rows for ``target_season`` yet (preseason draft).
    Uses each player's last available feature week (not only week 18), remaps
    teams from current rosters (dropping free agents), and normalizes aliases
    like AZ→ARI so schedule/depth joins hit the right club.
    """
    feature_season = feature_season or (target_season - 1)
    prior = panel.filter(pl.col("season") == feature_season)
    if prior.is_empty():
        raise ValueError(f"No panel rows for feature season {feature_season}")

    snapshot = end_of_season_player_snapshot(prior)
    if snapshot.is_empty():
        raise ValueError(f"Empty last-available snapshot for {feature_season}")
    source_weeks = sorted(int(w) for w in snapshot["week"].unique().to_list() if w is not None)

    schedules = load_schedules([target_season, feature_season])
    team_weeks = explode_schedules_to_team_weeks(schedules)
    # Pass-rate priors from historical player stats still on panel
    team_weeks = add_team_pass_rate(team_weeks, panel)
    team_weeks = add_prior_season_team_pass_rate(team_weeks, panel)
    try:
        from src.projection.weekly.data.nflverse_loader import load_team_stats

        # Target-season team stats do not exist during the preseason.  The
        # feature helper carries the completed season's final defensive form
        # into week 1 rather than failing the entire context join on a 404.
        team_stats = load_team_stats([feature_season])
        team_weeks = add_opponent_defense_features(team_weeks, team_stats)
    except Exception as exc:
        logger.warning("Outlook opponent defense skipped: %s", exc)

    target_ctx = team_weeks.filter(pl.col("season") == target_season)
    if target_ctx.is_empty():
        raise ValueError(f"No schedule team-weeks for season {target_season}")
    target_ctx = normalize_team_column(target_ctx, "team")
    if "opponent" in target_ctx.columns:
        target_ctx = normalize_team_column(target_ctx, "opponent")

    # Use the same synthetic preseason Vegas fallback for team totals, volume,
    # and efficiency.  Previously only the team model saw the filled values,
    # while player models silently median-imputed most future games.
    try:
        from src.projection.weekly.models.team_totals import (
            build_team_week_labels,
            fill_missing_vegas_from_team_strength,
        )

        history = build_team_week_labels(panel.filter(pl.col("season") <= feature_season))
        target_ctx = fill_missing_vegas_from_team_strength(target_ctx, history=history)
    except Exception as exc:
        logger.warning("Outlook Vegas fallback skipped: %s", exc)

    if {"team_pass_rate_l5", "team_pass_rate_prior_season"}.issubset(target_ctx.columns):
        target_ctx = target_ctx.with_columns(
            pl.coalesce(
                [pl.col("team_pass_rate_l5"), pl.col("team_pass_rate_prior_season")]
            ).alias("team_pass_rate_l5")
        )

    weeks = sorted(int(w) for w in target_ctx["week"].unique().to_list() if w is not None)
    # Keep the complete regular season.  Joining scheduled team-weeks below
    # naturally removes each club's bye while retaining its week-18 game.
    weeks = [w for w in weeks if 1 <= w <= 18] or weeks

    snap = _drop_existing(snapshot, ["season", "week"] + SCHEDULE_CTX_COLS + DEPTH_COLS)
    # Only players on the target-season roster (drops cut / unsigned junk)
    snap = apply_current_roster_teams(
        snap,
        target_season,
        force_rosters=force_rosters,
        drop_missing=True,
        roster_data=roster_data,
    )
    if "position" in snap.columns:
        snap = snap.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))

    # Backfill rostered depth-chart skill players missing from feature season
    try:
        depth_seed = (
            depth_data
            if depth_data is not None
            else load_depth_charts([target_season], force=force_rosters)
        )
        roster_seed = current_roster_teams(
            target_season, force=force_rosters, roster_data=roster_data
        )
        draft_seed = pl.DataFrame()
        combine_seed = pl.DataFrame()
        college_seed = pl.DataFrame()
        try:
            draft_seed = load_draft_picks([target_season], force=force_rosters)
            combine_seed = load_combine([target_season], force=force_rosters)
            if not draft_seed.is_empty():
                from src.projection.weekly.data.cfbd_loader import load_college_features_for_drafted

                college_seed = load_college_features_for_drafted(
                    draft_seed, seasons=[target_season], force=force_rosters
                )
        except Exception as exc:
            # Roster rookie_year/entry_year remains a valid identity fallback.
            logger.warning("Target rookie prospect data unavailable: %s", exc)
        roster_seed = attach_target_rookie_features(
            roster_seed,
            target_season=target_season,
            draft=draft_seed,
            combine=combine_seed,
            college=college_seed,
        )
        vacated = build_target_vacated_opportunity(
            panel,
            roster_seed,
            target_season=target_season,
            feature_season=feature_season,
        )
        if not vacated.is_empty():
            roster_seed = roster_seed.join(vacated, on="team", how="left")
        snap = seed_missing_depth_players(
            snap,
            panel=panel,
            depth=depth_seed,
            roster=roster_seed,
            target_season=target_season,
        )
        snap = normalize_team_column(snap, "team")
    except Exception as exc:
        logger.warning("Outlook depth-player seed skipped: %s", exc)

    try:
        availability_roster = (
            roster_data
            if roster_data is not None
            else load_rosters([target_season], force=force_rosters)
        )
        availability = estimate_projected_games(
            panel,
            snap,
            target_season=target_season,
            roster=availability_roster,
        )
        snap = snap.join(availability, on="gsis_id", how="left")
    except Exception as exc:
        logger.warning("Availability estimate skipped: %s", exc)

    # Clear current-week outcomes so models rely on lagged features only
    outcome_like = [
        c
        for c in STAT_MEAN_COLS
        + ["fantasy_points", "ypa", "ypc", "ypr", "catch_rate", "xfp", "fp_minus_xfp"]
        if c in snap.columns
    ]
    if outcome_like:
        snap = snap.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in outcome_like])

    # Every outlook week is forecast from the same preseason information state,
    # so zero target-season games have been observed for anyone.  Carrying the
    # stale in-season index off the snapshot row would rank players by how late
    # in the prior year they last appeared.
    snap = snap.with_columns(pl.lit(0.0).cast(pl.Float64).alias("games_played_prior"))
    if "prior_season_games_played" not in snap.columns:
        snap = snap.with_columns(
            pl.lit(None).cast(pl.Float64).alias("prior_season_games_played")
        )

    frames: list[pl.DataFrame] = []
    for week in weeks:
        ctx = target_ctx.filter(pl.col("week") == week).select(
            ["team"] + [c for c in SCHEDULE_CTX_COLS if c in target_ctx.columns]
        )
        week_df = snap.with_columns(
            [
                pl.lit(target_season).cast(pl.Int64).alias("season"),
                pl.lit(week).cast(pl.Int64).alias("week"),
            ]
        )
        # is_rookie for target season: drafted in target_season
        if "rookie_season" in week_df.columns:
            week_df = week_df.with_columns(
                (pl.col("rookie_season") == target_season)
                .fill_null(False)
                .cast(pl.Int8)
                .alias("is_rookie")
            )
        week_df = week_df.join(ctx, on="team", how="inner")
        frames.append(week_df)

    out = pl.concat(frames, how="vertical_relaxed")

    # Raw depth as-of kickoff, then effective depth (roles + availability)
    try:
        depth = (
            depth_data
            if depth_data is not None
            else load_depth_charts([target_season], force=force_rosters)
        )
        depth = normalize_team_column(depth, "team") if "team" in depth.columns else depth
        out = attach_depth_features(
            out, depth, schedules=schedules.filter(pl.col("season") == target_season)
        )
        rost_full = (
            roster_data
            if roster_data is not None
            else load_rosters([target_season], force=force_rosters)
        )
        rost_full = normalize_team_column(rost_full, "team") if "team" in rost_full.columns else rost_full
        eff = build_effective_depth(
            depth,
            rosters=rost_full,
            horizon="season",
            force_injuries=force_rosters,
        )
        out = attach_effective_depth(out, eff, overwrite_depth_rank=True)
    except Exception as exc:
        logger.warning("Outlook depth / effective depth skipped: %s", exc)

    # Current-season freshness overlay.  It is never attached to historical
    # backtests, and short-term Out/Q designations are still cleared below so a
    # point-in-time status is not frozen across the full season.  IR/PUP status
    # remains available to the season-long availability path.
    if target_season == date.today().year:
        try:
            from src.projection.weekly.data.nflverse_loader import load_ff_playerids
            from src.projection.weekly.data.sleeper import fetch_sleeper_players
            from src.projection.weekly.features.sleeper import attach_current_sleeper_overlay

            out = attach_current_sleeper_overlay(
                out,
                fetch_sleeper_players(force=False),
                load_ff_playerids(force=False),
                live_season=target_season,
            )
            sleeper_out = pl.col("sleeper_is_out").fill_null(False)
            sleeper_ir = pl.col("sleeper_is_ir").fill_null(False)
            out = out.with_columns(
                [
                    (pl.col("is_out").fill_null(False) | sleeper_out | sleeper_ir).alias(
                        "is_out"
                    ),
                    pl.when(sleeper_out | sleeper_ir)
                    .then(pl.lit(0.0))
                    .otherwise(pl.col("play_prob").fill_null(1.0))
                    .alias("play_prob"),
                    pl.when(sleeper_ir)
                    .then(pl.lit("Injured Reserve"))
                    .when(sleeper_out)
                    .then(pl.lit("Out"))
                    .otherwise(pl.col("injury_status"))
                    .alias("injury_status"),
                ]
            )
        except Exception as exc:
            logger.warning("Sleeper current-status overlay skipped: %s", exc)

    # Season outlook: do not freeze short-term ESPN Out/Q across all 17 weeks
    out = clear_short_term_injuries(out)
    out = condition_season_outlook_on_playing(out)

    logger.info(
        "Built outlook panel for %s from %s last-available weeks %s: %d players x %d weeks",
        target_season,
        feature_season,
        source_weeks[-5:],
        snap.height,
        len(weeks),
    )
    return out

def project_all_weeks(
    panel: pl.DataFrame,
    *,
    season: int,
    scoring: ScoringConfig | None = None,
    train_seasons: list[int] | None = None,
    weeks: list[int] | None = None,
) -> pl.DataFrame:
    """Project each week and stack results."""
    scoring = scoring or ScoringConfig.from_name("half_ppr")
    train_seasons = train_seasons or list(
        range(TRAIN_START_SEASON, min(TRAIN_END_SEASON, season - 1) + 1)
    )
    weeks = weeks or available_weeks(panel, season)
    if not weeks:
        raise ValueError(f"No weeks to project for season {season}")

    frames: list[pl.DataFrame] = []
    for week in weeks:
        try:
            proj = project_week_with_rookies(
                panel,
                season=season,
                week=week,
                scoring=scoring,
                train_seasons=train_seasons,
            )
        except Exception as exc:
            logger.warning("Skip %s week %s: %s", season, week, exc)
            continue
        # Carry depth chart features from the feature panel (not returned by models)
        depth_present = [c for c in DEPTH_COLS if c in panel.columns]
        if depth_present and "gsis_id" in proj.columns:
            depth_rows = (
                panel.filter((pl.col("season") == season) & (pl.col("week") == week))
                .select(["gsis_id"] + depth_present)
                .unique(subset=["gsis_id"], keep="first")
            )
            proj = _drop_existing(proj, depth_present).join(depth_rows, on="gsis_id", how="left")
        contract_present = [c for c in CONTRACT_COLS if c in panel.columns]
        if contract_present and "gsis_id" in proj.columns:
            contract_rows = (
                panel.filter((pl.col("season") == season) & (pl.col("week") == week))
                .select(["gsis_id"] + contract_present)
                .unique(subset=["gsis_id"], keep="first")
            )
            proj = _drop_existing(proj, contract_present).join(
                contract_rows, on="gsis_id", how="left"
            )
        frames.append(proj)
        logger.info("Projected %s week %s (%d players)", season, week, proj.height)

    if not frames:
        raise RuntimeError(f"No successful weekly projections for season {season}")
    return pl.concat(frames, how="vertical_relaxed")


def condition_season_receiving_on_availability(
    season_df: pl.DataFrame,
    *,
    scoring: ScoringConfig | None = None,
    max_scale: float = 1.35,
) -> pl.DataFrame:
    """Reallocate missed-game receiving volume across the named depth chart.

    Weekly accounting produces coherent team totals, but independently estimated
    player availability makes ``per_game * projected_games`` fall short at the
    season level.  Scale each team's named pass catchers pro rata so replacements
    absorb that opportunity while preserving relative player shares and
    efficiencies.  The cap prevents thin or incomplete rosters from creating an
    implausible multiplier; any remainder stays available to the explicit export
    reserve.
    """
    receiver_stats = ("targets", "receptions", "receiving_yards", "receiving_tds")
    required = {
        "team",
        "position",
        "weeks_projected",
        "projected_games",
        "fantasy_points",
        "fantasy_pts",
        "completions",
        "passing_yards",
        "passing_tds",
        *receiver_stats,
    }
    if season_df.is_empty() or not required.issubset(season_df.columns):
        return season_df

    scoring = scoring or ScoringConfig.from_name("half_ppr")
    receiver = pl.col("position").is_in(["RB", "WR", "TE"])
    totals = season_df.group_by("team").agg(
        [
            pl.col("weeks_projected").max().cast(pl.Float64).alias("_season_horizon"),
            pl.col("targets").filter(receiver).sum().alias("_targets_pg"),
            pl.col("completions").sum().alias("_receptions_budget_pg"),
            pl.col("passing_yards").sum().alias("_receiving_yards_budget_pg"),
            pl.col("passing_tds").sum().alias("_receiving_tds_budget_pg"),
            *[
                (pl.col(stat) * pl.col("projected_games"))
                .filter(receiver)
                .sum()
                .alias(f"_{stat}_named")
                for stat in receiver_stats
            ],
        ]
    )
    totals = totals.with_columns(
        [
            (pl.col("_targets_pg") * pl.col("_season_horizon")).alias(
                "_targets_budget"
            ),
            *[
                (pl.col(f"_{stat}_budget_pg") * pl.col("_season_horizon")).alias(
                    f"_{stat}_budget"
                )
                for stat in ("receptions", "receiving_yards", "receiving_tds")
            ],
        ]
    )
    totals = totals.with_columns(
        [
            pl.when(pl.col(f"_{stat}_named") > 1e-9)
            .then(
                (pl.col(f"_{stat}_budget") / pl.col(f"_{stat}_named")).clip(
                    1.0, max_scale
                )
            )
            .otherwise(1.0)
            .alias(f"_availability_scale_{stat}")
            for stat in receiver_stats
        ]
    )

    out = season_df.join(
        totals.select(
            ["team"] + [f"_availability_scale_{stat}" for stat in receiver_stats]
        ),
        on="team",
        how="left",
    )
    fp_delta = pl.when(receiver).then(
        pl.col("receptions")
        * (pl.col("_availability_scale_receptions") - 1.0)
        * scoring.reception_points
        + pl.col("receiving_yards")
        * (pl.col("_availability_scale_receiving_yards") - 1.0)
        * scoring.rush_rec_yard_points
        + pl.col("receiving_tds")
        * (pl.col("_availability_scale_receiving_tds") - 1.0)
        * scoring.rush_rec_td_points
    ).otherwise(0.0)
    out = out.with_columns(fp_delta.alias("_season_receiving_fp_delta"))
    out = out.with_columns(
        [
            pl.when(receiver)
            .then(pl.col(stat) * pl.col(f"_availability_scale_{stat}"))
            .otherwise(pl.col(stat))
            .alias(stat)
            for stat in receiver_stats
        ]
        + [
            (pl.col("fantasy_points") + pl.col("_season_receiving_fp_delta")).alias(
                "fantasy_points"
            ),
            (pl.col("fantasy_pts") + pl.col("_season_receiving_fp_delta")).alias(
                "fantasy_pts"
            ),
        ]
    )
    interval_cols = [c for c in ("floor", "ceiling", "fantasy_pts_low", "fantasy_pts_high") if c in out.columns]
    if interval_cols:
        out = out.with_columns(
            [
                (pl.col(col) + pl.col("_season_receiving_fp_delta")).alias(col)
                for col in interval_cols
            ]
        )
    return out.rename(
        {
            f"_availability_scale_{stat}": f"season_availability_scale_{stat}"
            for stat in receiver_stats
        }
    ).drop("_season_receiving_fp_delta")


def aggregate_season_projections(
    weekly: pl.DataFrame,
    *,
    projected_games: int | None = None,
    scoring: ScoringConfig | None = None,
) -> pl.DataFrame:
    """Collapse week-level projections to per-player season PPG / totals."""
    if weekly.is_empty():
        return weekly

    id_key = "gsis_id" if "gsis_id" in weekly.columns else "player_name"
    mean_cols = [
        c
        for c in (
            ["fantasy_points", "floor", "ceiling"]
            + STAT_MEAN_COLS
            + [
                "depth_rank",
                "is_listed_starter",
                "same_pos_depth_count",
                "play_prob",
                "is_out",
                "projected_games_estimate",
            ]
            + CONTRACT_COLS
        )
        if c in weekly.columns
    ]

    aggs: list[pl.Expr] = [pl.len().alias("weeks_projected")]
    for c in mean_cols:
        aggs.append(pl.col(c).mean().alias(c))
    for c in ("player_name", "position", "team"):
        if c in weekly.columns and c != id_key:
            aggs.append(pl.col(c).drop_nulls().first().alias(c))
    for c in ("is_rookie", "rookie_season"):
        if c in weekly.columns:
            aggs.append(pl.col(c).drop_nulls().max().alias(c))

    season_df = weekly.group_by(id_key).agg(aggs)

    games = projected_games
    if games is not None:
        projected_games_expr = pl.lit(float(games))
    elif "projected_games_estimate" in season_df.columns:
        projected_games_expr = pl.min_horizontal(
            pl.col("projected_games_estimate").fill_null(
                pl.col("weeks_projected").cast(pl.Float64)
            ),
            pl.col("weeks_projected").cast(pl.Float64),
        )
    else:
        projected_games_expr = pl.col("weeks_projected").cast(pl.Float64)
    season_df = season_df.with_columns(
        [
            pl.col("fantasy_points").alias("fantasy_pts"),
            pl.col("floor").alias("fantasy_pts_low")
            if "floor" in season_df.columns
            else pl.lit(None).cast(pl.Float64).alias("fantasy_pts_low"),
            pl.col("ceiling").alias("fantasy_pts_high")
            if "ceiling" in season_df.columns
            else pl.lit(None).cast(pl.Float64).alias("fantasy_pts_high"),
            projected_games_expr.alias("projected_games"),
        ]
    )
    season_df = condition_season_receiving_on_availability(
        season_df,
        scoring=scoring,
    )
    season_df = season_df.with_columns(
        (pl.col("fantasy_pts") * pl.col("projected_games")).alias("fantasy_pts_season")
    )

    # Depth / role helpers for draft CSV
    if "depth_rank" in season_df.columns:
        season_df = season_df.with_columns(
            [
                pl.when(pl.col("depth_rank") <= 1.0)
                .then(pl.lit("starter"))
                .when(pl.col("depth_rank") <= 2.0)
                .then(pl.lit("backup"))
                .when(pl.col("depth_rank").is_not_null())
                .then(pl.lit("depth"))
                .otherwise(pl.lit(None))
                .alias("role"),
                pl.when(pl.col("depth_rank") <= 1.0)
                .then(pl.lit("starter"))
                .when(pl.col("depth_rank").is_not_null())
                .then(pl.lit("listed"))
                .otherwise(pl.lit(None))
                .alias("depth_chart_status"),
            ]
        )
    else:
        season_df = season_df.with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("role"),
                pl.lit(None).cast(pl.Utf8).alias("depth_chart_status"),
            ]
        )

    rookie_expr = (
        pl.col("is_rookie").fill_null(0).cast(pl.Int8) == 1
        if "is_rookie" in season_df.columns
        else pl.lit(False)
    )
    availability_low = (
        pl.col("projected_games").fill_null(17.0) < 14.0
        if "projected_games" in season_df.columns
        else pl.lit(False)
    )
    out_expr = (
        pl.col("is_out").fill_null(0) >= 0.5
        if "is_out" in season_df.columns
        else pl.lit(False)
    )
    # Keep the reasons separable.  ``low_confidence`` is a display-level "this
    # interval is wide" flag; consumers that need *why* must not re-derive it,
    # because a thin projection and a fragile one warrant different treatment.
    season_df = season_df.with_columns(
        [
            rookie_expr.alias("is_rookie_projection"),
            (out_expr | availability_low).alias("low_availability"),
            (out_expr | rookie_expr | availability_low).alias("low_confidence"),
        ]
    )

    season_df = season_df.with_columns(
        pl.when(rookie_expr)
        .then(pl.lit("rookie_rule"))
        .otherwise(pl.lit("v2_team_first"))
        .alias("source")
    )
    return season_df.sort("fantasy_pts", descending=True)


def project_season(
    panel: pl.DataFrame,
    *,
    season: int,
    scoring: ScoringConfig | None = None,
    train_seasons: list[int] | None = None,
    projected_games: int | None = None,
    force_rosters: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (weekly_stack, season_aggregate) for a draft season.

    If ``season`` is absent from the panel, builds a preseason outlook from the
    prior season's final-week features overlaid on the target schedule, with
    teams remapped from current rosters.
    """
    scoring = scoring or ScoringConfig.from_name("half_ppr")
    weeks = available_weeks(panel, season)
    work_panel = panel

    if not weeks:
        logger.info(
            "Season %s not in panel; building preseason outlook from prior season",
            season,
        )
        outlook = build_outlook_panel(
            panel, target_season=season, force_rosters=force_rosters
        )
        # Merge outlook rows with historical panel so residual bands / models still see history
        work_panel = pl.concat([panel, outlook], how="diagonal_relaxed")
        weeks = available_weeks(work_panel, season)

    # Refresh as-of contracts for all seasons in work_panel (fixes outlook year remap)
    try:
        from src.projection.weekly.data.nflverse_loader import load_contracts
        from src.projection.weekly.features.contracts import CONTRACT_FEATURE_COLS, attach_contract_features

        drop = [c for c in CONTRACT_FEATURE_COLS if c in work_panel.columns]
        if drop:
            work_panel = work_panel.drop(drop)
        work_panel = attach_contract_features(work_panel, load_contracts())
    except Exception as exc:
        logger.warning("Contract refresh on season panel skipped: %s", exc)

    weekly = project_all_weeks(
        work_panel,
        season=season,
        scoring=scoring,
        train_seasons=train_seasons,
        weeks=weeks,
    )
    season_df = aggregate_season_projections(
        weekly,
        projected_games=projected_games,
        scoring=scoring,
    )
    season_df = season_df.with_columns(pl.lit(season).cast(pl.Int64).alias("season"))
    return weekly, season_df


def write_season_outputs(
    season_df: pl.DataFrame,
    weekly: pl.DataFrame | None = None,
    *,
    season: int,
    outputs_dir: Path | None = None,
) -> dict[str, Path]:
    """Write season CSV/parquet under outputs/ for draft adapter consumption."""
    ensure_dirs()
    out_dir = outputs_dir or OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    season_csv = out_dir / f"season_projections_{season}.csv"
    season_pq = out_dir / f"season_projections_{season}.parquet"
    season_df.write_csv(season_csv)
    season_df.write_parquet(season_pq)
    paths["season_csv"] = season_csv
    paths["season_parquet"] = season_pq

    if weekly is not None and not weekly.is_empty():
        weekly_path = out_dir / f"season_weekly_{season}.parquet"
        weekly.write_parquet(weekly_path)
        paths["weekly_parquet"] = weekly_path

    logger.info("Wrote season outputs for %s -> %s", season, season_csv)
    return paths
