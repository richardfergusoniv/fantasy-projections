"""Leakage-safe 2024 -> 2025 fantasy-season evaluation.

The forecast stage accepts only source-season-or-earlier NFL features plus a
frozen Week-1 roster snapshot.  Target-season outcomes are attached in a
separate function after forecasts have been finalized.  This is intentionally
different from a rate-only transition backtest: the evaluation population
contains every contracted preseason player, including zero-game outcomes and
players for whom the model has no prior-season feature row.

Scoring is half-PPR with four-point passing touchdowns.  Fumbles and two-point
conversions are omitted because they are not modeled upstream.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.composition import compose_board, leakage_safe_context
from src.projection.data_prep import get_conn
from src.projection.depth_history import (
    attach_availability_depth_rank,
    attach_depth_rank,
)
from src.projection.depth_rates import depth_rate_factor
from src.projection.fantasy_points import SCORING
from src.projection.features import TARGET_STATS, build_player_season_features
from src.projection.team_reconcile import (
    TEAM_ANCHOR_SPECS,
    _compose_reframed_receiving_predictions,
)
from src.projection.rookies import (
    TEAM_ABBR_FIX,
    _round_bucket,
    fit_rookie_baselines,
    load_combine_athletic_tier,
    predict_rookies,
    team_vacated_opportunity,
)
from src.projection.train import fit_availability, fit_one, fit_team_total
from src.projection.transitions import (
    ALL_FEATURES,
    AVAILABILITY_FEATURES,
    REFRAMED_SHARE_STATS,
    SEASON_GAMES,
    age_shrunk_predict,
    TEAM_FEATURES,
    TEAM_MODEL_FEATURES,
)

# Interval residuals ship in models/interval_residuals.csv, fit by backtest.py
# on every season including this fold's target. Reusing them here would leak, so
# the shared receiving composition is handed an EMPTY residual table: composed
# point predictions are identical, and the prediction interval is simply absent
# on this path rather than borrowed from a leaky artifact. Nothing scored by
# this harness reads pred_pg_low/high.
_NO_LEAKAGE_SAFE_RESIDUALS = pd.DataFrame(
    columns=["position", "stat", "resid_low", "resid_high", "low_n_flag"]
)


POSITIONS = tuple(TARGET_STATS)
CONTRACTED_STATUSES = frozenset({"ACT", "DEV", "RES", "INA", "EXE"})
DEFAULT_TIER_RANKS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
DEFAULT_REPLACEMENT_RANKS = {"QB": 13, "RB": 25, "WR": 37, "TE": 13}
OUTCOME_RATE_COLUMNS = sorted(
    {f"{stat}_pg" for stats in TARGET_STATS.values() for stat in stats}
)
OUTCOME_TOTAL_COLUMNS = sorted(
    {stat for stats in TARGET_STATS.values() for stat in stats}
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# `output/`, not `model/` or `models/`. These are shipped deliverables and
# belong beside the projection/fantasy/Sleeper CSVs: `models/` is gitignored
# (so a freeze manifest could hash artifacts that were never committed), and
# a singular `model/` beside the real `models/` is a directory pair nobody
# can keep straight.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def safe_training_pairs(source_season: int) -> list[tuple[int, int]]:
    """Production-era transitions whose label season is no later than source."""
    return [(year, year + 1) for year in range(2021, source_season) if year + 1 <= source_season]


def load_frozen_week1_roster(conn, target_season: int) -> pd.DataFrame:
    """Load the earliest regular-season roster and freeze its position/team.

    CUT and RET rows are excluded.  This is a contracted Week-1 universe, not
    an August camp universe; players cut before this snapshot cannot be scored.
    """
    week = pd.read_sql(
        "SELECT MIN(week) AS week FROM weekly_rosters "
        "WHERE season = ? AND game_type = 'REG'",
        conn,
        params=(int(target_season),),
    ).at[0, "week"]
    if pd.isna(week):
        raise ValueError(f"No regular-season weekly_rosters snapshot for {target_season}")
    roster = pd.read_sql(
        "SELECT player_id, player_name, team, position, status, years_exp, "
        "draft_number, pfr_id FROM weekly_rosters "
        "WHERE season = ? AND week = ? AND game_type = 'REG' "
        "AND position IN ('QB','RB','WR','TE') AND player_id IS NOT NULL",
        conn,
        params=(int(target_season), int(week)),
    )
    roster = roster[roster["status"].isin(CONTRACTED_STATUSES)].copy()
    roster["team"] = roster["team"].replace(TEAM_ABBR_FIX)
    roster = (
        roster.sort_values(["player_id", "position", "team"])
        .drop_duplicates("player_id", keep="first")
        .reset_index(drop=True)
    )
    roster["season"] = int(target_season)
    roster["roster_week"] = int(week)
    return roster


def build_preseason_rookie_cohort(
    conn, feature_table: pd.DataFrame, seasons: list[int] | range
) -> pd.DataFrame:
    """Build same-definition historical rookie cohorts from Week-1 rosters.

    Draft capital is joined to the contracted Week-1 roster; undrafted
    rookies are Week-1 players with years_exp=0 and no draft number.  No
    target-season participation or full-season roster table defines inclusion.
    Actual rates are attached only so <=source cohorts can fit bucket means.
    """
    pieces: list[pd.DataFrame] = []
    for season in seasons:
        roster = load_frozen_week1_roster(conn, int(season))
        draft = pd.read_sql(
            "SELECT gsis_id AS draft_player_id, pfr_player_id AS draft_pfr_id, "
            "round, pick, team AS draft_team, position AS draft_position "
            "FROM draft_picks WHERE season = ? "
            "AND position IN ('QB','RB','WR','TE')",
            conn,
            params=(int(season),),
        )
        by_id = draft.dropna(subset=["draft_player_id"]).drop_duplicates(
            "draft_player_id"
        ).set_index("draft_player_id")
        by_pfr = draft.dropna(subset=["draft_pfr_id"]).drop_duplicates(
            "draft_pfr_id"
        ).set_index("draft_pfr_id")

        rows = roster.copy()
        rows["round"] = rows["player_id"].map(by_id.get("round", pd.Series(dtype=float)))
        rows["pick"] = rows["player_id"].map(by_id.get("pick", pd.Series(dtype=float)))
        missing_round = rows["round"].isna() & rows["pfr_id"].notna()
        if missing_round.any() and not by_pfr.empty:
            rows.loc[missing_round, "round"] = rows.loc[missing_round, "pfr_id"].map(by_pfr["round"])
            rows.loc[missing_round, "pick"] = rows.loc[missing_round, "pfr_id"].map(by_pfr["pick"])
        drafted = rows["round"].notna()
        udfa = rows["years_exp"].fillna(-1).eq(0) & rows["draft_number"].isna()
        rows = rows[drafted | udfa].copy()
        rows["rookie_tier"] = np.where(rows["round"].notna(), "drafted", "udfa")
        rows["round_bucket"] = rows["round"].apply(_round_bucket)
        rows["name"] = rows["player_name"]
        pieces.append(rows)

    rookies = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if rookies.empty:
        return rookies

    actual_cols = [
        "player_id", "season", "games_played", "opportunity_games",
        *[c for c in OUTCOME_RATE_COLUMNS if c in feature_table.columns],
    ]
    actual = (
        feature_table[actual_cols]
        .sort_values(["player_id", "season", "games_played"], ascending=[True, True, False])
        .drop_duplicates(["player_id", "season"])
    )
    rookies = rookies.merge(actual, on=["player_id", "season"], how="left")
    rookies[["games_played", "opportunity_games"]] = rookies[
        ["games_played", "opportunity_games"]
    ].fillna(0.0)
    rookies = rookies.merge(
        team_vacated_opportunity(conn, list(seasons)),
        on=["season", "team"],
        how="left",
    )
    athletic = load_combine_athletic_tier(conn)
    rookies = rookies.merge(athletic, on="player_id", how="left")
    rookies["athletic_tier"] = rookies["athletic_tier"].fillna("no_data")
    ranked = []
    for season, group in rookies.groupby("season", sort=False):
        group = attach_availability_depth_rank(group, int(season), conn=conn)
        ranked.append(attach_depth_rank(group, int(season), conn=conn))
    return pd.concat(ranked, ignore_index=True, sort=False)


def sanitize_target_rookie_outcomes(
    rookie_df: pd.DataFrame, target_season: int
) -> pd.DataFrame:
    """Erase held-out outcomes while retaining preseason rookie inputs."""
    out = rookie_df.copy()
    mask = out["season"].eq(int(target_season))
    cols = [
        c for c in ["games_played", "opportunity_games", *OUTCOME_RATE_COLUMNS]
        if c in out.columns
    ]
    out.loc[mask, cols] = np.nan
    return out


def build_preseason_population(
    conn, target_season: int, rookie_cohort: pd.DataFrame
) -> pd.DataFrame:
    """One row per contracted Week-1 player, never outcome-conditioned."""
    roster = load_frozen_week1_roster(conn, target_season)
    rookie_ids = set(
        rookie_cohort.loc[rookie_cohort["season"].eq(target_season), "player_id"]
    )
    out = roster.rename(
        columns={
            "player_name": "display_name",
            "team": "preseason_team",
            "position": "preseason_position",
            "status": "preseason_status",
        }
    )
    out["is_rookie"] = out["player_id"].isin(rookie_ids)
    out["population_source"] = np.where(
        out["is_rookie"], "week1_roster_rookie", "week1_roster_veteran"
    )
    return out[
        [
            "player_id", "display_name", "preseason_team",
            "preseason_position", "preseason_status", "season", "roster_week",
            "is_rookie", "population_source",
        ]
    ]


def _score_totals(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for stat, weight in SCORING.items():
        if stat in frame.columns:
            score = score + pd.to_numeric(frame[stat], errors="coerce").fillna(0.0) * weight
    return score


def _actual_player_totals(feature_table: pd.DataFrame, season: int) -> pd.DataFrame:
    cols = [c for c in OUTCOME_TOTAL_COLUMNS if c in feature_table.columns]
    base_cols = ["player_id", *cols]
    if "games_played" in feature_table.columns:
        base_cols.append("games_played")
    rows = feature_table[feature_table["season"].eq(season)][base_cols].copy()
    if rows.empty:
        return pd.DataFrame(columns=[
            "player_id", *cols, "actual_games_played", "actual_row_present",
            "actual_played",
            "actual_points",
        ])
    totals = rows.groupby("player_id", as_index=False)[cols].sum(min_count=1)
    if "games_played" in rows.columns:
        games = rows.groupby("player_id")["games_played"].max()
        totals["actual_games_played"] = totals["player_id"].map(games).fillna(0.0)
    else:
        totals["actual_games_played"] = np.nan
    totals["actual_row_present"] = True
    totals["actual_played"] = totals["actual_games_played"].gt(0)
    totals["actual_points"] = _score_totals(totals)
    return totals


def _team_anchor_predictions(
    history: pd.DataFrame, source_season: int, pairs: list[tuple[int, int]]
) -> pd.DataFrame:
    labels = {label: pred_col for label, _key, pred_col in TEAM_ANCHOR_SPECS}
    source = (
        history[history["season"].eq(source_season)]
        .dropna(subset=["team"])
        .drop_duplicates("team")
        .copy()
    )
    out = source[["team"]].copy()
    for label, output_col in labels.items():
        model, _ = fit_team_total(history, pairs=pairs, label_col=label)
        inputs = source[TEAM_FEATURES].copy()
        inputs["team_naive_pred"] = source[label]
        out[output_col] = np.clip(model.predict(inputs[TEAM_MODEL_FEATURES]), 0, None)
    # Provenance columns carry the same meaning as canonical_team_anchor_frame's
    # in the shipped path, so propagate_team_anchors validates this frame with
    # the same invariants. The value differs only in how the model was fitted -
    # here, refit on pairs bounded at source_season.
    out["team_total_pred"] = out["team_passing_yards_pg_pred"]
    out["team_anchor_source_season"] = int(source_season)
    out["team_anchor_lag_team"] = out["team"]
    out["team_anchor_provenance"] = "leakage_safe_source_team_frame"
    return out


def _veteran_forecasts(
    conn,
    history: pd.DataFrame,
    population: pd.DataFrame,
    source_season: int,
    target_season: int,
    pairs: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rookies = set(population.loc[population["is_rookie"], "player_id"])
    eligible = population[~population["player_id"].isin(rookies)].copy()
    source = history[history["season"].eq(source_season)].copy()
    source = source.sort_values("games_played", ascending=False).drop_duplicates("player_id")
    base = eligible.merge(source, on="player_id", how="inner", suffixes=("_pre", ""))
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()
    base["team"] = base["preseason_team"]
    base["position"] = base["preseason_position"]

    # Team context is re-pointed to the target team using source-season data;
    # no 2025 realized team tendency enters a moved player's row.
    team_context = source.dropna(subset=["team"]).drop_duplicates("team").set_index("team")
    for col in TEAM_FEATURES:
        base[col] = base["team"].map(team_context[col]).fillna(base[col])
    base = attach_availability_depth_rank(base, target_season, conn=conn)
    base = attach_depth_rank(base, target_season, conn=conn)

    games = []
    rates = []
    for position, stats in TARGET_STATS.items():
        idx = base["position"].eq(position)
        if not idx.any():
            continue
        model, _ = fit_availability(history, position, pairs=pairs)
        predicted_games = np.clip(
            model.predict(base.loc[idx, AVAILABILITY_FEATURES]), 0, SEASON_GAMES
        )
        games.append(pd.DataFrame({
            "player_id": base.loc[idx, "player_id"].to_numpy(),
            "projected_games": predicted_games,
        }))
        for stat in stats:
            rate_model, _ = fit_one(history, position, stat, pairs=pairs)
            pred = np.clip(age_shrunk_predict(rate_model, base.loc[idx], position), 0, None)
            # Production applies the veteran-only depth ladder before share
            # composition.  The held-out chart is available preseason; the
            # factor is deterministic and contains no target outcome.
            factors = np.array([
                depth_rate_factor(position, rank)
                for rank in base.loc[idx, "nfl_depth_rank"]
            ], dtype=float)
            pred = pred * factors
            rates.append(pd.DataFrame({
                "player_id": base.loc[idx, "player_id"].to_numpy(),
                "position": position,
                "team": base.loc[idx, "team"].to_numpy(),
                "season": int(target_season),
                "stat": stat,
                "pred_pg": pred,
                "depth_rate_factor": factors,
                "is_receiving_share": (position, stat) in REFRAMED_SHARE_STATS,
            }))
    game_df = pd.concat(games, ignore_index=True).drop_duplicates("player_id")
    rate_df = pd.concat(rates, ignore_index=True).merge(game_df, on="player_id", how="left")
    depth = base[["player_id", "target_depth_rank", "nfl_depth_rank"]].drop_duplicates("player_id")
    return rate_df.merge(depth, on="player_id", how="left"), game_df


def _rookie_forecasts(
    rookie_cohort: pd.DataFrame,
    source_season: int,
    target_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baselines = fit_rookie_baselines(
        rookie_cohort, train_seasons=list(range(int(rookie_cohort["season"].min()), source_season + 1))
    )
    safe = sanitize_target_rookie_outcomes(rookie_cohort, target_season)
    wide = predict_rookies(safe, baselines, [target_season], depth_chart=None)
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for position, stats in TARGET_STATS.items():
        part = wide[wide["position"].eq(position)]
        for stat in stats:
            col = f"{stat}_pg"
            if col not in part:
                continue
            rows.append(pd.DataFrame({
                "player_id": part["player_id"].to_numpy(),
                "position": position,
                "team": part["team"].to_numpy(),
                "season": int(target_season),
                "stat": stat,
                "pred_pg": part[col].to_numpy(),
                "is_receiving_share": False,
                "depth_rate_factor": 1.0,
                "projected_games": part["projected_games"].to_numpy(),
                "target_depth_rank": part.get("target_depth_rank", pd.Series(np.nan, index=part.index)).to_numpy(),
                "nfl_depth_rank": part.get("nfl_depth_rank", pd.Series(np.nan, index=part.index)).to_numpy(),
            }))
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return long, wide[["player_id", "projected_games"]].drop_duplicates("player_id")


def _compose_and_reconcile(
    veteran_long: pd.DataFrame,
    rookie_long: pd.DataFrame,
    anchors: pd.DataFrame,
    context,
) -> pd.DataFrame:
    """Run the SHIPPED composition pipeline over leakage-safe artifacts.

    This function no longer contains any allocation logic of its own. It
    attaches the refit team anchors, hands the veteran share rows and the
    rookies' implied shares to the same ``_compose_reframed_receiving_
    predictions`` the shipped path uses, and then calls ``compose_board`` —
    which is literally the stage list ``project_season`` runs. Anything this
    harness scores is therefore something the shipped board also does.
    """
    veteran = veteran_long.merge(anchors, on="team", how="left")
    rookie = (
        rookie_long.merge(anchors, on="team", how="left")
        if not rookie_long.empty else rookie_long
    )

    # Rookie receiving enters the veteran share denominator as implied share -
    # the Robinson/Tate case: an incoming rookie consumes real target share the
    # veteran share models cannot see. Same input the shipped path passes.
    rookie_receiving = (
        rookie.loc[rookie["stat"].eq("receiving_yards"),
                   ["team", "pred_pg", "projected_games"]]
        if not rookie.empty else None
    )
    veteran = _compose_reframed_receiving_predictions(
        veteran,
        _NO_LEAKAGE_SAFE_RESIDUALS,
        rookie_receiving=rookie_receiving,
        # Elite shrinkage ships in models/corrections.joblib, fit on residuals
        # spanning the target season. There is no leakage-safe refit of it on
        # this path, so it is omitted rather than leaked - see coverage_limits.
        corrections=None,
    )

    out = pd.concat([veteran, rookie], ignore_index=True, sort=False)
    out["pred_pg"] = pd.to_numeric(out["pred_pg"], errors="coerce").clip(lower=0)
    # No leakage-safe interval residuals exist for this fold, so the endpoints
    # collapse onto the point prediction rather than being borrowed. Nothing in
    # the scoring path reads them; they exist so the reconcilers, which scale
    # all three together, have a defined value to scale.
    for col in ("pred_pg_low", "pred_pg_high"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(out["pred_pg"])
    return compose_board(out, context)


def _forecast_from_history(
    conn,
    history: pd.DataFrame,
    population: pd.DataFrame,
    rookie_cohort: pd.DataFrame,
    source_season: int,
    target_season: int,
) -> pd.DataFrame:
    """Internal forecast stage; `history` must contain no target outcomes."""
    if history["season"].max() > source_season:
        raise ValueError("forecast history extends beyond source_season")
    pairs = safe_training_pairs(source_season)
    if not pairs:
        raise ValueError("no safe production-era training transitions")
    anchors = _team_anchor_predictions(history, source_season, pairs)
    veteran_long, veteran_games = _veteran_forecasts(
        conn, history, population, source_season, target_season, pairs
    )
    rookie_long, rookie_games = _rookie_forecasts(
        rookie_cohort, source_season, target_season
    )
    context = leakage_safe_context(conn, target_season, source_season)
    long = _compose_and_reconcile(veteran_long, rookie_long, anchors, context)
    scored_stats = long.pivot_table(
        index="player_id", columns="stat", values="pred_season", aggfunc="first"
    )
    scored = _score_totals(scored_stats).rename("model_forecast_points").reset_index()
    games = pd.concat([veteran_games, rookie_games], ignore_index=True).drop_duplicates("player_id")
    exposure = long[["player_id", "projected_volume_games"]].drop_duplicates("player_id")
    factors = long[["player_id", "depth_rate_factor"]].drop_duplicates("player_id")
    scored = (
        scored.merge(games, on="player_id", how="left")
        .merge(exposure, on="player_id", how="left")
        .merge(factors, on="player_id", how="left")
    )
    expected_stat_count = population["preseason_position"].map(
        {position: len(stats) for position, stats in TARGET_STATS.items()}
    )
    stat_count = long.groupby("player_id")["stat"].nunique()
    scored["forecast_component_count"] = scored["player_id"].map(stat_count).fillna(0).astype(int)
    expected = population.set_index("player_id")["preseason_position"].map(
        {position: len(stats) for position, stats in TARGET_STATS.items()}
    )
    scored["forecast_expected_component_count"] = scored["player_id"].map(expected)
    scored["forecast_covered"] = (
        scored["forecast_component_count"].eq(scored["forecast_expected_component_count"])
        & scored["model_forecast_points"].notna()
    )
    # Carried out so run_evaluation can publish exactly which composition stages
    # ran on real inputs for this fold and which degraded to pass-throughs.
    # Coverage must never be readable as performance.
    scored.attrs["stage_coverage"] = context.describe_coverage()
    scored.attrs["artifact_provenance"] = context.artifact_provenance
    return scored


def build_leakage_safe_forecasts(
    conn,
    feature_table: pd.DataFrame,
    source_season: int = 2024,
    target_season: int = 2025,
) -> tuple[pd.DataFrame, dict]:
    """Freeze population and forecast without exposing any 2025 outcomes."""
    history = feature_table[feature_table["season"].le(source_season)].copy()
    rookie_cohort = build_preseason_rookie_cohort(
        conn, feature_table, list(range(2016, target_season + 1))
    )
    population = build_preseason_population(conn, target_season, rookie_cohort)
    forecasts = _forecast_from_history(
        conn,
        history,
        population,
        sanitize_target_rookie_outcomes(rookie_cohort, target_season),
        source_season,
        target_season,
    )
    out = population.merge(forecasts, on="player_id", how="left")
    out["forecast_covered"] = out["forecast_covered"].fillna(False)
    out["model_points_end_to_end"] = out["model_forecast_points"].fillna(0.0)

    prior = _actual_player_totals(history, source_season).rename(
        columns={"actual_points": "carry_forward_points"}
    )
    prior_games = (
        history[history["season"].eq(source_season)]
        .groupby("player_id", as_index=False)["games_played"].max()
        .rename(columns={"games_played": "source_games_played"})
    )
    out = out.merge(
        prior[["player_id", "carry_forward_points"]], on="player_id", how="left"
    ).merge(prior_games, on="player_id", how="left")
    out["carry_forward_points"] = out["carry_forward_points"].fillna(0.0)
    source_rate = out["carry_forward_points"] / out["source_games_played"].replace(0, np.nan)
    baseline_exposure = out["projected_volume_games"].fillna(out["projected_games"])
    out["availability_adjusted_points"] = (
        source_rate.fillna(0.0) * baseline_exposure.fillna(0.0)
    )
    metadata = {
        "source_season": int(source_season),
        "target_season": int(target_season),
        "training_pairs": pairs_to_json(safe_training_pairs(source_season)),
        "population_definition": (
            "earliest regular-season weekly_rosters snapshot; statuses "
            "ACT/DEV/RES/INA/EXE; position/team frozen before outcomes"
        ),
        "population_n": int(len(out)),
        "rookie_n": int(out["is_rookie"].sum()),
        "forecast_covered_n": int(out["forecast_covered"].sum()),
        "composition_pipeline": (
            "src.projection.composition.compose_board - the same stage sequence "
            "src.projection.predict.project_season ships"
        ),
        "composition_artifact_provenance": forecasts.attrs.get("artifact_provenance"),
        "composition_stage_coverage": forecasts.attrs.get("stage_coverage", {}),
    }
    return out, metadata


def pairs_to_json(pairs: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in pairs]


def attach_actual_outcomes(
    forecasts: pd.DataFrame, feature_table: pd.DataFrame, target_season: int
) -> pd.DataFrame:
    """Attach held-out component totals only after forecast construction."""
    actual = _actual_player_totals(feature_table, target_season)
    out = forecasts.merge(actual, on="player_id", how="left")
    for col in [*OUTCOME_TOTAL_COLUMNS, "actual_points"]:
        if col in out:
            out[col] = out[col].fillna(0.0)
    out["actual_row_present"] = out["actual_row_present"].fillna(False).astype(bool)
    out["actual_played"] = out["actual_played"].fillna(False).astype(bool)
    out["actual_games_played"] = out["actual_games_played"].fillna(0.0)
    out["actual_zero_game_outcome"] = out["actual_games_played"].le(0)
    out["actual_zero_point_outcome"] = out["actual_points"].eq(0.0)
    return out


def add_average_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Average ranks preserve ties (especially the zero-outcome tail)."""
    out = frame.copy()
    methods = {
        "actual_position_finish": "actual_points",
        "model_position_rank": "model_points_end_to_end",
        "carry_forward_position_rank": "carry_forward_points",
        "availability_adjusted_position_rank": "availability_adjusted_points",
    }
    for rank_col, value_col in methods.items():
        out[rank_col] = out.groupby("preseason_position")[value_col].rank(
            method="average", ascending=False
        )
    return out


def _kth_score(values: pd.Series, rank: int) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if values.empty:
        return float("nan")
    return float(values.iloc[min(max(int(rank), 1), len(values)) - 1])


def _metric_rows(
    frame: pd.DataFrame,
    tier_ranks: dict[str, int],
    replacement_ranks: dict[str, int],
) -> pd.DataFrame:
    methods = {
        "model": "model_points_end_to_end",
        "carry_forward": "carry_forward_points",
        "availability_adjusted": "availability_adjusted_points",
    }
    rows = []
    for position in POSITIONS:
        position_rows = frame[frame["preseason_position"].eq(position)]
        for scope, scoped in (
            ("all_eligible", position_rows),
            ("forecast_covered", position_rows[position_rows["forecast_covered"]]),
        ):
            if scoped.empty:
                continue
            actual_cut = _kth_score(scoped["actual_points"], tier_ranks[position])
            actual_top = scoped["actual_points"].ge(actual_cut)
            actual_replacement = _kth_score(
                scoped["actual_points"], replacement_ranks[position]
            )
            actual_vorp = scoped["actual_points"] - actual_replacement
            for method, value_col in methods.items():
                pred_cut = _kth_score(scoped[value_col], tier_ranks[position])
                pred_top = scoped[value_col].ge(pred_cut)
                hits = int((pred_top & actual_top).sum())
                pred_replacement = _kth_score(
                    scoped[value_col], replacement_ranks[position]
                )
                pred_vorp = scoped[value_col] - pred_replacement
                rows.append({
                    "position": position,
                    "scope": scope,
                    "method": method,
                    "n": int(len(scoped)),
                    "population_n": int(len(position_rows)),
                    "forecast_covered_n": int(position_rows["forecast_covered"].sum()),
                    "zero_outcome_n": int(scoped["actual_zero_game_outcome"].sum()),
                    "spearman": float(scoped[value_col].corr(scoped["actual_points"], method="spearman")),
                    "points_mae": float((scoped[value_col] - scoped["actual_points"]).abs().mean()),
                    "tier_rank": int(tier_ranks[position]),
                    "predicted_top_n": int(pred_top.sum()),
                    "actual_top_n": int(actual_top.sum()),
                    "tier_hits": hits,
                    "tier_precision": hits / int(pred_top.sum()) if pred_top.any() else float("nan"),
                    "tier_recall": hits / int(actual_top.sum()) if actual_top.any() else float("nan"),
                    "tier_hit_rate": hits / min(int(pred_top.sum()), int(actual_top.sum())) if pred_top.any() and actual_top.any() else float("nan"),
                    "replacement_rank": int(replacement_ranks[position]),
                    "predicted_replacement_points": pred_replacement,
                    "actual_replacement_points": actual_replacement,
                    "vorp_mae": float((pred_vorp - actual_vorp).abs().mean()),
                    "vorp_spearman": float(pred_vorp.corr(actual_vorp, method="spearman")),
                })
    return pd.DataFrame(rows)


def evaluate_forecasts(
    frame: pd.DataFrame,
    tier_ranks: dict[str, int] | None = None,
    replacement_ranks: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tier_ranks = tier_ranks or DEFAULT_TIER_RANKS
    replacement_ranks = replacement_ranks or DEFAULT_REPLACEMENT_RANKS
    ranked = add_average_ranks(frame)
    return ranked, _metric_rows(ranked, tier_ranks, replacement_ranks)


def _parse_rank_map(value: str, defaults: dict[str, int]) -> dict[str, int]:
    if not value:
        return defaults.copy()
    parsed = defaults.copy()
    for item in value.split(","):
        position, rank = item.split("=", 1)
        position = position.strip().upper()
        if position not in POSITIONS:
            raise ValueError(f"unknown position in rank map: {position}")
        parsed[position] = int(rank)
    return parsed


def run_evaluation(
    source_season: int = 2024,
    target_season: int = 2025,
    tier_ranks: dict[str, int] | None = None,
    replacement_ranks: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    conn = get_conn()
    try:
        feat = build_player_season_features(conn)
        forecasts, metadata = build_leakage_safe_forecasts(
            conn, feat, source_season, target_season
        )
    finally:
        conn.close()
    with_actual = attach_actual_outcomes(forecasts, feat, target_season)
    ranked, summary = evaluate_forecasts(
        with_actual, tier_ranks=tier_ranks, replacement_ranks=replacement_ranks
    )
    metadata["tier_ranks"] = tier_ranks or DEFAULT_TIER_RANKS
    metadata["replacement_ranks"] = replacement_ranks or DEFAULT_REPLACEMENT_RANKS
    metadata["coverage_limits"] = [
        "Week-1 contracted roster snapshot excludes August camp cuts.",
        "Veterans without a source-season feature row remain in all_eligible and score as zero model points.",
        "Composition/allocation is the shipped pipeline itself (composition.compose_board), "
        "run over artifacts refit on seasons <= source_season. See "
        "composition_stage_coverage for the per-stage active/degraded map.",
        "STAGES THAT CANNOT BE MEASURED ON A HISTORICAL FOLD, and why: "
        "(a) curated depth-chart membership, roles, formation roles and reviewed "
        "usage-share priors - src/depth_chart/starters_<season>.csv is hand-researched "
        "and exists for 2026 only, so apply_curated_availability_override, "
        "apply_depth_chart_gating's curated branch, replacement-level rows for curated "
        "players neither model path reaches, and the LWR/RWR/SWR within-WR split all "
        "no-op; (b) dated status overrides (IR/PUP) - status_overrides_<season>.csv is "
        "likewise 2026-only; (c) elite residual correction - models/corrections.joblib "
        "is fit on residuals spanning the target season; (d) prediction intervals - "
        "models/interval_residuals.csv is fit by backtest.py across the target season. "
        "None of these are faked or silently skipped: each degrades to an explicit "
        "pass-through recorded in composition_stage_coverage.",
        "Roster reassignment (reassign_team_changers) is not run: the target team comes "
        "from the frozen Week-1 roster, which is a stricter preseason source than the "
        "seasonal_rosters lookup the shipped path uses.",
        "No Sleeper or other external projection is used.",
    ]
    return ranked, summary, metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-season", type=int, default=2024)
    parser.add_argument("--target-season", type=int, default=2025)
    parser.add_argument("--tier-ranks", default="QB=12,RB=24,WR=36,TE=12")
    parser.add_argument("--replacement-ranks", default="QB=13,RB=25,WR=37,TE=13")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    tier = _parse_rank_map(args.tier_ranks, DEFAULT_TIER_RANKS)
    replacement = _parse_rank_map(args.replacement_ranks, DEFAULT_REPLACEMENT_RANKS)
    rows, summary, metadata = run_evaluation(
        args.source_season, args.target_season, tier, replacement
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / f"fantasy_evaluation_{args.target_season}.csv"
    summary_path = output / f"fantasy_evaluation_summary_{args.target_season}.csv"
    json_path = output / f"fantasy_evaluation_summary_{args.target_season}.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps({"metadata": metadata, "metrics": summary.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {rows_path}, {summary_path}, and {json_path}")


if __name__ == "__main__":
    main()
