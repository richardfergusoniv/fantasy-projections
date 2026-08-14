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

from src.projection.data_prep import get_conn
from src.projection.depth_history import (
    attach_availability_depth_rank,
    attach_depth_rank,
)
from src.projection.fantasy_points import SCORING
from src.projection.features import TARGET_STATS, build_player_season_features
from src.projection.predict import (
    add_projected_season_totals,
    apply_usage_share_prior,
    depth_rate_factor,
    fit_usage_share_priors,
    normalize_team_passing_volume,
    normalize_team_rushing_volume,
    reconcile_qb_projected_volume_games,
    reconcile_stat_constraints,
    reconcile_team_pass_receive_counts,
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
    TEAM_ATTEMPTS_LABEL,
    age_shrunk_predict,
    TEAM_CARRIES_LABEL,
    TEAM_FEATURES,
    TEAM_MODEL_FEATURES,
    TEAM_RUSH_YARDS_LABEL,
    TEAM_TOTAL_LABEL,
    receiving_share_scale,
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
    labels = {
        TEAM_TOTAL_LABEL: "team_passing_yards_pg_pred",
        TEAM_ATTEMPTS_LABEL: "team_pass_attempts_pg_pred",
        TEAM_CARRIES_LABEL: "team_carries_pg_pred",
        TEAM_RUSH_YARDS_LABEL: "team_rushing_yards_pg_pred",
    }
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
    long: pd.DataFrame, anchors: pd.DataFrame, target_season: int, conn=None
) -> pd.DataFrame:
    out = long.merge(anchors, on="team", how="left")
    share_mask = out["is_receiving_share"].fillna(False)
    if share_mask.any():
        share = out.loc[share_mask, ["team", "pred_pg", "projected_games"]].copy()
        share["share"] = share["pred_pg"]
        share["weight"] = (share["projected_games"] / SEASON_GAMES).clip(0, 1)
        rookie_recv = out[
            ~share_mask & out["position"].isin(["RB", "WR", "TE"])
            & out["stat"].eq("receiving_yards")
        ].copy()
        rookie_recv["anchor"] = rookie_recv["team_passing_yards_pg_pred"]
        rookie_extra = (
            rookie_recv["pred_pg"] / rookie_recv["anchor"].replace(0, np.nan)
            * (rookie_recv["projected_games"] / SEASON_GAMES).clip(0, 1)
        ).groupby(rookie_recv["team"]).sum()
        scale, _ = receiving_share_scale(
            share[["team", "share", "weight"]], extra_team_share=rookie_extra
        )
        out.loc[share_mask, "pred_pg"] = (
            out.loc[share_mask, "pred_pg"].to_numpy()
            * scale.to_numpy()
            * out.loc[share_mask, "team_passing_yards_pg_pred"].to_numpy()
        )
    out["pred_pg"] = pd.to_numeric(out["pred_pg"], errors="coerce").clip(lower=0)
    out["pred_pg_low"] = out["pred_pg"]
    out["pred_pg_high"] = out["pred_pg"]
    out = reconcile_stat_constraints(out)
    out = reconcile_qb_projected_volume_games(out, season_games=SEASON_GAMES)
    # Same room reordering predict.py applies, fit strictly on seasons before
    # this fold's target so the evaluation stays leakage-safe. This is the
    # only place the blend can be scored against real outcomes rather than
    # against consensus.
    if conn is not None:
        out = apply_usage_share_prior(
            out, fit_usage_share_priors(conn, list(range(2016, target_season))))
    out = normalize_team_passing_volume(out, season_games=SEASON_GAMES)
    out = normalize_team_rushing_volume(out, season_games=SEASON_GAMES)
    out = reconcile_team_pass_receive_counts(out, season_games=SEASON_GAMES)
    out = reconcile_stat_constraints(out)
    return add_projected_season_totals(out)


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
    long = pd.concat([veteran_long, rookie_long], ignore_index=True, sort=False)
    long = _compose_and_reconcile(long, anchors, target_season, conn=conn)
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
        "The shipped veteran depth-rate ladder and current team/QB/stat reconciliation are applied; historical curated roles, target-year coordinator context, and elite residual correction remain unavailable.",
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
