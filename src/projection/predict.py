"""Phase 5 entry point: generate per-game rate projections for a target
season, combining the veteran LightGBM models (trained in train.py, saved
under models/) and the rookie rule-based path (rookies.py). Does not
re-derive anything from Phase 2/3 - just loads saved artifacts and this
phase's feature-building code.

Usage: `python -m src.projection.predict --season 2026`, or import
`project_season(conn, season)` directly.

Output: one DataFrame, one row per (player_id, position, stat), with a
`source` column ('veteran_model' | 'rookie_rule') and `low_confidence`
(True for every rookie row, per the hard project rule that rookie
projections must be flagged separately from veteran ones - never silently
mixed in as equally-confident numbers).

Prediction intervals (`pred_pg_low`/`pred_pg_high`, 10th/90th empirical
percentile, i.e. an 80% interval - see PHASE5_REPORT.md for why this width
and why empirical over a second quantile-regression model): veteran rows
use `models/interval_residuals.csv` (built by `backtest.py` from the SAME
2025 held-out backtest as the MAE table - genuine out-of-sample residuals,
added to pred_pg). Rookie rows have no naive-baseline backtest to draw
residuals from, so they use a different, multiplicative fallback (within-
bucket historical ratio of actual/bucket-mean per-game rate - see
rookies.py's rookie_interval_ratios) instead of reusing the veteran
residuals, since rookie-year variance is a distinct, larger regime.
`models/interval_residuals.csv` must exist before calling this (run
`python -m src.projection.backtest` once, after `train.py`).

Framing caveat carried from train.py/transitions.py: veteran projections
for season N+1 use season N's own observed opportunity/scheme features as
the best available proxy for season N+1 conditions (season N+1 hasn't been
played yet, so its own oc_tendency_profiles/OL-quality rows don't exist).
If a team's OC situation is KNOWN to change entering the target season
(e.g. a newly hired play-caller), the caller should override the relevant
oc_tendency_profiles columns with that OC's `inherited_*` row before
calling this function - this module does not do that substitution
automatically, since it doesn't know how new the coaching sitution is,
only what's already computed in the DB for completed seasons.

Phase B: stage logic lives in sibling modules; this file orchestrates
`project_season` / export / CLI and re-exports public names for tests.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.projection.data_prep import get_conn
from src.projection.depth_history import (
    DEPTH_TIER_COLUMN,
    attach_availability_depth_rank,
    attach_depth_rank,
    depth_tiers,
)
from src.projection.features import build_player_season_features, TARGET_STATS
from src.projection.rookies import (
    build_rookie_dataset, fit_rookie_baselines, predict_rookies,
    identify_target_season_rookie_class,
    team_vacated_opportunity, rookie_interval_ratios, combine_athletic_scores_by_pfr_id,
)
from src.projection.transitions import SEASON_GAMES
from src.projection.composition import CompositionContext, compose_board, shipped_context
from src.projection.contracts import (
    REPO_ROOT,
    MODELS_DIR,
    OUTPUT_DIR,
    INTERVAL_RESIDUALS_PATH,
    CORRECTIONS_PATH,
    DEPTH_CHART_PATH,
    LIVE_DEPTH_CHART_PATH,
    STATUS_OVERRIDES_PATH,
    CURATED_RESEARCH_DEPTH,
    DEEP_BENCH_GAMES_CAP,
    ROOKIE_RATIO_FALLBACK,
    VACATED_CLIP,
    TEAM_CHANGE_SHARE_CLIP,
    DEPTH_RATE_LADDER,
    DEPTH_RATE_DEEP,
    DEPTH_RATE_OFF_CHART,
    BOOST_ELIGIBLE_ROLES,
    INCUMBENT_VACANCY_ALPHA,
    TEAM_CHANGE_VACANCY_ALPHA,
    INCUMBENT_VACANCY_NET_CLIP,
    INCUMBENT_VACANCY_SCALE_CAP,
    REPLACEMENT_POSITIONS,
    REPLACEMENT_MIN_CELL,
    REPLACEMENT_DEPTH_BANDS,
    RUSH_ATTEMPTS_PER_APPEARANCE_MAX,
    RUSH_YARDS_PER_CARRY_MAX,
    OL_TRAILING_SEASONS,
    TEAM_ANCHOR_OUTPUT_COLS,
    OUTPUT_COLUMNS,
)
from src.projection.depth_rates import depth_rate_factor
from src.projection.artifacts import (
    load_availability_models,
    load_models,
    load_interval_residuals,
    load_corrections,
)
from src.projection.depth_gating import (
    load_depth_chart,
    load_status_overrides,
    apply_curated_availability_override,
    apply_status_overrides,
    apply_full_season_games_baseline,
    enforce_availability_chart_review,
    apply_curated_depth_tier,
    apply_depth_chart_gating,
)
from src.projection.roster_moves import (
    TEAM_CONTEXT_COLS,
    load_target_roster_map,
    _incoming_volume_share,
    _effective_vacated_target,
    _attach_rookie_residual_vacancy,
    drop_players_absent_from_target_season,
    apply_incumbent_vacancy_boost,
    reassign_team_changers,
)
from src.projection.replacement import (
    _replacement_depth_band,
    fit_replacement_level_baselines,
    build_replacement_level_rows,
    _replacement_availability,
)
from src.projection.veterans import (
    project_veterans,
    _warn_availability_chart_disagreement,
    _warn_discounted_high_usage,
    TRIPWIRE_SEASON_TARGETS,
    TRIPWIRE_TARGETS_PG,
    TRIPWIRE_REC_YPG,
    TRIPWIRE_SEASON_ATTEMPTS,
    TRIPWIRE_ATTEMPTS_PG,
    TRIPWIRE_SEASON_CARRIES,
    TRIPWIRE_CARRIES_PG,
    TRIPWIRE_SEVERE_FACTOR,
    TRIPWIRE_RUSH_MIN_GAMES,
)
from src.projection.team_reconcile import (
    TEAM_ANCHOR_SPECS,
    canonical_team_anchor_frame,
    _attach_team_total_pred,
    _HELPER_COLS,
    ELITE_COMPANION_STATS,
    _propagate_elite_correction,
    _compose_reframed_receiving_predictions,
    reconcile_stat_constraints,
    _row_exposure,
    add_projected_season_totals,
    _apply_rookie_depth_rate_gating,
    propagate_team_anchors,
)

# Re-export contracts for backward-compatible `from src.projection.predict import …`.
__all_contracts__ = [
    "REPO_ROOT", "MODELS_DIR", "OUTPUT_DIR", "INTERVAL_RESIDUALS_PATH",
    "CORRECTIONS_PATH", "DEPTH_CHART_PATH", "LIVE_DEPTH_CHART_PATH",
    "STATUS_OVERRIDES_PATH", "CURATED_RESEARCH_DEPTH", "DEEP_BENCH_GAMES_CAP",
    "ROOKIE_RATIO_FALLBACK", "VACATED_CLIP", "TEAM_CHANGE_SHARE_CLIP",
    "DEPTH_RATE_LADDER", "DEPTH_RATE_DEEP", "DEPTH_RATE_OFF_CHART",
    "BOOST_ELIGIBLE_ROLES", "INCUMBENT_VACANCY_ALPHA", "TEAM_CHANGE_VACANCY_ALPHA",
    "INCUMBENT_VACANCY_NET_CLIP", "INCUMBENT_VACANCY_SCALE_CAP",
    "REPLACEMENT_POSITIONS", "REPLACEMENT_MIN_CELL", "REPLACEMENT_DEPTH_BANDS",
    "RUSH_ATTEMPTS_PER_APPEARANCE_MAX", "RUSH_YARDS_PER_CARRY_MAX",
    "OL_TRAILING_SEASONS", "TEAM_ANCHOR_OUTPUT_COLS", "OUTPUT_COLUMNS",
]


def project_season(conn, target_season, as_of=None):
    """Project `target_season` per-game rates using `target_season - 1`
    features for veterans, and the rookie rule-based path for
    `target_season`'s actual draft-year rookies.

    ``as_of`` (optional ISO date) filters status overrides and selects the
    latest nflverse daily depth snapshot on or before that date for 2025+.
    """
    source_season = target_season - 1
    models = load_models()
    resid = load_interval_residuals()

    feat = build_player_season_features(
        conn,
        seasons=list(range(2016, target_season)),
        ol_trailing_for_seasons={source_season},
    )
    vet, rookie_residual = project_veterans(
        conn, feat, source_season, models, resid, target_season, as_of=as_of)
    vet["season"] = target_season

    # Rookie path: bucket rates and availability are fit on full preseason
    # cohorts from completed seasons, including zero-game rookies, then
    # applied to the target class from draft_picks + seasonal_rosters.
    hist_seasons = list(range(2016, target_season))
    hist_feat = build_player_season_features(conn, seasons=hist_seasons)
    rdf = build_rookie_dataset(conn, hist_feat, seasons=hist_seasons)
    baselines = fit_rookie_baselines(rdf, hist_seasons)
    ratios = rookie_interval_ratios(rdf, baselines, hist_seasons)

    target_class = identify_target_season_rookie_class(conn, target_season)
    vacated = team_vacated_opportunity(conn, [target_season])
    target_class = target_class.merge(vacated, on=["season", "team"], how="left")
    # Combine-athleticism tier (Addendum 4, Part 3) - joined via pfr_id
    # (identify_target_season_rookie_class now carries draft_picks'
    # pfr_player_id / seasonal_rosters' pfr_id directly), NOT via player_id,
    # because target_season's drafted rookies have a placeholder gsis_id
    # (see that function's docstring) that would silently fail to match
    # combine_athletic_scores' player_id-keyed form for nearly the entire
    # drafted class. 'no_data' (not NaN) for any rookie with no combine_data
    # match at all (didn't test, or genuinely absent from the pull), so
    # predict_rookies' scale lookup always resolves.
    athletic = combine_athletic_scores_by_pfr_id(conn)
    target_class = target_class.merge(athletic, on="pfr_id", how="left")
    target_class["athletic_tier"] = target_class["athletic_tier"].fillna("no_data")
    # Canonical rookie IDs now resolve against the same preseason chart used
    # by veteran availability. This is an internal, historically testable
    # availability signal; Sleeper projections are comparison-only.
    target_class = attach_availability_depth_rank(
        target_class, target_season, conn=conn, as_of=as_of)
    target_class = attach_depth_rank(target_class, target_season, conn=conn, as_of=as_of)
    depth_chart = load_depth_chart(target_season)
    target_class = apply_curated_availability_override(target_class, depth_chart)
    status_overrides = load_status_overrides(target_season, as_of=as_of)
    enforce_availability_chart_review(
        target_class, depth_chart, status_overrides, target_season, conn=conn)
    # Vacancy still unclaimed after the veteran paths took their cut - see
    # _attach_rookie_residual_vacancy. Without it a rookie is scaled by the
    # team's whole gross vacancy while incumbents are simultaneously
    # credited with absorbing part of the same opening.
    for kind in ("carry", "target"):
        col = f"rookie_residual_{kind}_fraction"
        target_class[col] = (
            target_class["team"].map(rookie_residual[col]).fillna(1.0)
            if col in rookie_residual.columns else 1.0
        )
    rookie_preds = predict_rookies(target_class, baselines, [target_season], depth_chart=depth_chart)

    pg_cols = [c for c in rookie_preds.columns if c.endswith("_pg")]
    rookie_long = rookie_preds.melt(
        id_vars=["player_id", "team", "position", "season", "rookie_tier", "round_bucket",
                 "projected_games", "athletic_tier", "target_depth_rank",
                 "nfl_depth_rank", "rookie_depth_band", "rookie_vacancy_scale",
                 "rookie_availability_cell_n", "rookie_availability_fallback_used",
                 "rookie_id_unresolved"],
        value_vars=pg_cols, var_name="stat", value_name="pred_pg",
    )
    rookie_long["stat"] = rookie_long["stat"].str.replace("_pg", "", regex=False)
    rookie_long["source"] = "rookie_rule"
    rookie_long["low_confidence"] = True
    rookie_long = rookie_long.dropna(subset=["pred_pg"])
    rookie_long = rookie_long[
        rookie_long.apply(lambda r: r["stat"] in TARGET_STATS.get(r["position"], []), axis=1)
    ]

    rookie_long = _attach_rookie_intervals(rookie_long, ratios)

    # Rookies already use the correct target-season team. Attach curated role
    # for transparency and for auditing the role-filtered vacancy gate that
    # ran in predict_rookies. Rookie conditional rates remain neutral because
    # the depth-rate ladder was fit only on veteran transition pairs.
    depth_chart = load_depth_chart(target_season)
    rookie_long["team_changed"] = False
    rookie_long["roster_status"] = np.nan
    if not depth_chart.empty:
        dc = depth_chart[["position", "gsis_id", "depth_rank", "role"]].rename(columns={"gsis_id": "player_id"})
        dc = dc.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
        rookie_long = rookie_long.merge(dc, on=["player_id", "position"], how="left")
    else:
        rookie_long["depth_rank"], rookie_long["role"] = np.nan, None
    rookie_long = _attach_rookie_depth_tier(rookie_long, depth_chart)
    rookie_long["depth_chart_status"] = "rookie_path"
    # projected_games was estimated on the full historical rookie cohort in
    # rookies.py, by position/draft bucket and preseason depth band — keep
    # that in projected_games_raw; draft exposure is a full season except
    # IR/PUP/Sus overrides.
    rookie_long = _apply_rookie_depth_rate_gating(rookie_long)
    if not depth_chart.empty:
        curated_ids = set(depth_chart.dropna(subset=["gsis_id"])["gsis_id"])
        off = ~rookie_long["player_id"].isin(curated_ids)
        rookie_long.loc[off, "depth_chart_status"] = "deep_bench_discounted"
        rookie_long.loc[off, "role"] = "deep_bench"
    rookie_long = apply_full_season_games_baseline(rookie_long, season_games=SEASON_GAMES)
    rookie_long = apply_status_overrides(rookie_long, status_overrides)

    # Compose the veteran reframed receiving shares into real per-game
    # rates now that the rookie path exists: rookie receiving predictions
    # enter the share-sum denominator as implied shares (Phase 2 of the
    # consensus-gap work - the user-diagnosed Robinson/Tate case, where a
    # 1st-round rookie's incoming target share must squeeze the veterans).
    rookie_receiving = rookie_long[rookie_long["stat"] == "receiving_yards"][[
        "team", "pred_pg", "projected_games"]]
    vet = _compose_reframed_receiving_predictions(
        vet, resid, rookie_receiving=rookie_receiving, corrections=load_corrections())

    # Curated depth-chart players neither path reaches: no source-season
    # production for the veteran models and not in the rookie class. Without
    # a row they do not merely go unreported - their share of the team is
    # never held open, so the reconcilers hand it to whoever is present.
    # Green Bay's MarShawn Lloyd is the case that surfaced this.
    present_ids = set(vet["player_id"]) | set(rookie_long["player_id"])
    replacement = build_replacement_level_rows(
        conn, feat, depth_chart, present_ids, target_season, hist_seasons)
    if not replacement.empty:
        named = replacement.drop_duplicates("player_id")
        print(f"Added {len(named)} curated depth-chart player(s) at a replacement-level "
              f"prior - no source-season production and not in the rookie class:")
        for _, r in named.iterrows():
            print(f"    {r['player_id']} ({r['position']}, {r['team']}, role={r['role']})")

    combined = pd.concat([vet, rookie_long, replacement], ignore_index=True, sort=False)
    # One composition pipeline, shared with the leakage-safe evaluation
    # harness - see composition.py. Hygiene only (games caps, anchors,
    # stat identities, season totals); no team-volume invent/redistribute.
    combined = compose_board(
        combined,
        shipped_context(conn, target_season, hist_seasons, as_of=as_of),
    )
    _warn_board_level_allocation(conn, combined, depth_chart)
    return combined


def _attach_rookie_intervals(rookie_long, ratios):
    """Attach empirical cell bands, with an honest zero-floor fallback."""
    out = rookie_long.merge(
        ratios, on=["position", "round_bucket", "stat"], how="left")
    no_ratio = out["ratio_low"].isna()
    out.loc[no_ratio, "ratio_low"] = ROOKIE_RATIO_FALLBACK[0]
    out.loc[no_ratio, "ratio_high"] = ROOKIE_RATIO_FALLBACK[1]
    out.loc[no_ratio, "interval_low_n_flag"] = True
    out["interval_low_n_flag"] = out["interval_low_n_flag"].fillna(False)
    out["pred_pg_low"] = np.minimum(
        (out["pred_pg"] * out["ratio_low"]).clip(lower=0), out["pred_pg"])
    out["pred_pg_high"] = np.maximum(
        out["pred_pg"] * out["ratio_high"], out["pred_pg"])
    return out.drop(
        columns=["ratio_low", "ratio_high", "round_bucket", "n"], errors="ignore")


def _attach_rookie_depth_tier(rookie_long, depth_chart):
    """Materialize the tier consumed by team reconciliation for rookies.

    Rookie availability already reads the curated chart, but the final
    reconciler protects a QB starter only through ``depth_tier == 1``. Without
    this bridge a genuine rookie QB1 is treated as bench volume even when the
    curated chart names him the starter.
    """
    out = rookie_long.copy()
    nfl_rank = (
        out["nfl_depth_rank"]
        if "nfl_depth_rank" in out
        else pd.Series(np.nan, index=out.index)
    )
    out[DEPTH_TIER_COLUMN] = depth_tiers(nfl_rank)
    out["depth_tier_source"] = "nflverse"
    return apply_curated_depth_tier(out, depth_chart)


# Board-level tripwires. Same contract as _warn_discounted_high_usage and
# _warn_availability_chart_disagreement: stderr only, never changes a number.
# These watch the finished board rather than any single stage, because the
# failures they look for are products of the reconcilers agreeing with each
# other and being wrong together - which is exactly what no per-stage
# assertion can see.
QB_ATTEMPTS_PER_GAME_CEILING = 42.0
TRIPWIRE_CAP_TOLERANCE = 1e-6
TRIPWIRE_RB_CARRY_SHARE = 0.70
TRIPWIRE_NEWCOMER_MARGIN = 1.0


def _warn_board_level_allocation(conn, combined, depth_chart):
    """Four checks on the finished board, printed for a human to judge."""
    warnings = []
    names = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
    names = names.drop_duplicates("player_id").set_index("player_id")["display_name"]

    def name(player_id):
        return names.get(player_id, player_id)

    # 1. A rate sitting exactly on a support ceiling. The allocator wanted
    # to give more than the evidence allows and was clipped; the number is
    # a bound, not a projection. This is what Josh Jacobs at exactly 25.00
    # carries/game looked like before the rushing fix.
    for stat, ceilings in (("carries", RUSH_ATTEMPTS_PER_APPEARANCE_MAX),
                           ("attempts", {"QB": QB_ATTEMPTS_PER_GAME_CEILING})):
        rows = combined[combined["stat"].eq(stat)]
        for _, r in rows.iterrows():
            ceiling = ceilings.get(r["position"])
            value = pd.to_numeric(pd.Series([r["pred_pg"]]), errors="coerce").iloc[0]
            if ceiling and pd.notna(value) and value >= ceiling - TRIPWIRE_CAP_TOLERANCE:
                warnings.append(
                    f"  CAPPED: {name(r['player_id'])} "
                    f"({r['position']}, {r['team']}) {stat} pinned at the "
                    f"{ceiling:g}/game support ceiling")

    # 2. A curated contributor with no row at all. Should be empty for
    # RB/WR/TE now that they get a replacement-level prior, so anything here
    # is either a QB (deliberately excluded - see REPLACEMENT_POSITIONS) or
    # real drift, e.g. a newly curated player with no historical band.
    if not depth_chart.empty:
        present = set(combined["player_id"].dropna())
        for _, r in depth_chart.iterrows():
            if pd.notna(r.get("gsis_id")) and r["gsis_id"] not in present:
                warnings.append(
                    f"  MISSING: curated {r['position']} {r.get('player_name') or name(r['gsis_id'])} "
                    f"({r['team']}, role={r.get('role')}) has no projection row")

    # 3. One back taking a share of his team's carries that few real
    # backfields concentrate. Legitimate for a genuine bell cow, which is
    # why it warns rather than clips.
    carries = combined[combined["stat"].eq("carries")].copy()
    if not carries.empty:
        carries["season"] = (
            pd.to_numeric(carries["pred_pg"], errors="coerce").clip(lower=0)
            * _row_exposure(carries))
        anchor = carries.drop_duplicates("team").set_index("team")["team_carries_pg_pred"] * SEASON_GAMES
        for _, r in carries[carries["position"].eq("RB")].iterrows():
            total = anchor.get(r["team"], np.nan)
            if pd.notna(total) and total > 0 and r["season"] / total >= TRIPWIRE_RB_CARRY_SHARE:
                warnings.append(
                    f"  RB SHARE: {name(r['player_id'])} ({r['team']}) "
                    f"{r['season']:.0f} of {total:.0f} team carries "
                    f"({r['season']/total:.0%})")

    # 4. A rookie or new arrival projected past the established starter the
    # curated chart puts ahead of him. The Makai Lemon case: real vacancy,
    # real eligibility, wrong conclusion - and invisible to every check that
    # looks at one player at a time.
    if not depth_chart.empty:
        starters = set(depth_chart[depth_chart["role"].eq("starter")]["gsis_id"].dropna())
        for stat in ("targets", "carries"):
            rows = combined[combined["stat"].eq(stat)].copy()
            if rows.empty:
                continue
            rows["season"] = (
                pd.to_numeric(rows["pred_pg"], errors="coerce").clip(lower=0)
                * _row_exposure(rows))
            newcomer = rows["source"].eq("rookie_rule") | rows["team_changed"].fillna(False)
            for (team, position), group in rows.groupby(["team", "position"]):
                incumbent = group[
                    group["player_id"].isin(starters) & ~newcomer.reindex(group.index).fillna(False)]
                if incumbent.empty:
                    continue
                best = incumbent["season"].max()
                for _, r in group[newcomer.reindex(group.index).fillna(False)].iterrows():
                    if r["season"] > best + TRIPWIRE_NEWCOMER_MARGIN:
                        top = incumbent.loc[incumbent["season"].idxmax()]
                        warnings.append(
                            f"  NEWCOMER: {name(r['player_id'])} "
                            f"({position}, {team}) {r['season']:.0f} {stat} exceeds curated "
                            f"starter {name(top['player_id'])} "
                            f"({best:.0f})")

    if warnings:
        print(f"CURATION TRIPWIRE: {len(warnings)} board-level allocation warning(s) - "
              f"informational, no projection was changed:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)


# OUTPUT_DIR / OUTPUT_COLUMNS live in contracts.py.


def with_display_names(conn, out, target_season):
    """Join players.display_name onto the combined projection output - a
    human reading this CSV needs a name, not just a raw gsis_id.

    Data-quality gap found and worked around: nfl_data_py's 2026
    draft_picks.gsis_id column does NOT contain real gsis_ids for this
    draft class (nflverse hasn't back-filled them yet - spot-checked: 0/230
    2026 rows match the `00-0######` gsis_id format, vs 256/256 for 2025)
    - it's some other placeholder id, so these players are structurally
    absent from `players.gsis_id` too. Falls back to draft_picks'
    `pfr_player_name` (drafted rookies) and seasonal_rosters' `player_name`
    (UDFA) for exactly the rows players.display_name can't resolve, rather
    than shipping a CSV with blank names for the entire incoming rookie
    class."""
    players = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
    out = out.merge(players, on="player_id", how="left")

    draft_names = pd.read_sql(
        f"select gsis_id as player_id, pfr_player_name as name from draft_picks where season = {target_season}", conn,
    ).drop_duplicates(subset=["player_id"]).set_index("player_id")["name"]
    roster_names = pd.read_sql(
        f"select player_id, player_name as name from seasonal_rosters where season = {target_season}", conn,
    ).drop_duplicates(subset=["player_id"]).set_index("player_id")["name"]

    missing = out["display_name"].isna()
    out.loc[missing, "display_name"] = out.loc[missing, "player_id"].map(draft_names)
    still_missing = out["display_name"].isna()
    out.loc[still_missing, "display_name"] = out.loc[still_missing, "player_id"].map(roster_names)

    unresolved = out["display_name"].isna().sum()
    if unresolved:
        print(f"WARNING: {unresolved} projection rows have no resolvable display name at all "
              f"(not in players, draft_picks, or seasonal_rosters for {target_season}) - left null, not faked.")
    return out


def _ensure_output_parent(path):
    """Create an output parent even when `path` is a bare filename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def export_projections(conn, target_season, path, as_of=None):
    out = project_season(conn, target_season, as_of=as_of)
    out = with_display_names(conn, out, target_season)
    out = out[OUTPUT_COLUMNS].sort_values(["position", "team", "player_id", "stat"])
    # `--out projections.csv` has an empty dirname; normalizing to an
    # absolute path gives it the current directory as a real parent.
    _ensure_output_parent(path)
    out.to_csv(path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=None, help="CSV output path (default: output/projections_<season>.csv)")
    ap.add_argument(
        "--as-of",
        default=None,
        help="ISO date: filter status overrides and use latest nflverse depth snapshot on/before this date",
    )
    args = ap.parse_args()
    out_path = args.out or os.path.join(OUTPUT_DIR, f"projections_{args.season}.csv")

    conn = get_conn()
    out = export_projections(conn, args.season, out_path, as_of=args.as_of)
    conn.close()

    pd.set_option("display.width", 200)
    print(out.head(30).to_string(index=False))
    print(f"\n{len(out)} projection rows for season {args.season} "
          f"({(out.source=='rookie_rule').sum()} rookie rows flagged low_confidence, "
          f"{out['interval_low_n_flag'].sum()} rows with a low-n interval flag)")
    print(f"Written -> {out_path}")


if __name__ == "__main__":
    main()
