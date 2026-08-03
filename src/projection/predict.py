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
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features, TARGET_STATS
from src.projection.transitions import ALL_FEATURES
from src.projection.rookies import (
    build_rookie_dataset, fit_rookie_baselines, predict_rookies,
    identify_rookie_seasons, identify_udfa_rookie_seasons, identify_target_season_rookie_class,
    team_vacated_opportunity, rookie_interval_ratios,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INTERVAL_RESIDUALS_PATH = os.path.join(MODELS_DIR, "interval_residuals.csv")
# Rookie ratio fallback if a bucket/stat combo has too few historical rows
# for its own empirical ratio (rookie_interval_ratios drops any with <3
# values) - deliberately wide, and always flagged via interval_low_n_flag.
ROOKIE_RATIO_FALLBACK = (0.2, 3.0)


def load_models():
    models = {}
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            path = os.path.join(MODELS_DIR, f"{position}_{stat}.joblib")
            models[(position, stat)] = joblib.load(path)
    return models


def load_interval_residuals():
    if not os.path.exists(INTERVAL_RESIDUALS_PATH):
        raise FileNotFoundError(
            f"{INTERVAL_RESIDUALS_PATH} not found - run `python -m src.projection.backtest` "
            "once (after train.py) to build it before calling project_season."
        )
    return pd.read_csv(INTERVAL_RESIDUALS_PATH)


def project_veterans(conn, feat, source_season, models, resid):
    """source_season's feature rows -> next-season per-game rate
    predictions, for every non-rookie player active in source_season.
    Rookies are excluded here (source_season IS their only season, so they
    have no real trailing features) and projected separately below via the
    rule-based path."""
    rookie_ids = set(identify_rookie_seasons(conn, [source_season])["player_id"]) | \
        set(identify_udfa_rookie_seasons(conn, [source_season])["player_id"])
    rows = []
    for position, stats in TARGET_STATS.items():
        pos_df = feat[(feat["position"] == position) & (feat["season"] == source_season) & (feat["games_played"] > 0)]
        pos_df = pos_df[~pos_df["player_id"].isin(rookie_ids)]
        if pos_df.empty:
            continue
        X = pos_df[ALL_FEATURES]
        for stat in stats:
            m = models[(position, stat)]
            preds = m["model"].predict(X)
            out = pos_df[["player_id", "team", "position"]].copy()
            out["stat"] = stat
            out["pred_pg"] = np.clip(preds, 0, None)  # a per-game rate can't be negative; LightGBM isn't constrained
            out["source"] = "veteran_model"
            out["low_confidence"] = False

            r = resid[(resid["position"] == position) & (resid["stat"] == stat)]
            if r.empty:
                out["pred_pg_low"], out["pred_pg_high"], out["interval_low_n_flag"] = np.nan, np.nan, True
            else:
                r = r.iloc[0]
                out["pred_pg_low"] = (out["pred_pg"] + r["resid_low"]).clip(lower=0)
                out["pred_pg_high"] = out["pred_pg"] + r["resid_high"]
                out["interval_low_n_flag"] = bool(r["low_n_flag"])
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def project_season(conn, target_season):
    """Project `target_season` per-game rates using `target_season - 1`
    features for veterans, and the rookie rule-based path for
    `target_season`'s actual draft-year rookies."""
    source_season = target_season - 1
    models = load_models()
    resid = load_interval_residuals()

    feat = build_player_season_features(conn, seasons=list(range(2016, target_season)))
    vet = project_veterans(conn, feat, source_season, models, resid)
    vet["season"] = target_season

    # rookie path: bucket baselines fit on all completed rookie seasons
    # strictly before target_season (drafted + UDFA, both requiring confirmed
    # active-week production - see rookies.py), applied to target_season's
    # rookie class. target_season's own rookie class is identified separately
    # (identify_target_season_rookie_class) since target_season has no played
    # games yet to confirm anyone's "first active season" against - it reads
    # the class directly off draft_picks + seasonal_rosters instead.
    hist_seasons = list(range(2016, target_season))
    hist_feat = build_player_season_features(conn, seasons=hist_seasons)
    rdf = build_rookie_dataset(conn, hist_feat, seasons=hist_seasons)
    baselines = fit_rookie_baselines(rdf, hist_seasons)
    ratios = rookie_interval_ratios(rdf, baselines, hist_seasons)

    target_class = identify_target_season_rookie_class(conn, target_season)
    vacated = team_vacated_opportunity(conn, [target_season])
    target_class = target_class.merge(vacated, on=["season", "team"], how="left")
    rookie_preds = predict_rookies(target_class, baselines, [target_season])

    pg_cols = [c for c in rookie_preds.columns if c.endswith("_pg")]
    rookie_long = rookie_preds.melt(
        id_vars=["player_id", "team", "position", "season", "rookie_tier", "round_bucket"], value_vars=pg_cols,
        var_name="stat", value_name="pred_pg",
    )
    rookie_long["stat"] = rookie_long["stat"].str.replace("_pg", "", regex=False)
    rookie_long["source"] = "rookie_rule"
    rookie_long["low_confidence"] = True
    rookie_long = rookie_long.dropna(subset=["pred_pg"])
    rookie_long = rookie_long[
        rookie_long.apply(lambda r: r["stat"] in TARGET_STATS.get(r["position"], []), axis=1)
    ]

    rookie_long = rookie_long.merge(ratios, on=["position", "round_bucket", "stat"], how="left")
    no_ratio = rookie_long["ratio_low"].isna()
    rookie_long.loc[no_ratio, "ratio_low"] = ROOKIE_RATIO_FALLBACK[0]
    rookie_long.loc[no_ratio, "ratio_high"] = ROOKIE_RATIO_FALLBACK[1]
    rookie_long.loc[no_ratio, "interval_low_n_flag"] = True
    rookie_long["interval_low_n_flag"] = rookie_long["interval_low_n_flag"].fillna(False)
    rookie_long["pred_pg_low"] = (rookie_long["pred_pg"] * rookie_long["ratio_low"]).clip(lower=0)
    rookie_long["pred_pg_high"] = rookie_long["pred_pg"] * rookie_long["ratio_high"]
    rookie_long = rookie_long.drop(columns=["ratio_low", "ratio_high", "round_bucket", "n"], errors="ignore")

    combined = pd.concat([vet, rookie_long], ignore_index=True, sort=False)
    return combined


OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

OUTPUT_COLUMNS = [
    "player_id", "display_name", "team", "position", "stat",
    "pred_pg", "pred_pg_low", "pred_pg_high",
    "source", "low_confidence", "rookie_tier", "interval_low_n_flag", "season",
]


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


def export_projections(conn, target_season, path):
    out = project_season(conn, target_season)
    out = with_display_names(conn, out, target_season)
    out = out[OUTPUT_COLUMNS].sort_values(["position", "team", "player_id", "stat"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=None, help="CSV output path (default: output/projections_<season>.csv)")
    args = ap.parse_args()
    out_path = args.out or os.path.join(OUTPUT_DIR, f"projections_{args.season}.csv")

    conn = get_conn()
    out = export_projections(conn, args.season, out_path)
    conn.close()

    pd.set_option("display.width", 200)
    print(out.head(30).to_string(index=False))
    print(f"\n{len(out)} projection rows for season {args.season} "
          f"({(out.source=='rookie_rule').sum()} rookie rows flagged low_confidence, "
          f"{out['interval_low_n_flag'].sum()} rows with a low-n interval flag)")
    print(f"Written -> {out_path}")


if __name__ == "__main__":
    main()
