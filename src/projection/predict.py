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
from src.projection.rookies import build_rookie_dataset, fit_rookie_baselines, predict_rookies, identify_rookie_seasons

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")


def load_models():
    models = {}
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            path = os.path.join(MODELS_DIR, f"{position}_{stat}.joblib")
            models[(position, stat)] = joblib.load(path)
    return models


def project_veterans(conn, feat, source_season, models):
    """source_season's feature rows -> next-season per-game rate
    predictions, for every non-rookie player active in source_season.
    Rookies are excluded here (source_season IS their only season, so they
    have no real trailing features) and projected separately below via the
    rule-based path."""
    rookie_ids = set(identify_rookie_seasons(conn, [source_season])["player_id"])
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
            out["pred_pg"] = preds
            out["source"] = "veteran_model"
            out["low_confidence"] = False
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def project_season(conn, target_season):
    """Project `target_season` per-game rates using `target_season - 1`
    features for veterans, and the rookie rule-based path for
    `target_season`'s actual draft-year rookies."""
    source_season = target_season - 1
    models = load_models()

    feat = build_player_season_features(conn, seasons=list(range(2016, target_season)))
    vet = project_veterans(conn, feat, source_season, models)
    vet["season"] = target_season

    # rookie path: bucket baselines fit on all completed rookie seasons
    # strictly before target_season, applied to target_season's draft class.
    full_feat = build_player_season_features(conn, seasons=list(range(2016, target_season + 1)))
    rdf = build_rookie_dataset(conn, full_feat, seasons=list(range(2016, target_season + 1)))
    train_seasons = [s for s in rdf["season"].unique() if s < target_season]
    baselines = fit_rookie_baselines(rdf, train_seasons)
    rookie_preds = predict_rookies(rdf, baselines, [target_season])

    pg_cols = [c for c in rookie_preds.columns if c.endswith("_pg")]
    rookie_long = rookie_preds.melt(
        id_vars=["player_id", "team", "position", "season"], value_vars=pg_cols,
        var_name="stat", value_name="pred_pg",
    )
    rookie_long["stat"] = rookie_long["stat"].str.replace("_pg", "", regex=False)
    rookie_long["source"] = "rookie_rule"
    rookie_long["low_confidence"] = True
    rookie_long = rookie_long.dropna(subset=["pred_pg"])
    rookie_long = rookie_long[
        rookie_long.apply(lambda r: r["stat"] in TARGET_STATS.get(r["position"], []), axis=1)
    ]

    combined = pd.concat([vet, rookie_long], ignore_index=True, sort=False)
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    conn = get_conn()
    out = project_season(conn, args.season)
    conn.close()

    pd.set_option("display.width", 160)
    print(out.head(30).to_string(index=False))
    print(f"\n{len(out)} projection rows for season {args.season} "
          f"({(out.source=='rookie_rule').sum()} rookie rows flagged low_confidence)")


if __name__ == "__main__":
    main()
