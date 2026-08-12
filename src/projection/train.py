"""Fit and save the production LightGBM per-position/per-stat regressors.

Scope decision (per spec, stated plainly): training is restricted to
2021-2025 transitions ONLY, across every position/stat model, not just the
OL-conditioned ones. 2016-2020 has team-scheme features (oc_tendency_profiles)
but no OL quality at all, and mixing eras would mean either (a) two
different feature sets for early vs. late seasons, which breaks a single
LightGBM model's fixed input schema, or (b) leaving OL columns NaN for
~40% of rows, which understates how much signal the model could get from
those columns for the years that DO have them. Given the project's own
architecture explicitly wants efficiency conditioned on OL quality, (a)
restricting to 2021-2025 was chosen over "train opportunity-only models
back to 2016 and a separate OL-conditioned model 2021+" - that alternative
was considered but rejected as unnecessary complexity for what would be a
second full model per stat, when the 2021-2025 window already gives a
workable (if small) sample - see PHASE4_REPORT.md for the honest sample-size
caveat this creates.

These are the PRODUCTION models (trained on all 4 available transitions,
2021->22, 22->23, 23->24, 24->25) for Phase 5 to consume via predict.py.
backtest.py fits separate, held-out versions (train on the first 3
transitions only) purely to score against the 2025 holdout - it does not
save models for production use.
"""
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV

from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features, TARGET_STATS
from src.projection.transitions import (
    build_transition_pairs, build_team_transition_pairs, build_availability_pairs,
    ALL_FEATURES, TEAM_FEATURES, REFRAMED_SHARE_STATS, RECEIVING_SHARE_LABEL,
    TEAM_TOTAL_LABEL, AVAILABILITY_LABEL,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
ALL_PAIRS = [(s, s + 1) for s in TRAIN_SEASONS[:-1]]  # (2021,22),(22,23),(23,24),(24,25)

# Small, fixed hyperparameters, not tuned - the training set per position/
# stat is a few hundred rows at most (see PHASE4_REPORT.md), so a shallow,
# heavily-regularized tree ensemble is used to avoid overfitting rather
# than for any measured performance reason. Documented as un-tuned.
LGBM_PARAMS = dict(
    n_estimators=100, learning_rate=0.05, max_depth=3, num_leaves=8,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, verbosity=-1, random_state=0,
)


def fit_one(feat, position, stat, pairs=ALL_PAIRS):
    """(position, stat) in REFRAMED_SHARE_STATS (joint/multi-output Phase A
    - see transitions.py) trains on RECEIVING_SHARE_LABEL instead of the
    default `{stat}_pg` rate - everything else about fitting (same
    LGBM_PARAMS, same ALL_FEATURES input) is unchanged, since this is a
    label swap, not a capacity change."""
    label_col = RECEIVING_SHARE_LABEL if (position, stat) in REFRAMED_SHARE_STATS else None
    data = build_transition_pairs(feat, position, stat, pairs, label_col=label_col)
    y_col = label_col or f"{stat}_pg"
    X = data[ALL_FEATURES]
    y = data[y_col]
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)
    return model, len(data)


def fit_team_total(feat, pairs=ALL_PAIRS):
    """Team-season-grain model for TEAM_TOTAL_LABEL - deliberately a
    RidgeCV linear model, not LightGBM: build_team_transition_pairs
    produces only ~96-128 rows (32 teams x 3-4 transitions), smaller than
    any per-player dataset, and team passing volume is a much smoother,
    more autocorrelated target (this year's team volume is close to last
    year's) than an individual player's rate - a shrinkage linear model is
    the "add structure, not capacity" choice here, consistent with this
    project's own controlled finding (see PHASE4_REPORT.md's hyperparameter
    experiment) that MORE model capacity measurably hurts at this sample
    size rather than helping. `alphas` is a log-spaced grid, not a single
    hand-picked value - RidgeCV selects the regularization strength via
    efficient leave-one-out CV rather than this being another stated-but-
    untuned constant."""
    data = build_team_transition_pairs(feat, pairs)
    X = data[TEAM_FEATURES]
    y = data[TEAM_TOTAL_LABEL]
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(X, y)
    return model, len(data)


def fit_availability(feat, position, pairs=ALL_PAIRS):
    """Per-position season N -> season N+1 games-played model (Phase 11).

    Trained on build_availability_pairs, which keeps the
    games_played_to = 0 rows every other model in this module drops - the
    whole point is to predict the availability those rows represent.

    Measured against carrying season-N games forward, leave-one-transition
    -out over 2017-2025 (MAE, and the per-season consistency ratio this
    project uses as its ship/no-ship gate elsewhere):
      WR 4.160 vs 4.431  (+6.1%, consistency 4.11, 7/8 seasons positive)
      RB 4.592 vs 4.861  (+5.3%, consistency 5.03, 8/8)
      TE 3.902 vs 4.106  (+4.6%, consistency 1.42, 5/8) - marginal
      QB 4.256 vs 4.287  (-1.5%, consistency -0.25, 4/8) - NO better than
         carry-forward; shipped for uniformity because it is not
         materially WORSE either, but a QB's projected_games should be
         read as barely more than "what he played last year."
    Tail separation is the useful part and is real for WR/RB: the 30
    lowest predictions average 5.0-5.6 actual games against 13.8 for the
    30 highest."""
    data = build_availability_pairs(feat, position, pairs)
    if data.empty:
        return None, 0
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(data[ALL_FEATURES], data[AVAILABILITY_LABEL])
    return model, len(data)


def main():
    conn = get_conn()
    feat = build_player_season_features(conn)
    conn.close()

    os.makedirs(MODELS_DIR, exist_ok=True)
    manifest = []

    team_model, team_n = fit_team_total(feat)
    team_path = os.path.join(MODELS_DIR, "team_passing_yards.joblib")
    joblib.dump(
        {"model": team_model, "features": TEAM_FEATURES, "label": TEAM_TOTAL_LABEL},
        team_path,
    )
    manifest.append(("TEAM", "passing_yards", team_n))
    print(f"TEAM passing_yards: trained on {team_n} rows -> {team_path}")

    for position, stats in TARGET_STATS.items():
        for stat in stats:
            model, n = fit_one(feat, position, stat)
            path = os.path.join(MODELS_DIR, f"{position}_{stat}.joblib")
            reframed = (position, stat) in REFRAMED_SHARE_STATS
            label = RECEIVING_SHARE_LABEL if reframed else f"{stat}_pg"
            joblib.dump(
                {"model": model, "features": ALL_FEATURES, "position": position, "stat": stat, "label": label},
                path,
            )
            manifest.append((position, stat, n))
            note = " (reframed: predicts receiving_yards_share, composed with team_passing_yards at predict time)" if reframed else ""
            print(f"{position} {stat}: trained on {n} rows -> {path}{note}")

    # Post-hoc correction parameters (Phase 7 of the consensus-gap work).
    # Imported HERE rather than at module scope because corrections.py
    # imports LGBM_PARAMS/fit_team_total from this module - a local import
    # is the simplest honest way to break that cycle without duplicating
    # the fitting code into both files.
    #
    # Fit on leave-one-transition-out residuals across ALL_PAIRS: these
    # are the PRODUCTION parameters predict.py consumes. backtest.py
    # deliberately refits its own version over its training pairs only, so
    # nothing fit with knowledge of the 2025 holdout ever scores against
    # it.
    from src.projection.corrections import compute_loo_receiving_residuals, fit_elite_shrinkage

    loo_resid = compute_loo_receiving_residuals(feat, ALL_PAIRS)
    corr_params = fit_elite_shrinkage(loo_resid)
    corr_path = os.path.join(MODELS_DIR, "corrections.joblib")
    joblib.dump(corr_params, corr_path)
    if corr_params:
        for position, p in sorted(corr_params.items()):
            print(f"{position} elite-shrinkage correction: beta={p['beta']:.4f} "
                  f"knot={p['knot']:.0f} ypg cap=+{p['cap']:.0f} (fit on {p['n_above']} rows above knot, "
                  f"season-consistency {p['season_consistency']:.1f}) -> {corr_path}")
    else:
        print(f"No elite-shrinkage correction fit (no position cleared the evidence gate) -> {corr_path}")

    for position in TARGET_STATS:
        model, n = fit_availability(feat, position)
        if model is None:
            print(f"{position} games: no availability rows, skipped")
            continue
        path = os.path.join(MODELS_DIR, f"{position}_games.joblib")
        joblib.dump(
            {"model": model, "features": ALL_FEATURES, "position": position,
             "stat": "games", "label": AVAILABILITY_LABEL},
            path,
        )
        manifest.append((position, "games", n))
        print(f"{position} games (availability): trained on {n} rows "
              f"(includes never-played-again rows) -> {path}")

    joblib.dump(manifest, os.path.join(MODELS_DIR, "manifest.joblib"))
    print("Done.")


if __name__ == "__main__":
    main()
