"""Pooled multi-season ridge fit for the OL attribution model (Phase 2 rebuild).

One regression per sub-model across all 2021-2025 plays: lineman indicator
columns (one per gsis_id seen anywhere in the window) + season fixed-effect
dummies + the same situational controls as the original per-season fit.
A player who appears across seasons gets one coefficient instead of five
independent noisy per-season estimates - see PHASE2_REBUILD_REPORT.md.

Alpha: RidgeCV picks the predictive-fit-optimal alpha on the pooled data,
then the final Ridge fit uses ALPHA_STABILITY_MULT times that. Per
PHASE2_STABILITY_INVESTIGATION.md's alpha-sensitivity check, split-half
coefficient stability keeps improving up to ~10x the CV-optimal alpha and
flattens out beyond that (confirmed again on pooled data, see rebuild
report) - RidgeCV alone optimizes held-out prediction, not coefficient
stability, so a fixed higher alpha is used for the final attribution fit.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.ol_model.fit import ALPHAS, PASS_CONTROLS, RUN_CONTROLS  # noqa: F401 (re-exported)

ALPHA_STABILITY_MULT = 10


def _design_matrix(df, controls):
    """Sparse lineman indicator + season one-hot + scaled control columns."""
    all_ol = sorted({pid for ids in df.ol_ids for pid in ids})
    col_idx = {pid: i for i, pid in enumerate(all_ol)}
    rows, cols = [], []
    for r, ids in enumerate(df.ol_ids):
        for pid in ids:
            rows.append(r)
            cols.append(col_idx[pid])
    indicator = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(all_ol)))

    seasons = sorted(df.season.unique())
    season_idx = {s: i for i, s in enumerate(seasons)}
    srows = np.arange(len(df))
    scols = df.season.map(season_idx).to_numpy()
    season_dummies = sparse.csr_matrix((np.ones(len(df)), (srows, scols)), shape=(len(df), len(seasons)))

    control_vals = StandardScaler().fit_transform(df[controls].to_numpy(dtype=float))
    X = sparse.hstack([indicator, season_dummies, sparse.csr_matrix(control_vals)]).tocsr()
    return X, all_ol, seasons


def fit_pooled_submodel(df, outcome_col, controls):
    """Fit the pooled ridge model for one sub-model across all seasons.

    Returns (player_coefs, season_coefs, alpha_used, cv_alpha, n)."""
    X, all_ol, seasons = _design_matrix(df, controls)
    y = df[outcome_col].to_numpy(dtype=float)

    cv_model = RidgeCV(alphas=ALPHAS, cv=5)
    cv_model.fit(X, y)
    alpha_used = cv_model.alpha_ * ALPHA_STABILITY_MULT

    model = Ridge(alpha=alpha_used)
    model.fit(X, y)

    n_ol, n_season = len(all_ol), len(seasons)
    player_coefs = pd.DataFrame({"gsis_id": all_ol, "coef": model.coef_[:n_ol]})
    season_coefs = pd.DataFrame({"season": seasons, "coef": model.coef_[n_ol:n_ol + n_season]})
    return player_coefs, season_coefs, alpha_used, cv_model.alpha_, len(df)


def split_half_stability(df, outcome_col, controls, alpha, n_splits=5, seed=100):
    """Split-half coefficient correlation on the pooled dataset, split by
    game within each season (so no game's context leaks across halves)."""
    corrs = []
    for i in range(n_splits):
        rng = np.random.default_rng(seed + i)
        # split games independently within each season, then pool halves
        half_a_masks, half_b_masks = [], []
        for season in sorted(df.season.unique()):
            games = df.loc[df.season == season, "nflverse_game_id"].unique()
            shuffled = rng.permutation(games)
            half = len(shuffled) // 2
            half_a_masks.append(df.season.eq(season) & df.nflverse_game_id.isin(shuffled[:half]))
            half_b_masks.append(df.season.eq(season) & df.nflverse_game_id.isin(shuffled[half:]))
        mask_a = np.logical_or.reduce(half_a_masks)
        mask_b = np.logical_or.reduce(half_b_masks)
        df_a, df_b = df[mask_a], df[mask_b]

        Xa, ol_a, _ = _design_matrix(df_a, controls)
        Xb, ol_b, _ = _design_matrix(df_b, controls)
        ma = Ridge(alpha=alpha).fit(Xa, df_a[outcome_col].to_numpy(dtype=float))
        mb = Ridge(alpha=alpha).fit(Xb, df_b[outcome_col].to_numpy(dtype=float))
        ca = pd.Series(ma.coef_[:len(ol_a)], index=ol_a)
        cb = pd.Series(mb.coef_[:len(ol_b)], index=ol_b)
        both = pd.DataFrame({"a": ca, "b": cb}).dropna()
        corrs.append(both["a"].corr(both["b"]))
    return corrs
