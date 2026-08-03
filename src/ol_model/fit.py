"""Fit per-season ridge regressions for the OL attribution model."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 25)

PASS_CONTROLS = ["down", "ydstogo", "score_differential", "game_seconds_remaining", "opp_pass_rush_quality"]
RUN_CONTROLS = ["down", "ydstogo", "score_differential", "defenders_in_box"]


def _design_matrix(df, controls):
    """Sparse one-hot lineman indicator matrix + scaled control columns."""
    all_ol = sorted({pid for ids in df.ol_ids for pid in ids})
    col_idx = {pid: i for i, pid in enumerate(all_ol)}

    rows, cols = [], []
    for r, ids in enumerate(df.ol_ids):
        for pid in ids:
            rows.append(r)
            cols.append(col_idx[pid])
    indicator = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(all_ol)))

    control_vals = StandardScaler().fit_transform(df[controls].to_numpy(dtype=float))
    X = sparse.hstack([indicator, sparse.csr_matrix(control_vals)]).tocsr()
    return X, all_ol


def fit_submodel(df, outcome_col, controls):
    """Fit RidgeCV for one season/sub-model. Returns DataFrame of
    gsis_id -> coefficient, plus alpha and n."""
    X, all_ol = _design_matrix(df, controls)
    y = df[outcome_col].to_numpy(dtype=float)

    model = RidgeCV(alphas=ALPHAS, cv=5)
    model.fit(X, y)

    n_ol = len(all_ol)
    coefs = pd.DataFrame({"gsis_id": all_ol, "coef": model.coef_[:n_ol]})
    return coefs, model.alpha_, len(df)
