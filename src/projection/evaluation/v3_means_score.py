"""Score v3 mean forecasts against fantasy holdouts."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIER_RANKS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}


def spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    return float(pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"), method="spearman"))


def mae(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(x[mask] - y[mask])))


def score_predictions(frame: pd.DataFrame, pred_col: str, actual_col: str = "actual_points") -> dict:
    valid = frame.dropna(subset=[pred_col, actual_col]).copy()
    out = {
        "n": int(len(valid)),
        "points_mae": mae(valid[actual_col], valid[pred_col]),
        "spearman": spearman(valid[actual_col], valid[pred_col]),
        "by_position": {},
    }
    pos_col = "preseason_position" if "preseason_position" in valid.columns else "position"
    if pos_col not in valid.columns:
        return out
    for pos, sub in valid.groupby(pos_col, observed=True):
        out["by_position"][str(pos)] = {
            "n": int(len(sub)),
            "points_mae": mae(sub[actual_col], sub[pred_col]),
            "spearman": spearman(sub[actual_col], sub[pred_col]),
        }
    return out


def beats_incumbent(candidate: dict, incumbent: dict) -> dict:
    """Require MAE no worse and Spearman no worse (NaN-safe)."""
    c_mae = candidate.get("points_mae")
    i_mae = incumbent.get("points_mae")
    c_sp = candidate.get("spearman")
    i_sp = incumbent.get("spearman")
    mae_ok = (
        c_mae is not None
        and i_mae is not None
        and np.isfinite(c_mae)
        and np.isfinite(i_mae)
        and float(c_mae) <= float(i_mae) + 1e-9
    )
    sp_ok = (
        c_sp is not None
        and i_sp is not None
        and np.isfinite(c_sp)
        and np.isfinite(i_sp)
        and float(c_sp) >= float(i_sp) - 1e-9
    )
    return {"mae_ok": bool(mae_ok), "spearman_ok": bool(sp_ok), "pass": bool(mae_ok and sp_ok)}
