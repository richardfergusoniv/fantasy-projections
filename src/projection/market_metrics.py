"""Draft-edge metrics: model vs contemporaneous ADP/ECR without touching training.

Primary decision surface is preseason draft value versus market rank. Holdout
MAE/Spearman remain secondary accuracy checks (see fantasy_evaluation).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_MAX_MARKET_RANK = 120


def norm_name(name: str | None) -> str:
    if not name:
        return ""
    s = str(name).lower().strip()
    s = s.replace(".", "").replace("'", "").replace("’", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def _mae(a: pd.Series, b: pd.Series) -> float:
    return float(np.mean(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))))


def re_rank(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    """Dense ranks inside the current frame (1 = best when ascending=True for ranks)."""
    return series.rank(method="first", ascending=ascending).astype(int)


def matched_market_frame(
    board: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    market_col: str = "adp",
    model_points_col: str = "model_points",
    actual_points_col: str | None = "actual_points",
    max_market_rank: int | None = DEFAULT_MAX_MARKET_RANK,
) -> pd.DataFrame:
    """Join board to consensus and re-rank both sides inside the matched set.

    Mirrors scripts/consensus_spread.py join rules:
    1. Match on player_id, else normalized name+position.
    2. Optionally truncate to top N by raw market rank.
    3. Re-rank model and market inside the intersection.
    """
    cons = consensus.dropna(subset=[market_col]).copy()
    cons["player_id"] = cons["player_id"].astype(str)
    cons["_norm"] = cons["display_name"].map(norm_name)
    cons["_pos"] = cons["position"].astype(str)

    board = board.copy()
    board["player_id"] = board["player_id"].astype(str)
    if "display_name" not in board.columns and "name" in board.columns:
        board["display_name"] = board["name"]
    if "position" not in board.columns and "preseason_position" in board.columns:
        board["position"] = board["preseason_position"]
    board["_norm"] = board["display_name"].map(norm_name)
    board["_pos"] = board["position"].astype(str)

    by_id = board.set_index("player_id", drop=False)
    by_name = board.set_index(["_norm", "_pos"], drop=False)

    rows: list[dict[str, Any]] = []
    for _, c in cons.iterrows():
        pid = str(c["player_id"])
        if pid in by_id.index:
            p = by_id.loc[pid]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
        else:
            key = (c["_norm"], c["_pos"])
            if key not in by_name.index:
                continue
            p = by_name.loc[key]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
        row: dict[str, Any] = {
            "player_id": str(p["player_id"]),
            "display_name": c["display_name"],
            "position": c["position"],
            "team": c.get("team"),
            "mkt_raw": float(c[market_col]),
            "model_points": float(p[model_points_col])
            if model_points_col in p.index and pd.notna(p.get(model_points_col))
            else 0.0,
        }
        if actual_points_col and actual_points_col in p.index and pd.notna(p[actual_points_col]):
            row["actual_points"] = float(p[actual_points_col])
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("mkt_raw").reset_index(drop=True)
    if max_market_rank is not None:
        out = out.head(int(max_market_rank)).copy()

    # Model rank: higher points = better = lower rank number
    out["our_raw"] = (-out["model_points"]).rank(method="first", ascending=True)
    out = out.sort_values("our_raw").reset_index(drop=True)
    out["our"] = np.arange(1, len(out) + 1)
    out = out.sort_values("mkt_raw").reset_index(drop=True)
    out["mkt"] = np.arange(1, len(out) + 1)
    out["d"] = out["our"] - out["mkt"]  # negative => we rank higher than market
    if "actual_points" in out.columns:
        out["actual_rank"] = (-out["actual_points"]).rank(method="first", ascending=True).astype(int)
    return out


def market_agreement(matched: pd.DataFrame) -> dict[str, Any]:
    if matched.empty:
        return {"n": 0}
    return {
        "n": int(len(matched)),
        "spearman_vs_market": _spearman(matched["our"], matched["mkt"]),
        "mean_abs_rank_delta": float(matched["d"].abs().mean()),
        "median_abs_rank_delta": float(matched["d"].abs().median()),
    }


def draft_edge_proxy(matched: pd.DataFrame) -> dict[str, Any]:
    """Does model−market disagreement predict actual outperformance?

    Residual: model_rank − market_rank (negative = model likes player more).
    After controlling for market rank, correlate residual with actual points /
    actual rank error.
    """
    if matched.empty or "actual_points" not in matched.columns:
        return {"n": 0, "has_actuals": False}

    df = matched.dropna(subset=["actual_points", "d", "mkt"]).copy()
    if len(df) < 10:
        return {"n": int(len(df)), "has_actuals": True, "insufficient": True}

    # Partial association: residual vs actual points after residualizing both
    # on market rank (linear OLS via numpy).
    mkt = df["mkt"].to_numpy(dtype=float)
    resid = df["d"].to_numpy(dtype=float)
    actual = df["actual_points"].to_numpy(dtype=float)
    actual_rank = df["actual_rank"].to_numpy(dtype=float)

    def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        x1 = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
        return y - x1 @ beta

    resid_adj = _residualize(resid, mkt)
    actual_adj = _residualize(actual, mkt)
    # Preferring a player (negative d) should associate with higher actual points
    # => residual_adj and actual_adj should be *negatively* correlated if edge exists.
    edge_corr_points = float(pd.Series(resid_adj).corr(pd.Series(actual_adj)))
    # Preferring a player should associate with better (lower) actual rank
    actual_rank_adj = _residualize(actual_rank, mkt)
    edge_corr_rank = float(pd.Series(resid_adj).corr(pd.Series(actual_rank_adj)))

    # Bucket: when model ranks player higher than market (d < 0), mean actual
    # points surplus vs when model ranks lower.
    higher = df[df["d"] < 0]
    lower = df[df["d"] > 0]
    surplus = float("nan")
    if len(higher) >= 5 and len(lower) >= 5:
        surplus = float(higher["actual_points"].mean() - lower["actual_points"].mean())

    # Top-N hit: among players model ranks above ADP, share finishing top-24 overall
    top_n = 24
    if len(higher) >= 5:
        hit_rate = float((higher["actual_rank"] <= top_n).mean())
    else:
        hit_rate = float("nan")

    return {
        "n": int(len(df)),
        "has_actuals": True,
        "edge_corr_residual_vs_actual_points": edge_corr_points,
        "edge_corr_residual_vs_actual_rank": edge_corr_rank,
        # Negative corr on points residualization = actionable (model-high => more points)
        "actionable_points_edge": bool(edge_corr_points < -0.05),
        "actionable_rank_edge": bool(edge_corr_rank > 0.05),
        "mean_actual_points_when_model_higher": float(higher["actual_points"].mean())
        if len(higher)
        else float("nan"),
        "mean_actual_points_when_model_lower": float(lower["actual_points"].mean())
        if len(lower)
        else float("nan"),
        "points_surplus_model_higher_minus_lower": surplus,
        "top24_hit_rate_when_model_higher": hit_rate,
        "n_model_higher": int(len(higher)),
        "n_model_lower": int(len(lower)),
    }


def accuracy_block(
    frame: pd.DataFrame,
    pred_col: str,
    actual_col: str = "actual_points",
    position_col: str = "position",
) -> dict[str, Any]:
    valid = frame.dropna(subset=[pred_col, actual_col]).copy()
    out: dict[str, Any] = {
        "overall": {
            "n": int(len(valid)),
            "spearman": _spearman(valid[actual_col], valid[pred_col]),
            "points_mae": _mae(valid[actual_col], valid[pred_col]) if len(valid) else float("nan"),
        },
        "by_position": {},
    }
    if position_col not in valid.columns:
        return out
    for pos, sub in valid.groupby(position_col):
        out["by_position"][str(pos)] = {
            "n": int(len(sub)),
            "spearman": _spearman(sub[actual_col], sub[pred_col]),
            "points_mae": _mae(sub[actual_col], sub[pred_col]),
        }
    return out


def fit_nonnegative_blend_weights(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual_points",
    model_cols: tuple[str, ...] = ("v1_pred", "v2_pred"),
    position_col: str = "position",
) -> dict[str, dict[str, float]]:
    """Per-position nonnegative weights summing to 1 that minimize MAE.

    Grid over blend weight for the first model in 0.05 steps; remaining mass
    goes to the second. Extends to two models only (v1/v2).
    """
    if len(model_cols) != 2:
        raise ValueError("fit_nonnegative_blend_weights supports exactly two model columns")
    a_col, b_col = model_cols
    weights: dict[str, dict[str, float]] = {}
    for pos in POSITIONS:
        sub = frame[frame[position_col] == pos].dropna(subset=[actual_col, a_col, b_col])
        if len(sub) < 8:
            weights[pos] = {a_col: 0.5, b_col: 0.5}
            continue
        best_w, best_mae = 0.5, float("inf")
        actual = sub[actual_col].to_numpy(dtype=float)
        a = sub[a_col].to_numpy(dtype=float)
        b = sub[b_col].to_numpy(dtype=float)
        for w100 in range(0, 101, 5):
            w = w100 / 100.0
            pred = w * a + (1.0 - w) * b
            mae = float(np.mean(np.abs(pred - actual)))
            if mae < best_mae:
                best_mae = mae
                best_w = w
        weights[pos] = {a_col: float(best_w), b_col: float(1.0 - best_w)}
    return weights


def apply_blend(
    frame: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    *,
    model_cols: tuple[str, ...] = ("v1_pred", "v2_pred"),
    position_col: str = "position",
    out_col: str = "blend_pred",
) -> pd.DataFrame:
    a_col, b_col = model_cols
    out = frame.copy()
    preds = []
    for _, row in out.iterrows():
        pos = str(row[position_col])
        w = weights.get(pos, {a_col: 0.5, b_col: 0.5})
        wa = float(w.get(a_col, 0.5))
        wb = float(w.get(b_col, 0.5))
        va = float(row[a_col]) if pd.notna(row[a_col]) else 0.0
        vb = float(row[b_col]) if pd.notna(row[b_col]) else 0.0
        preds.append(wa * va + wb * vb)
    out[out_col] = preds
    return out
