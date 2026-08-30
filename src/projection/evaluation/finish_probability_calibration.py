"""Finish-probability calibration against a leakage-safe rank-to-finish baseline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.draft_assistant.draft_value_simulation import FINISH_CUTOFFS, compute_finish_probabilities
from src.projection.evaluation.accuracy_first import TOP_ADP

DEFAULT_EVAL_CUTOFFS = (12, 24, 36)
RELIABILITY_BINS = 10


def attach_positional_ranks(
    frame: pd.DataFrame,
    *,
    actual_points_col: str = "actual_points",
    projected_points_col: str = "selected_pred",
) -> pd.DataFrame:
    """Add within-position preseason projected rank and realized finish rank."""
    out = frame.copy()
    actual = pd.to_numeric(out[actual_points_col], errors="coerce")
    projected = pd.to_numeric(out[projected_points_col], errors="coerce")
    out["actual_rank"] = (
        actual.groupby(out["position"], observed=True)
        .rank(ascending=False, method="first")
        .astype(float)
    )
    out["pred_rank"] = (
        projected.groupby(out["position"], observed=True)
        .rank(ascending=False, method="first")
        .astype(float)
    )
    return out


def load_top120_evaluation_frame(
    *,
    season: int,
    eval_players_path: str | None = None,
    projected_col: str | None = None,
) -> pd.DataFrame:
    """Top-ADP rows for one season from the accuracy-first evaluation frame."""
    from pathlib import Path

    from src.projection.contracts import OUTPUT_DIR

    path = (
        Path(eval_players_path)
        if eval_players_path
        else Path(OUTPUT_DIR) / "accuracy_first_2026" / "evaluation_players.parquet"
    )
    frame = pd.read_parquet(path)
    frame = frame[frame["season"].eq(int(season))].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    adp = pd.to_numeric(frame["adp"], errors="coerce")
    frame = frame.loc[adp.notna() & adp.le(TOP_ADP)].copy()
    if projected_col is None:
        projected_col = (
            "selected_pred"
            if frame["selected_pred"].notna().any()
            else "incumbent_pred"
        )
    frame["projected_points"] = pd.to_numeric(frame[projected_col], errors="coerce")
    return attach_positional_ranks(
        frame,
        projected_points_col="projected_points",
    )


def fit_rank_to_finish_rates(
    training: pd.DataFrame,
    cutoff: int,
    *,
    rank_col: str = "pred_rank",
    actual_rank_col: str = "actual_rank",
) -> pd.DataFrame:
    """Empirical finish rates by position and preseason projected rank."""
    rows: list[dict] = []
    for (position, pred_rank), grp in training.groupby(
        ["position", rank_col], observed=True
    ):
        actual = pd.to_numeric(grp[actual_rank_col], errors="coerce")
        if actual.notna().sum() == 0:
            continue
        rows.append({
            "position": str(position),
            "pred_rank": float(pred_rank),
            "n": int(actual.notna().sum()),
            "finish_rate": float((actual <= cutoff).mean()),
        })
    return pd.DataFrame(rows)


def apply_rank_to_finish_baseline(
    frame: pd.DataFrame,
    rates: pd.DataFrame,
    cutoff: int,
    *,
    rank_col: str = "pred_rank",
    out_col: str | None = None,
) -> pd.Series:
    """Map holdout preseason rank to the training-season finish curve."""
    out_col = out_col or f"baseline_p_finish_top{cutoff}"
    lookup = rates.set_index(["position", "pred_rank"])["finish_rate"]
    keys = list(zip(frame["position"].astype(str), pd.to_numeric(frame[rank_col], errors="coerce")))
    mapped = pd.Series(
        [lookup.get(key, np.nan) for key in keys],
        index=frame.index,
        dtype=float,
    )
    # Unseen ranks fall back to the position-wide training rate.
    fallback = rates.groupby("position", observed=True)["finish_rate"].mean()
    mapped = mapped.fillna(frame["position"].astype(str).map(fallback))
    return mapped.fillna(0.0).rename(out_col)


def _calibration_regression(prob: np.ndarray, outcome: np.ndarray) -> dict:
    if len(prob) < 3:
        return {"intercept": float("nan"), "slope": float("nan")}
    design = np.column_stack([np.ones(len(prob)), prob])
    coeff, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return {"intercept": float(coeff[0]), "slope": float(coeff[1])}


def _reliability_curve(
    prob: np.ndarray,
    outcome: np.ndarray,
    *,
    n_bins: int = RELIABILITY_BINS,
) -> list[dict]:
    frame = pd.DataFrame({"prob": prob, "outcome": outcome}).dropna()
    if frame.empty:
        return []
    try:
        frame["bin"] = pd.qcut(frame["prob"], n_bins, duplicates="drop")
    except ValueError:
        frame["bin"] = 0
    grouped = frame.groupby("bin", observed=True).agg(
        mean_prob=("prob", "mean"),
        mean_outcome=("outcome", "mean"),
        n=("outcome", "size"),
    )
    return grouped.reset_index(drop=True).to_dict(orient="records")


def _brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((prob - outcome) ** 2))


def evaluate_finish_probability_slice(
    frame: pd.DataFrame,
    *,
    cutoff: int,
    prob_col: str,
    baseline_col: str,
    actual_rank_col: str = "actual_rank",
) -> dict:
    """Score one cutoff for overall and by-position slices."""
    prob = pd.to_numeric(frame[prob_col], errors="coerce")
    baseline = pd.to_numeric(frame[baseline_col], errors="coerce")
    actual = (pd.to_numeric(frame[actual_rank_col], errors="coerce") <= cutoff).astype(float)
    mask = prob.notna() & baseline.notna() & actual.notna()
    if not mask.any():
        return {
            "cutoff": cutoff,
            "n": 0,
            "passes": False,
            "reason": "no_scored_rows",
        }
    p = prob[mask].to_numpy(dtype=float)
    b = baseline[mask].to_numpy(dtype=float)
    y = actual[mask].to_numpy(dtype=float)
    calib = _calibration_regression(p, y)
    result = {
        "cutoff": cutoff,
        "n": int(mask.sum()),
        "brier": _brier(p, y),
        "baseline_brier": _brier(b, y),
        "brier_improvement": _brier(b, y) - _brier(p, y),
        "mean_predicted": float(np.mean(p)),
        "mean_observed": float(np.mean(y)),
        "calibration_intercept": calib["intercept"],
        "calibration_slope": calib["slope"],
        "reliability_curve": _reliability_curve(p, y),
        "passes": _brier(p, y) <= _brier(b, y),
    }
    by_position: dict[str, dict] = {}
    sub = frame.loc[mask].copy()
    for position, grp in sub.groupby("position", observed=True):
        gp = pd.to_numeric(grp[prob_col], errors="coerce").to_numpy(dtype=float)
        gb = pd.to_numeric(grp[baseline_col], errors="coerce").to_numpy(dtype=float)
        gy = (pd.to_numeric(grp[actual_rank_col], errors="coerce") <= cutoff).astype(float).to_numpy()
        pos_calib = _calibration_regression(gp, gy)
        by_position[str(position)] = {
            "n": int(len(grp)),
            "brier": _brier(gp, gy),
            "baseline_brier": _brier(gb, gy),
            "brier_improvement": _brier(gb, gy) - _brier(gp, gy),
            "mean_predicted": float(np.mean(gp)),
            "mean_observed": float(np.mean(gy)),
            "calibration_intercept": pos_calib["intercept"],
            "calibration_slope": pos_calib["slope"],
            "reliability_curve": _reliability_curve(gp, gy),
            "passes": _brier(gp, gy) <= _brier(gb, gy),
        }
    result["by_position"] = by_position
    return result


def build_finish_probability_frame(
    recentered_draws: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    training_seasons: Iterable[int] = (2024,),
    cutoffs: Iterable[int] = FINISH_CUTOFFS,
    eval_cutoffs: Iterable[int] = DEFAULT_EVAL_CUTOFFS,
) -> tuple[pd.DataFrame, dict]:
    """Attach candidate finish probabilities and leakage-safe baseline to holdout."""
    holdout = holdout.copy()
    holdout["player_id"] = holdout["player_id"].astype(str)
    projected_col = (
        "selected_pred"
        if holdout["selected_pred"].notna().any()
        else "incumbent_pred"
    )
    holdout["projected_points"] = pd.to_numeric(holdout[projected_col], errors="coerce")
    ranked = attach_positional_ranks(holdout, projected_points_col="projected_points")

    finish_probs = compute_finish_probabilities(recentered_draws, cutoffs=cutoffs)
    ranked = ranked.merge(finish_probs, on="player_id", how="left")

    training_frames = [
        load_top120_evaluation_frame(season=season) for season in training_seasons
    ]
    training = pd.concat(training_frames, ignore_index=True)
    baseline_tables: dict[int, pd.DataFrame] = {}
    for cutoff in set(cutoffs) | set(eval_cutoffs):
        rates = fit_rank_to_finish_rates(training, cutoff)
        baseline_tables[cutoff] = rates
        ranked[f"baseline_p_finish_top{cutoff}"] = apply_rank_to_finish_baseline(
            ranked,
            rates,
            cutoff,
        )

    meta = {
        "training_seasons": [int(s) for s in training_seasons],
        "training_n": int(len(training)),
        "holdout_n": int(len(ranked)),
        "projected_points_col": projected_col,
        "baseline_tables": {
            str(cutoff): table.to_dict(orient="records")
            for cutoff, table in baseline_tables.items()
            if cutoff in eval_cutoffs or cutoff in cutoffs
        },
    }
    return ranked, meta


def evaluate_finish_probability_report(
    scored: pd.DataFrame,
    *,
    cutoffs: Iterable[int] = DEFAULT_EVAL_CUTOFFS,
) -> dict:
    """Full finish-probability calibration report for the holdout population."""
    checks = []
    for cutoff in cutoffs:
        prob_col = f"p_finish_top{cutoff}"
        baseline_col = f"baseline_p_finish_top{cutoff}"
        if prob_col not in scored.columns or baseline_col not in scored.columns:
            checks.append({
                "cutoff": cutoff,
                "n": 0,
                "passes": False,
                "reason": "missing_columns",
            })
            continue
        checks.append(
            evaluate_finish_probability_slice(
                scored,
                cutoff=cutoff,
                prob_col=prob_col,
                baseline_col=baseline_col,
            )
        )
    passes = all(check.get("passes", False) for check in checks if check.get("n", 0) > 0)
    return {
        "passes": passes,
        "cutoffs": list(cutoffs),
        "checks": checks,
        "wr_summary": next(
            (
                check.get("by_position", {}).get("WR")
                for check in checks
                if check.get("cutoff") == 12
            ),
            None,
        ),
    }
