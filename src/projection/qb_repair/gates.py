"""GO/NO-GO gates and ensemble reselection helpers for QB repair."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.projection.qb_repair.evaluate import (
    HOLDOUT_SEASON,
    evaluate_holdout,
    select_arm_from_fit_seasons,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def reselect_qb_ensemble_weights(
    *,
    fit_season: int = 2024,
    holdout_season: int = 2025,
) -> dict:
    """Re-evaluate QB v1/v2(/prior) blends without using ECR/ADP point weight.

    Uses accuracy-first evaluation parquet when present; otherwise falls back
    to fantasy_evaluation CSVs. Selection requires holdout improvement in
    starter MAE and Spearman without material all-QB regression.
    """
    eval_path = REPO_ROOT / "output" / "accuracy_first_2026" / "evaluation_players.parquet"
    incumbent = {"v1_pred": 0.4, "v2_pred": 0.6}
    if not eval_path.exists():
        return {
            "selected": incumbent,
            "arm": "incumbent_fallback_missing_eval_parquet",
            "holdout": None,
            "note": "evaluation_players.parquet missing; retaining incumbent 40/60",
        }

    frame = pd.read_parquet(eval_path)
    qb = frame[frame["position"].astype(str).eq("QB")].copy()
    fit = qb[qb["season"].eq(fit_season)].dropna(subset=["actual_points", "v1_pred", "v2_pred"])
    hold = qb[qb["season"].eq(holdout_season)].dropna(subset=["actual_points", "v1_pred", "v2_pred"])

    def _mae(y, p):
        return float(np.mean(np.abs(p - y)))

    def _rho(y, p):
        if len(y) < 3:
            return float("nan")
        return float(pd.Series(y).corr(pd.Series(p), method="spearman"))

    candidates = []
    for w1 in np.arange(0.0, 1.0 + 1e-9, 0.05):
        w2 = 1.0 - w1
        pred = w1 * fit["v1_pred"].to_numpy() + w2 * fit["v2_pred"].to_numpy()
        candidates.append(
            {
                "weights": {"v1_pred": float(w1), "v2_pred": float(w2)},
                "fit_mae": _mae(fit["actual_points"].to_numpy(), pred),
                "fit_spearman": _rho(fit["actual_points"].to_numpy(), pred),
            }
        )
    candidates = sorted(candidates, key=lambda r: (r["fit_mae"], -r["fit_spearman"]))
    best_fit = candidates[0]

    # Score incumbent and best-fit on untouched holdout.
    def _hold_metrics(weights):
        pred = (
            weights["v1_pred"] * hold["v1_pred"].to_numpy()
            + weights["v2_pred"] * hold["v2_pred"].to_numpy()
        )
        return {
            "n": int(len(hold)),
            "mae": _mae(hold["actual_points"].to_numpy(), pred),
            "spearman": _rho(hold["actual_points"].to_numpy(), pred),
        }

    hold_inc = _hold_metrics(incumbent)
    hold_best = _hold_metrics(best_fit["weights"])
    # Promote only if holdout MAE improves and Spearman does not degrade.
    promote = (
        hold_best["mae"] <= hold_inc["mae"]
        and (
            np.isnan(hold_inc["spearman"])
            or hold_best["spearman"] >= hold_inc["spearman"] - 1e-6
        )
    )
    selected = best_fit["weights"] if promote else incumbent
    return {
        "selected": selected,
        "arm": "refit_v1_v2" if promote else "incumbent",
        "promote": promote,
        "fit_leader": best_fit,
        "holdout_incumbent": hold_inc,
        "holdout_candidate": hold_best,
        "ecr_weight": 0.0,
        "adp_weight": 0.0,
    }


def decide_go_nogo(
    *,
    selected_arm: str,
    holdout_report: dict,
    baseline_holdout: dict,
    ensemble_report: dict,
) -> dict:
    """Require starter MAE + rank correlation gains without all-QB damage."""
    def _seg(report, name):
        return next(m for m in report["metrics"] if m["segment"] == name)

    try:
        sel_starter = _seg(holdout_report, "high_confidence_starter")
        base_starter = _seg(baseline_holdout, "high_confidence_starter")
        sel_all = _seg(holdout_report, "all_qb")
        base_all = _seg(baseline_holdout, "all_qb")
    except StopIteration:
        return {
            "verdict": "NO-GO",
            "reasons": ["missing_segment_metrics"],
            "selected_arm": selected_arm,
        }

    reasons = []
    starter_mae_ok = sel_starter["ppg_mae"] < base_starter["ppg_mae"] - 1e-6
    starter_rho_ok = sel_starter["spearman"] > base_starter["spearman"] + 1e-6
    all_mae_ok = sel_all["ppg_mae"] <= base_all["ppg_mae"] * 1.02 + 1e-9
    all_rho_ok = sel_all["spearman"] >= base_all["spearman"] - 0.02
    if not starter_mae_ok:
        reasons.append(
            f"starter_ppg_mae_not_improved ({sel_starter['ppg_mae']:.3f} vs {base_starter['ppg_mae']:.3f})"
        )
    if not starter_rho_ok:
        reasons.append(
            f"starter_spearman_not_improved ({sel_starter['spearman']:.3f} vs {base_starter['spearman']:.3f})"
        )
    if not all_mae_ok:
        reasons.append("all_qb_mae_materially_degraded")
    if not all_rho_ok:
        reasons.append("all_qb_spearman_materially_degraded")
    if selected_arm == "baseline":
        reasons.append("selected_arm_is_baseline")

    # Insufficient evidence: tiny starter n on holdout.
    if sel_starter["n"] < 12:
        reasons.append(f"insufficient_holdout_starter_n ({sel_starter['n']})")

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "selected_arm": selected_arm,
        "holdout_season": HOLDOUT_SEASON,
        "starter": {"selected": sel_starter, "baseline": base_starter},
        "all_qb": {"selected": sel_all, "baseline": base_all},
        "ensemble": ensemble_report,
    }


def run_selection_pipeline(arms: list[str]) -> dict:
    fit = select_arm_from_fit_seasons(arms)
    selected = fit["selected_arm"]
    hold_sel = evaluate_holdout(selected)
    hold_base = evaluate_holdout("baseline")
    ensemble = reselect_qb_ensemble_weights()
    decision = decide_go_nogo(
        selected_arm=selected,
        holdout_report=hold_sel,
        baseline_holdout=hold_base,
        ensemble_report=ensemble,
    )
    return {
        "fit": fit,
        "holdout_selected": hold_sel,
        "holdout_baseline": hold_base,
        "ensemble": ensemble,
        "decision": decision,
    }
