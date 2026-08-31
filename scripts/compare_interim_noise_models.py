"""Compare noise models for the interim simulation, held out.

The shipped interim path draws ``clip(pred + residual, 0)``. Clipping is
asymmetric -- E[max(0, X)] > E[X] -- so an unbiased additive residual becomes
a biased draw, measured at roughly +8 fantasy points per player with ~29% of
draws clipped. That is why the interim arm loses to v1 on every backtest fold
despite being built from v1's own point estimates.

This scores candidate replacements the same way the calibration gate now
works: for each (position, stat) and each test season after the first, the
noise model is calibrated on STRICTLY EARLIER seasons and scored on the
untouched one. Nothing here changes the shipped path.

Candidates
----------
additive_clipped
    Current behaviour. Additive residual, floored at zero.
additive_unclipped
    Reference only -- shows exactly what the floor costs. Can go negative,
    so it is not shippable for a counting stat.
additive_rescaled
    Floor kept, then scaled so the draw mean returns to the point estimate.
    Non-negative and mean-unbiased; distorts the low tail's shape.
multiplicative
    Bootstrap actual/pred ratios instead of differences. Non-negative by
    construction, never needs a floor. Matches what the rookie interval path
    already does.

Usage:
    python scripts/compare_interim_noise_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.projection.contracts import BACKTEST_DIR
from src.projection.fantasy_points import SCORING

QUANTILES = (0.10, 0.90)
TARGET_COVERAGE = QUANTILES[1] - QUANTILES[0]
# Ratios need a denominator. Below this the ratio explodes, so those rows fall
# back to the additive pool rather than inventing a multiplier.
RATIO_PRED_FLOOR = 0.05
OUT_PATH = ROOT / "output" / "backtest" / "interim_noise_model_comparison.json"


def _draws_for_row(pred: float, pool: np.ndarray, ratios: np.ndarray, model: str) -> np.ndarray:
    """Predictive draws for one row under one noise model."""
    if model == "additive_unclipped":
        return pred + pool
    if model == "additive_clipped":
        return np.clip(pred + pool, 0.0, None)
    if model == "additive_rescaled":
        drawn = np.clip(pred + pool, 0.0, None)
        mean = drawn.mean()
        # Scale back onto the point estimate; preserves the zero floor.
        return drawn * (pred / mean) if mean > 0 else drawn
    if model == "multiplicative":
        if ratios.size == 0 or pred < RATIO_PRED_FLOOR:
            return np.clip(pred + pool, 0.0, None)
        return pred * ratios
    raise ValueError(model)


MODELS = (
    "additive_clipped",
    "additive_unclipped",
    "additive_rescaled",
    "multiplicative",
)


def run_fantasy_level(residuals: pd.DataFrame, *, n_draws: int = 400, seed: int = 17) -> dict:
    """Score each noise model on SUMMED fantasy points, the shipped grain.

    Scoring per-stat rates is the wrong level and inverts the answer. Residual
    right-skew pushes each stat's median BELOW its point estimate, while the
    zero floor pushes each stat's mean ABOVE it. Summing eight stats largely
    removes the skew, so the total's median migrates off the first bias and
    onto the second: measured on the 2026 board, per-stat medians sum to
    -3.93 against v1 while the median of the sum lands at +4.26.

    The shipped artifact is a fantasy-points percentile, so that is the grain
    the comparison has to run at. Games are held fixed across models -- the
    binomial games draw is shared by all of them and would only add noise to
    a comparison of the noise models themselves.
    """
    rng = np.random.default_rng(seed)
    out_rows = []
    scoring_stats = set(SCORING)
    for season in sorted(residuals["test_season"].unique())[1:]:
        cal = residuals[residuals["test_season"] < season]
        test = residuals[residuals["test_season"] == season]
        if cal.empty or test.empty:
            continue
        pools, ratio_pools = {}, {}
        for key, grp in cal.groupby(["position", "stat"], observed=True):
            pools[key] = grp["resid"].to_numpy(dtype=float)
            p = grp["pred"].to_numpy(dtype=float)
            a = grp["actual"].to_numpy(dtype=float)
            usable = p >= RATIO_PRED_FLOOR
            ratio_pools[key] = a[usable] / p[usable] if usable.any() else np.array([])

        # Donor pool for the joint bootstrap: each donor is one player-season's
        # WHOLE residual vector, so the correlation between a player's stats is
        # carried over instead of being destroyed by independent draws.
        donors: dict[str, list[dict]] = {}
        for (_, position), grp in cal.groupby(["player_id", "position"], observed=True):
            donors.setdefault(position, []).append(
                dict(zip(grp["stat"], grp["resid"].astype(float))))

        scored = test[test["stat"].isin(scoring_stats)]
        # joint_bootstrap needs the player's whole row set at once, so it is
        # scored separately from the per-row marginal models.
        jp10, jp50, jp90, jact, jpred = [], [], [], [], []
        for player_id, prow in scored.groupby("player_id", observed=True):
            position = prow["position"].iloc[0]
            pool = donors.get(position, [])
            totals = np.zeros(n_draws)
            a_tot = p_tot = 0.0
            rows_list = list(prow.itertuples(index=False))
            if pool:
                picks = rng.integers(0, len(pool), size=n_draws)
                for r in rows_list:
                    weight = SCORING[r.stat]
                    donor_resid = np.array(
                        [pool[i].get(r.stat, 0.0) for i in picks], dtype=float)
                    totals += np.clip(float(r.pred) + donor_resid, 0.0, None) * weight
            for r in rows_list:
                a_tot += float(r.actual) * SCORING[r.stat]
                p_tot += float(r.pred) * SCORING[r.stat]
            lo, med, hi = np.quantile(totals, [QUANTILES[0], 0.5, QUANTILES[1]])
            jp10.append(lo); jp50.append(med); jp90.append(hi)
            jact.append(a_tot); jpred.append(p_tot)
        jp10 = np.asarray(jp10); jp50 = np.asarray(jp50); jp90 = np.asarray(jp90)
        jact = np.asarray(jact); jpred = np.asarray(jpred)
        out_rows.append({
            "test_season": int(season), "model": "joint_bootstrap", "n": int(len(jp50)),
            "p50_mae": float(np.mean(np.abs(jp50 - jact))),
            "pred_mae": float(np.mean(np.abs(jpred - jact))),
            "p50_minus_pred": float(np.mean(jp50 - jpred)),
            "p50_bias": float(np.mean(jp50 - jact)),
            "coverage": float(np.mean((jp10 <= jact) & (jact <= jp90))),
            "pred_inside_band": float(np.mean((jp10 <= jpred) & (jpred <= jp90))),
            "mean_width": float(np.mean(jp90 - jp10)),
        })

        for model in MODELS:
            sim_p10, sim_p50, sim_p90, actual_pts, pred_pts = [], [], [], [], []
            for player_id, prow in scored.groupby("player_id", observed=True):
                totals = np.zeros(n_draws)
                a_tot = p_tot = 0.0
                for r in prow.itertuples(index=False):
                    weight = SCORING[r.stat]
                    pool = pools.get((r.position, r.stat), np.array([0.0]))
                    ratios = ratio_pools.get((r.position, r.stat), np.array([]))
                    draws = _draws_for_row(float(r.pred), pool, ratios, model)
                    totals += rng.choice(draws, size=n_draws) * weight
                    a_tot += float(r.actual) * weight
                    p_tot += float(r.pred) * weight
                lo, med, hi = np.quantile(totals, [QUANTILES[0], 0.5, QUANTILES[1]])
                sim_p10.append(lo); sim_p50.append(med); sim_p90.append(hi)
                actual_pts.append(a_tot); pred_pts.append(p_tot)
            sim_p10 = np.asarray(sim_p10); sim_p50 = np.asarray(sim_p50)
            sim_p90 = np.asarray(sim_p90)
            actual_pts = np.asarray(actual_pts); pred_pts = np.asarray(pred_pts)
            out_rows.append({
                "test_season": int(season), "model": model, "n": int(len(sim_p50)),
                "p50_mae": float(np.mean(np.abs(sim_p50 - actual_pts))),
                "pred_mae": float(np.mean(np.abs(pred_pts - actual_pts))),
                "p50_minus_pred": float(np.mean(sim_p50 - pred_pts)),
                "p50_bias": float(np.mean(sim_p50 - actual_pts)),
                "coverage": float(np.mean((sim_p10 <= actual_pts) & (actual_pts <= sim_p90))),
                "pred_inside_band": float(
                    np.mean((sim_p10 <= pred_pts) & (pred_pts <= sim_p90))),
                "mean_width": float(np.mean(sim_p90 - sim_p10)),
            })
    detail = pd.DataFrame(out_rows)
    summary = (
        detail.groupby("model")
        .apply(lambda d: pd.Series({
            "folds": len(d),
            "p50_mae": np.average(d["p50_mae"], weights=d["n"]),
            "pred_mae": np.average(d["pred_mae"], weights=d["n"]),
            "p50_minus_pred": np.average(d["p50_minus_pred"], weights=d["n"]),
            "p50_bias": np.average(d["p50_bias"], weights=d["n"]),
            "coverage": np.average(d["coverage"], weights=d["n"]),
            "pred_inside_band": np.average(d["pred_inside_band"], weights=d["n"]),
            "mean_width": np.average(d["mean_width"], weights=d["n"]),
        }), include_groups=False)
        .reset_index()
    )
    return {"summary": summary.to_dict("records"), "by_fold": detail.to_dict("records")}


def run(residuals: pd.DataFrame) -> dict:
    rows = []
    for (position, stat), grp in residuals.groupby(["position", "stat"], observed=True):
        seasons = sorted(grp["test_season"].unique())
        for season in seasons[1:]:
            cal = grp[grp["test_season"] < season]
            test = grp[grp["test_season"] == season]
            if cal.empty or test.empty:
                continue
            pool = cal["resid"].to_numpy(dtype=float)
            cal_pred = cal["pred"].to_numpy(dtype=float)
            cal_actual = cal["actual"].to_numpy(dtype=float)
            usable = cal_pred >= RATIO_PRED_FLOOR
            ratios = (
                cal_actual[usable] / cal_pred[usable] if usable.any() else np.array([])
            )
            for model in MODELS:
                med, lo, hi, covered = [], [], [], []
                for pred, actual in zip(
                    test["pred"].to_numpy(dtype=float),
                    test["actual"].to_numpy(dtype=float),
                ):
                    draws = _draws_for_row(pred, pool, ratios, model)
                    q_lo, q_med, q_hi = np.quantile(draws, [QUANTILES[0], 0.5, QUANTILES[1]])
                    med.append(q_med)
                    lo.append(q_lo)
                    hi.append(q_hi)
                    covered.append(bool(q_lo <= actual <= q_hi))
                med = np.asarray(med); lo = np.asarray(lo); hi = np.asarray(hi)
                actual = test["actual"].to_numpy(dtype=float)
                pred = test["pred"].to_numpy(dtype=float)
                rows.append({
                    "position": position, "stat": stat, "test_season": int(season),
                    "model": model, "n": int(len(test)),
                    # Point quality of the median draw, and of the raw point
                    # estimate it is meant to reproduce.
                    "median_mae": float(np.mean(np.abs(med - actual))),
                    "median_bias": float(np.mean(med - actual)),
                    "pred_mae": float(np.mean(np.abs(pred - actual))),
                    # How far the median draw drifts from the point estimate.
                    "median_minus_pred": float(np.mean(med - pred)),
                    "coverage": float(np.mean(covered)),
                    "coverage_gap": float(np.mean(covered) - TARGET_COVERAGE),
                    "mean_width": float(np.mean(hi - lo)),
                })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("model")
        .apply(lambda d: pd.Series({
            "cells": len(d),
            "median_mae": np.average(d["median_mae"], weights=d["n"]),
            "pred_mae": np.average(d["pred_mae"], weights=d["n"]),
            "median_bias": np.average(d["median_bias"], weights=d["n"]),
            "median_minus_pred": np.average(d["median_minus_pred"], weights=d["n"]),
            "coverage": np.average(d["coverage"], weights=d["n"]),
            "mean_width": np.average(d["mean_width"], weights=d["n"]),
        }), include_groups=False)
        .reset_index()
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_coverage": TARGET_COVERAGE,
        "note": (
            "Calibrated on strictly earlier test seasons, scored on the held-out "
            "one. median_minus_pred is the drift between the simulated median "
            "and the point estimate it should reproduce; a value far from 0 is "
            "the bias that makes a simulated board disagree with its own board."
        ),
        "summary": summary.to_dict("records"),
        "by_cell": detail.to_dict("records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    path = Path(BACKTEST_DIR) / "residuals_rolling.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run scripts/run_rolling_backtest.py first")
    residuals = pd.read_parquet(path)
    report = run(residuals)
    report["fantasy_level"] = run_fantasy_level(residuals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.out}\n")
    s = pd.DataFrame(report["summary"])
    print(f"PER-STAT RATE level (diagnostic only - inverts the answer, see "
          f"run_fantasy_level), target coverage {TARGET_COVERAGE:.2f}\n")
    print(f"{'model':20s} {'med MAE':>9s} {'pred MAE':>9s} {'med-pred':>9s} "
          f"{'bias':>8s} {'coverage':>9s} {'width':>8s}")
    for _, r in s.iterrows():
        print(f"{r['model']:20s} {r['median_mae']:9.4f} {r['pred_mae']:9.4f} "
              f"{r['median_minus_pred']:+9.4f} {r['median_bias']:+8.4f} "
              f"{r['coverage']:9.4f} {r['mean_width']:8.4f}")

    f = pd.DataFrame(report["fantasy_level"]["summary"])
    print(f"\nSUMMED FANTASY POINTS - the shipped grain, target coverage "
          f"{TARGET_COVERAGE:.2f}\n")
    print(f"{'model':20s} {'p50 MAE':>9s} {'pred MAE':>9s} {'p50-pred':>9s} "
          f"{'bias':>8s} {'coverage':>9s} {'pred in':>8s} {'width':>8s}")
    for _, r in f.iterrows():
        print(f"{r['model']:20s} {r['p50_mae']:9.4f} {r['pred_mae']:9.4f} "
              f"{r['p50_minus_pred']:+9.4f} {r['p50_bias']:+8.4f} "
              f"{r['coverage']:9.4f} {r['pred_inside_band']:8.4f} {r['mean_width']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
