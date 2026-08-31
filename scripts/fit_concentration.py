"""Fit and promote leakage-safe team-position concentration exponents."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.concentration import CONCENTRATION_FAMILIES
from src.projection.contracts import CONCENTRATION_PATH
from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features
from src.projection.fantasy_evaluation import build_leakage_safe_long_board


GAMMAS = np.round(np.arange(1.0, 1.501, 0.05), 2)


def transform_predictions(rows: pd.DataFrame, gamma: float) -> pd.Series:
    pred = pd.to_numeric(rows["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    adjusted = pred.copy()
    keys = [rows["test_season"], rows["stat"], rows["team"].fillna("__NO_TEAM__")]
    for _, idx in rows.groupby(keys, observed=True).groups.items():
        raw = pred.loc[idx]
        positive = raw.gt(0)
        if positive.sum() <= 1 or raw.sum() <= 0:
            continue
        powered = raw.loc[positive].pow(gamma)
        adjusted.loc[powered.index] = powered * raw.sum() / powered.sum()
    return adjusted


def score(rows: pd.DataFrame, pred: pd.Series) -> dict:
    actual = pd.to_numeric(rows["actual"], errors="coerce")
    valid = actual.notna() & pred.notna()
    actual, pred = actual[valid], pred[valid]
    error = pred - actual
    mae = float(error.abs().mean()) if len(error) else float("nan")
    bias = float(error.mean()) if len(error) else float("nan")
    spearman = float(pred.corr(actual, method="spearman")) if len(error) > 1 else float("nan")
    return {"n": int(len(error)), "mae": mae, "signed_bias": bias, "spearman": spearman}


def objective(rows: pd.DataFrame, gamma: float) -> float:
    adjusted = transform_predictions(rows, gamma)
    ratios = []
    for stat, part in rows.groupby("stat", observed=True):
        baseline = score(part, part["pred"])["mae"]
        candidate = score(part, adjusted.loc[part.index])["mae"]
        if baseline > 0 and np.isfinite(candidate):
            ratios.append(candidate / baseline)
    return float(np.mean(ratios)) if ratios else float("inf")


def best_gamma(rows: pd.DataFrame) -> float:
    candidates = [(objective(rows, float(g)), float(g)) for g in GAMMAS]
    # Prefer the least aggressive exponent on an exact tie.
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _top_role_mask(rows: pd.DataFrame) -> pd.Series:
    ranks = rows.groupby(["test_season", "stat", "team"], observed=True)["actual"].rank(
        method="first", ascending=False
    )
    return ranks.eq(1)


def _stable_producer_ids(feat: pd.DataFrame, rows: pd.DataFrame, position: str,
                         family: str) -> set[tuple[int, str]]:
    usage = "receiving_yards" if family == "receiving" else "rushing_yards"
    hist = feat[feat["position"].eq(position)].copy()
    hist["_usage"] = pd.to_numeric(hist.get(usage), errors="coerce").fillna(0.0)
    hist["_top2"] = hist.groupby(["season", "team", "position"], observed=True)["_usage"].rank(
        method="first", ascending=False
    ).le(2)
    out: set[tuple[int, str]] = set()
    for season in sorted(rows["test_season"].unique()):
        prior = hist[hist["season"].between(int(season) - 3, int(season) - 1)]
        counts = prior[prior["_top2"]].groupby("player_id", observed=True)["season"].nunique()
        out.update((int(season), str(pid)) for pid in counts[counts.ge(2)].index)
    return out


def fit_cell(rows: pd.DataFrame, feat: pd.DataFrame, position: str, family: str) -> dict:
    seasons = sorted(int(s) for s in rows["test_season"].unique())
    nested_parts = []
    fold_metrics = []
    for season in seasons:
        prior = rows[rows["test_season"].lt(season)]
        held = rows[rows["test_season"].eq(season)]
        if prior.empty or held.empty:
            continue
        gamma = best_gamma(prior)
        adjusted = transform_predictions(held, gamma)
        nested_parts.append(held.assign(candidate=adjusted))
        top = _top_role_mask(held)
        b_top = score(held[top], held.loc[top, "pred"])
        c_top = score(held[top], adjusted.loc[top])
        fold_metrics.append({
            "season": season,
            "selected_exponent": gamma,
            "top_role_mae_before": b_top["mae"],
            "top_role_mae_after": c_top["mae"],
            "top_role_bias_before": b_top["signed_bias"],
            "top_role_bias_after": c_top["signed_bias"],
        })

    production_gamma = best_gamma(rows)
    production_adjusted = transform_predictions(rows, production_gamma)
    nested = pd.concat(nested_parts, ignore_index=False) if nested_parts else pd.DataFrame()
    eval_rows = nested if not nested.empty else rows.assign(candidate=production_adjusted)
    before = score(eval_rows, eval_rows["pred"])
    after = score(eval_rows, eval_rows["candidate"])

    per_stat = {}
    no_stat_worse = True
    spearman_ok = True
    for stat, part in eval_rows.groupby("stat", observed=True):
        b = score(part, part["pred"])
        a = score(part, part["candidate"])
        ratio = a["mae"] / b["mae"] if b["mae"] else 1.0
        decline = b["spearman"] - a["spearman"]
        per_stat[str(stat)] = {
            "mae_before": b["mae"], "mae_after": a["mae"], "ratio": ratio,
            "spearman_before": b["spearman"], "spearman_after": a["spearman"],
            "spearman_decline": decline,
        }
        no_stat_worse &= ratio <= 1.01
        spearman_ok &= not np.isfinite(decline) or decline <= 0.01

    top_improvement_folds = sum(
        (m["top_role_mae_after"] < m["top_role_mae_before"])
        or (abs(m["top_role_bias_after"]) < abs(m["top_role_bias_before"]))
        for m in fold_metrics[-3:]
    )
    required_top_folds = min(2, len(fold_metrics[-3:]))

    stable_ids = _stable_producer_ids(feat, rows, position, family)
    stable = rows.apply(lambda r: (int(r["test_season"]), str(r["player_id"])) in stable_ids, axis=1)
    yard_stat = "receiving_yards" if family == "receiving" else "rushing_yards"
    stable_yards = stable & rows["stat"].eq(yard_stat)
    stable_before = score(rows[stable_yards], rows.loc[stable_yards, "pred"])
    stable_after = score(rows[stable_yards], production_adjusted.loc[stable_yards])
    stable_improves = (
        stable_before["n"] == 0
        or abs(stable_after["signed_bias"]) < abs(stable_before["signed_bias"])
    )

    pooled_improves = after["mae"] < before["mae"]
    promoted = bool(
        production_gamma > 1.0
        and pooled_improves
        and no_stat_worse
        and top_improvement_folds >= required_top_folds
        and spearman_ok
        and stable_improves
    )
    return {
        "exponent": production_gamma if promoted else 1.0,
        "fitted_exponent": production_gamma,
        "promoted": promoted,
        "fit_seasons": seasons,
        "sample_count": int(len(rows)),
        "metrics": {
            "pooled_before": before,
            "pooled_after": after,
            "per_stat": per_stat,
            "top_role_improvement_folds": int(top_improvement_folds),
            "required_top_role_folds": int(required_top_folds),
            "spearman_decline": before["spearman"] - after["spearman"],
            "stable_producer_before": stable_before,
            "stable_producer_after": stable_after,
            "stable_producer_bias_improved": bool(stable_improves),
            "folds": fold_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=CONCENTRATION_PATH)
    args = parser.parse_args()
    conn = get_conn()
    try:
        feat = build_player_season_features(conn)
        board_rows = []
        for target_season in (2023, 2024, 2025):
            board = build_leakage_safe_long_board(
                conn, feat, target_season - 1, target_season
            )
            actual = feat[feat["season"].eq(target_season)].copy()
            for stat in sorted({s for values in CONCENTRATION_FAMILIES.values() for s in values}):
                if stat not in actual.columns:
                    continue
                lookup = actual.drop_duplicates(["player_id", "position"])[
                    ["player_id", "position", stat]
                ].rename(columns={stat: "actual"})
                part = board[board["stat"].eq(stat)].merge(
                    lookup, on=["player_id", "position"], how="left"
                )
                part["actual"] = pd.to_numeric(part["actual"], errors="coerce").fillna(0.0)
                part["pred"] = pd.to_numeric(part["pred_season"], errors="coerce").fillna(0.0)
                part["test_season"] = target_season
                board_rows.append(part[[
                    "position", "stat", "test_season", "player_id", "team", "pred", "actual"
                ]])
        residuals = pd.concat(board_rows, ignore_index=True)
    finally:
        conn.close()
    cells = {}
    for (position, family), stats in CONCENTRATION_FAMILIES.items():
        rows = residuals[
            residuals["position"].eq(position) & residuals["stat"].isin(stats)
        ].copy()
        if not rows.empty:
            cells[f"{position}:{family}"] = fit_cell(rows, feat, position, family)
    artifact = {
        "version": "concentration_v1_full_board",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_exponents": GAMMAS.tolist(),
        "fit_method": "nested_rolling_origin_full_board_player_season_power_share",
        "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    temp = f"{args.out}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
    os.replace(temp, args.out)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
