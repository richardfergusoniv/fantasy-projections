"""Fit leakage-safe team volume reconciliation alpha weights."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.contracts import RECONCILE_CALIBRATION_PATH, TEAM_RECONCILE_ALPHA
from src.projection.data_prep import get_conn
from src.projection.fantasy_evaluation import build_leakage_safe_long_board
from src.projection.features import build_player_season_features
from src.projection.team_reconcile import add_projected_season_totals, reconcile_team_volume

ALPHAS = [0.25, 0.50, 0.75, 1.0]
FOLDS = [(2022, 2023), (2023, 2024), (2024, 2025)]


def _season_total_mae(frame: pd.DataFrame) -> float:
    pred = pd.to_numeric(frame["pred_season"], errors="coerce")
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    mask = pred.notna() & actual.notna()
    if not mask.any():
        return float("inf")
    return float((pred[mask] - actual[mask]).abs().mean())


def _attach_actuals(frame: pd.DataFrame, feat: pd.DataFrame, target_season: int) -> pd.DataFrame:
    actual_lookup = feat[feat["season"] == target_season].set_index("player_id")
    out = frame.copy()
    out["actual"] = float("nan")
    for stat in out["stat"].dropna().unique():
        if stat not in actual_lookup.columns:
            continue
        mask = out["stat"].eq(stat)
        out.loc[mask, "actual"] = out.loc[mask, "player_id"].map(actual_lookup[stat])
    return out


def _restore_pre_reconcile(board: pd.DataFrame) -> pd.DataFrame:
    """Undo the shipped reconcile step so alpha can be re-swept."""
    out = board.copy()
    scale = pd.to_numeric(out.get("team_volume_scale"), errors="coerce").fillna(1.0)
    scale = scale.replace(0, 1.0)
    for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / scale
    out["team_volume_scale"] = 1.0
    return out


def score_alpha(long_board: pd.DataFrame, feat: pd.DataFrame, target_season: int, alpha: float) -> float:
    if long_board.empty:
        return float("inf")
    base = _restore_pre_reconcile(long_board)
    reconciled = add_projected_season_totals(reconcile_team_volume(base, alpha=alpha))
    scored = _attach_actuals(reconciled, feat, target_season)
    return _season_total_mae(scored)


def fit_reconcile_alpha() -> dict:
    conn = get_conn()
    feat = build_player_season_features(conn)
    rows = []
    board_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for source, target in FOLDS:
        print(f"Building fold {source}->{target}...", flush=True)
        board_cache[(source, target)] = build_leakage_safe_long_board(
            conn, feat, source, target
        )
        for alpha in ALPHAS:
            mae = score_alpha(board_cache[(source, target)], feat, target, alpha)
            rows.append({
                "source_season": source,
                "target_season": target,
                "alpha": alpha,
                "mae": mae,
            })
            print(f"  alpha={alpha:.2f} mae={mae:.2f}", flush=True)
    conn.close()
    grid = pd.DataFrame(rows)
    summary = (
        grid.groupby("alpha")["mae"]
        .mean()
        .reset_index()
        .sort_values("mae")
    )
    best = float(summary.iloc[0]["alpha"]) if not summary.empty else TEAM_RECONCILE_ALPHA
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_alpha": best,
        "fallback_alpha": TEAM_RECONCILE_ALPHA,
        "grid": grid.to_dict("records"),
        "summary": summary.to_dict("records"),
    }
    os.makedirs(os.path.dirname(RECONCILE_CALIBRATION_PATH), exist_ok=True)
    with open(RECONCILE_CALIBRATION_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    manifest = fit_reconcile_alpha()
    print(f"Best alpha: {manifest['default_alpha']}")
    print(f"Wrote {RECONCILE_CALIBRATION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
