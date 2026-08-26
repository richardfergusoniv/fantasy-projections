"""Held-out interval quality for the interim vs generative simulation modes.

The interim arm's published bands cover 0.546 against a 0.80 target, because
per-stat residuals are drawn independently while a player's stats correlate
+0.62 to +0.88. The generative path is expected to do better for a structural
reason -- a player's stats all descend from one drawn volume, so they move
together -- but "expected to" is not a measurement, and these percentiles
ship next to the numbers they describe.

This scores both modes on a genuinely held-out season: the board is built
from history through ``source_season`` only, then compared against that
season's realised fantasy points.

Usage:
    python scripts/compare_simulation_modes.py --season 2025
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

from src.projection.data_prep import get_conn
from src.projection.fantasy_evaluation import (
    attach_actual_outcomes,
    build_leakage_safe_long_board,
)
from src.projection.features import build_player_season_features
from src.projection.fantasy_points import SCORING
from src.projection.inference.simulate import (
    simulate_season_distributions,
    summarize_simulations,
)

OUT_PATH = ROOT / "output" / "backtest" / "simulation_mode_comparison.json"
TARGET = 0.80


def _actual_fantasy_points(conn, feat, season: int) -> pd.Series:
    """Realised season fantasy points, via the shipped outcome helper."""
    frame = pd.DataFrame({"player_id": feat["player_id"].unique()})
    scored = attach_actual_outcomes(frame, feat, season)
    return scored.set_index("player_id")["actual_points"]


def score_mode(
    board: pd.DataFrame, actual: pd.Series, *, mode: str, n_draws: int
) -> dict:
    draws = simulate_season_distributions(board, n_draws=n_draws, mode=mode)
    if draws.empty:
        return {"mode": mode, "n": 0}
    summary = summarize_simulations(draws).set_index("player_id")
    joined = summary.join(actual.rename("actual"), how="inner").dropna(
        subset=["p10", "p50", "p90", "actual"])
    if joined.empty:
        return {"mode": mode, "n": 0}

    def _stats(frame: pd.DataFrame) -> dict:
        covered = (frame["p10"] <= frame["actual"]) & (frame["actual"] <= frame["p90"])
        return {
            "n": int(len(frame)),
            "coverage": float(covered.mean()),
            "coverage_gap": float(covered.mean() - TARGET),
            "mean_width": float((frame["p90"] - frame["p10"]).mean()),
            "p50_mae": float((frame["p50"] - frame["actual"]).abs().mean()),
            "p50_bias": float((frame["p50"] - frame["actual"]).mean()),
            "p50_spearman": float(frame["p50"].corr(frame["actual"], method="spearman")),
        }

    out = {"mode": mode, **_stats(joined)}
    out["by_position"] = {
        str(pos): _stats(grp) for pos, grp in joined.groupby("position", observed=True)
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    target = args.season
    source = target - 1

    conn = get_conn()
    feat = build_player_season_features(conn)
    print(f"building leakage-safe board {source}->{target}...", flush=True)
    board = build_leakage_safe_long_board(conn, feat, source, target)
    actual = _actual_fantasy_points(conn, feat, target)
    conn.close()

    results = []
    for mode in ("interim", "full"):
        print(f"simulating mode={mode} ({args.draws} draws)...", flush=True)
        results.append(score_mode(board, actual, mode=mode, n_draws=args.draws))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_season": target,
        "source_season": source,
        "n_draws": args.draws,
        "target_coverage": TARGET,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}\n")
    print(f"held out on {target}, target coverage {TARGET:.2f}\n")
    print(f"{'mode':10s} {'n':>5s} {'coverage':>9s} {'width':>9s} "
          f"{'p50 MAE':>9s} {'p50 bias':>9s} {'rho':>7s}")
    for r in results:
        if not r.get("n"):
            print(f"{r['mode']:10s} {'-':>5s}  (no draws)")
            continue
        print(f"{r['mode']:10s} {r['n']:5d} {r['coverage']:9.4f} {r['mean_width']:9.2f} "
              f"{r['p50_mae']:9.3f} {r['p50_bias']:+9.3f} {r['p50_spearman']:7.4f}")
        for pos, cell in sorted((r.get("by_position") or {}).items()):
            print(f"    {pos:6s} {cell['n']:5d} {cell['coverage']:9.4f} {cell['mean_width']:9.2f} "
                  f"{cell['p50_mae']:9.3f} {cell['p50_bias']:+9.3f} {cell['p50_spearman']:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
