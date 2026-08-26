"""Measure season-scale efficiency dispersion for the generative conversions.

``draw_receiving_line`` and friends multiply volume by a lognormal efficiency
factor. Its sigma was chosen when the path emitted PER-GAME lines, where a
player's efficiency swings far more than it does over a season, and it was
never re-derived after the path moved to season totals.

The right value is the spread of a player's realised season efficiency around
the efficiency his own projection implied -- i.e. the SD of

    log( (actual_yards / actual_volume) / (pred_yards / pred_volume) )

over held-out player-seasons. Reads the rolling residuals, so it needs no
model refit.

Usage:
    python scripts/fit_conversion_sigmas.py
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

# kind -> (yards stat, volume stat)
CONVERSIONS = {
    "receiving": ("receiving_yards", "receptions"),
    "passing": ("passing_yards", "completions"),
    "rushing": ("rushing_yards", "carries"),
}
MIN_VOLUME = 0.5   # rate units; below this the ratio is noise
MIN_CELL = 30
OUT_PATH = ROOT / "output" / "backtest" / "conversion_sigmas.json"


def _efficiency(frame: pd.DataFrame, value_col: str, num: str, den: str) -> pd.Series:
    piv = frame.pivot_table(
        index=["player_id", "test_season", "position"], columns="stat", values=value_col)
    if num not in piv.columns or den not in piv.columns:
        return pd.Series(dtype=float)
    volume = piv[den]
    return (piv[num] / volume).where(volume > MIN_VOLUME)


def fit(residuals: pd.DataFrame) -> dict:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "cells": {}, "defaults": {}}
    for kind, (num, den) in CONVERSIONS.items():
        actual = _efficiency(residuals, "actual", num, den)
        pred = _efficiency(residuals, "pred", num, den)
        if actual.empty or pred.empty:
            continue
        frame = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
        frame = frame[(frame["actual"] > 0) & (frame["pred"] > 0)]
        if frame.empty:
            continue
        log_ratio = np.log(frame["actual"] / frame["pred"])
        out["defaults"][kind] = {
            "sigma": float(log_ratio.std()), "n": int(len(log_ratio))}
        positions = frame.index.get_level_values("position")
        for position in sorted(set(positions)):
            sub = log_ratio[positions == position]
            if len(sub) < MIN_CELL:
                continue
            out["cells"][f"{position}:{kind}"] = {
                "sigma": float(sub.std()), "n": int(len(sub))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    path = Path(BACKTEST_DIR) / "residuals_rolling.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run scripts/run_rolling_backtest.py first")
    report = fit(pd.read_parquet(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}\n")
    print("season-scale efficiency sigma  (SD of log actual/predicted)\n")
    for key, cell in sorted(report["cells"].items()):
        print(f"  {key:18s} {cell['sigma']:.3f}   n={cell['n']}")
    print()
    for kind, cell in sorted(report["defaults"].items()):
        print(f"  default {kind:10s} {cell['sigma']:.3f}   n={cell['n']}")
    print("\nThese are the values hard-coded in inference/reconcile.SEASON_SIGMA;")
    print("re-run after a rolling backtest to check they have not drifted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
