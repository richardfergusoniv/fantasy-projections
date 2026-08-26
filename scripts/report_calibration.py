"""Generate calibration report from persisted backtest residuals."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projection.contracts import BACKTEST_DIR
from src.projection.evaluation.calibration import (
    coverage_by_group,
    reliability_table,
    summarize_forward_interval_calibration,
    summarize_interval_calibration,
)


def build_report(backtest_dir: Path) -> dict:
    residuals_path = backtest_dir / "residuals_rolling.parquet"
    if not residuals_path.exists():
        raise FileNotFoundError(f"Missing {residuals_path}; run scripts/run_rolling_backtest.py first")
    residuals = pd.read_parquet(residuals_path)
    lo_q, hi_q = 0.10, 0.90
    rows = []
    for (position, stat), grp in residuals.groupby(["position", "stat"], observed=True):
        lo, hi = grp["resid"].quantile([lo_q, hi_q])
        frame = grp.copy()
        frame["pred_low"] = frame["pred"] + lo
        frame["pred_high"] = frame["pred"] + hi
        rows.append({
            "position": position,
            "stat": stat,
            "coverage": float(frame["actual"].between(frame["pred_low"], frame["pred_high"]).mean()),
            "reliability": reliability_table(frame["actual"], frame["pred"]).to_dict("records"),
        })
    coverage_groups = []
    for season in sorted(residuals["test_season"].unique()):
        sub = residuals[residuals["test_season"] == season].copy()
        lo, hi = sub["resid"].quantile([lo_q, hi_q])
        sub["pred_low"] = sub["pred"] + lo
        sub["pred_high"] = sub["pred"] + hi
        cov = coverage_by_group(sub, group_cols=["position"])
        cov["test_season"] = int(season)
        coverage_groups.append(cov)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # `summary` is in-sample and lands on the nominal target by
        # construction; `forward_summary` is the held-out number, and is what
        # the promotion gate reads. Both are kept so the gap between them is
        # visible rather than inferred.
        "summary": summarize_interval_calibration(residuals),
        "forward_summary": summarize_forward_interval_calibration(residuals),
        "by_position_stat": rows,
        "coverage_by_position_season": pd.concat(coverage_groups, ignore_index=True).to_dict("records")
        if coverage_groups
        else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(BACKTEST_DIR))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.dir)
    out = args.out or (args.dir / "calibration_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Mean coverage (in-sample, target 0.80): {report['summary'].get('mean_coverage')}")
    fwd = report.get("forward_summary") or {}
    print(
        f"Mean coverage (held-out, target 0.80):   {fwd.get('mean_coverage')}"
        f"  [n_scored={fwd.get('n_scored')}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
