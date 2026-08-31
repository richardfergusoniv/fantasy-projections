"""Rolling leakage-safe fantasy evaluation across multiple target seasons.

Wraps src.projection.fantasy_evaluation.run_evaluation without changing
compose_board. Writes per-season CSVs plus a fold-dispersion summary that tags
levers that cannot be measured on historical folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.projection.fantasy_evaluation import run_evaluation  # noqa: E402

OUT_DIR = REPO_ROOT / "output"
SUMMARY_DIR = REPO_ROOT / "output" / "test_before_rewrite"

UNTESTABLE_ON_HISTORY = [
    {
        "lever": "curated_depth_chart",
        "why": "src/depth_chart/starters_<season>.csv is hand-researched for 2026 only",
    },
    {
        "lever": "status_overrides_ir_pup",
        "why": "status_overrides_<season>.csv is 2026-only",
    },
    {
        "lever": "elite_residual_correction",
        "why": "models/corrections.joblib spans the target season on the ship path",
    },
    {
        "lever": "prediction_intervals",
        "why": "interval_residuals.csv is fit across the target season",
    },
    {
        "lever": "roster_vacancy_boosts",
        "why": "vacancy alphas live in roster_moves upstream of the leakage-safe harness",
    },
]


def _headline_from_summary(summary: pd.DataFrame) -> dict:
    """Extract all_eligible / method=model rows by position + n-weighted overall."""
    out: dict = {"by_position": {}}
    if summary.empty:
        return out
    sub = summary[
        (summary["scope"].astype(str) == "all_eligible")
        & (summary["method"].astype(str) == "model")
    ].copy()
    if sub.empty:
        return out
    for _, row in sub.iterrows():
        pos = str(row["position"])
        out["by_position"][pos] = {
            "spearman": float(row["spearman"]) if pd.notna(row["spearman"]) else None,
            "points_mae": float(row["points_mae"]) if pd.notna(row["points_mae"]) else None,
            "n": int(row["n"]) if pd.notna(row["n"]) else None,
        }
    rows = list(out["by_position"].values())
    ns = [r["n"] or 0 for r in rows]
    if sum(ns) > 0:
        out["overall"] = {
            "spearman": sum((r["spearman"] or 0) * (r["n"] or 0) for r in rows) / sum(ns),
            "points_mae": sum((r["points_mae"] or 0) * (r["n"] or 0) for r in rows) / sum(ns),
            "n": sum(ns),
        }
    return out


def run_fold(target_season: int, *, skip_existing: bool) -> dict:
    source_season = target_season - 1
    rows_path = OUT_DIR / f"fantasy_evaluation_{target_season}.csv"
    summary_path = OUT_DIR / f"fantasy_evaluation_summary_{target_season}.csv"
    json_path = OUT_DIR / f"fantasy_evaluation_summary_{target_season}.json"

    if skip_existing and rows_path.exists() and summary_path.exists():
        summary = pd.read_csv(summary_path)
        meta = {}
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8")).get("metadata") or {}
        return {
            "target_season": target_season,
            "source_season": source_season,
            "skipped_existing": True,
            "headline": _headline_from_summary(summary),
            "coverage_limits": meta.get("coverage_limits"),
            "paths": {
                "rows": str(rows_path),
                "summary": str(summary_path),
                "json": str(json_path),
            },
        }

    rows, summary, metadata = run_evaluation(source_season, target_season)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps({"metadata": metadata, "metrics": summary.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    return {
        "target_season": target_season,
        "source_season": source_season,
        "skipped_existing": False,
        "headline": _headline_from_summary(summary),
        "coverage_limits": metadata.get("coverage_limits"),
        "paths": {
            "rows": str(rows_path),
            "summary": str(summary_path),
            "json": str(json_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        default="2023,2024,2025",
        help="Target seasons (source = target-1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Reuse existing fantasy_evaluation_{season}.csv (default true)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when outputs exist",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SUMMARY_DIR / "rolling_eval_dispersion.json",
    )
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    skip = args.skip_existing and not args.force

    folds = []
    for season in seasons:
        print(f"=== fantasy evaluation {season - 1}->{season} (skip_existing={skip}) ===")
        folds.append(run_fold(season, skip_existing=skip))

    # Dispersion table
    dispersion = []
    for fold in folds:
        h = fold.get("headline") or {}
        overall = h.get("overall") or {}
        dispersion.append(
            {
                "target_season": fold["target_season"],
                "spearman": overall.get("spearman"),
                "points_mae": overall.get("points_mae"),
                "n": overall.get("n"),
                "by_position": h.get("by_position"),
                "skipped_existing": fold.get("skipped_existing"),
            }
        )

    spearman_vals = [d["spearman"] for d in dispersion if d["spearman"] is not None]
    mae_vals = [d["points_mae"] for d in dispersion if d["points_mae"] is not None]
    report = {
        "metadata": {
            "seasons": seasons,
            "untestable_on_history": UNTESTABLE_ON_HISTORY,
            "note": (
                "Curated 2026 levers are tagged untestable; do not use them as "
                "rewrite justification until snapshot tables exist for prior years."
            ),
        },
        "folds": folds,
        "dispersion": dispersion,
        "dispersion_summary": {
            "spearman_min": min(spearman_vals) if spearman_vals else None,
            "spearman_max": max(spearman_vals) if spearman_vals else None,
            "spearman_range": (max(spearman_vals) - min(spearman_vals))
            if len(spearman_vals) >= 2
            else None,
            "mae_min": min(mae_vals) if mae_vals else None,
            "mae_max": max(mae_vals) if mae_vals else None,
            "mae_range": (max(mae_vals) - min(mae_vals)) if len(mae_vals) >= 2 else None,
            "n_folds": len(dispersion),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["dispersion_summary"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
