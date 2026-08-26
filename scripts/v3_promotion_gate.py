"""v3 promotion gate: simulation UI readiness vs means cutover.

Verdicts:
- ``simulation_ready`` — artifacts + calibration OK for percentile UI overlay
- ``promote_v3_means`` — generative means beat v1 and v1/v2 blend on rolling fantasy holdouts
- ``hold_v1_default`` — do not use v3 means; keep v1 (+ optional blend) as point engine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "model_v3"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_promotion_gate(season: int = 2026) -> dict:
    calibration = _read_json(REPO_ROOT / "output" / "backtest" / "calibration_report.json") or {}
    means_backtest = _read_json(OUT_DIR / "means_backtest.json") or {}
    v3_summary = OUT_DIR / f"simulation_summary_{season}.csv"

    # Read the HELD-OUT coverage, not the in-sample one. The in-sample figure
    # is the empirical quantile of the rows it scores, so it lands on 0.80 by
    # construction and cannot fail -- gating on it authorised the percentile
    # overlay on a check that could not have said no. A report predating
    # forward_summary fails closed rather than falling back to the in-sample
    # number, which would silently restore the old behaviour.
    forward = calibration.get("forward_summary") or {}
    coverage = forward.get("mean_coverage")
    coverage_in_sample = calibration.get("summary", {}).get("mean_coverage")
    calibration_ok = coverage is not None and abs(float(coverage) - 0.80) <= 0.05
    simulation_ready = bool(
        v3_summary.exists()
        and bool(calibration)
        and calibration_ok
    )

    summary = means_backtest.get("summary") or {}
    means_ready = bool(summary.get("promote_v3_means"))
    # Absent means an older backtest that predates the flag, when the blend
    # arm silently fell back to a copy of v1. Treat that as unusable rather
    # than assuming it was real.
    blend_usable = bool(summary.get("blend_usable_all_folds"))
    interim_ok = bool(summary.get("interim_beats_v1_all_folds")) and bool(
        summary.get("interim_beats_blend_all_folds")
    )
    generative_ok = bool(summary.get("generative_beats_v1_all_folds")) and bool(
        summary.get("generative_beats_blend_all_folds")
    )

    # Pull latest fold metrics for transparency when present.
    fold_metrics = {}
    for fold in means_backtest.get("folds") or []:
        if fold.get("target_season") == 2025 and fold.get("metrics"):
            fold_metrics = {
                "v1_mae": fold["metrics"].get("v1", {}).get("points_mae"),
                "blend_mae": fold["metrics"].get("blend", {}).get("points_mae"),
                "v3_interim_mae": fold["metrics"].get("v3_interim", {}).get("points_mae"),
                "v3_generative_mae": fold["metrics"].get("v3_generative", {}).get("points_mae"),
                "v1_spearman": fold["metrics"].get("v1", {}).get("spearman"),
                "blend_spearman": fold["metrics"].get("blend", {}).get("spearman"),
                "v3_interim_spearman": fold["metrics"].get("v3_interim", {}).get("spearman"),
                "v3_generative_spearman": fold["metrics"].get("v3_generative", {}).get("spearman"),
            }

    if means_ready and simulation_ready:
        verdict = "promote_v3_means"
        rationale = (
            "Generative v3 means beat v1 and v1/v2 blend on rolling fantasy MAE/Spearman; "
            "simulation artifacts ready for distributional UI."
        )
    elif simulation_ready:
        verdict = "simulation_ready"
        rationale = (
            "v3 simulation overlay is ready for percentiles/volatility UI only. "
            "Do not replace the v1 point engine; means backtest has not cleared "
            "promote_v3_means gates."
        )
        if not blend_usable:
            rationale += (
                " NOTE: the v1/v2 blend arm was not usable on every fold, so "
                "'beats blend' was not independently tested -- promotion "
                "cannot clear on this backtest regardless of the v3 numbers."
            )
    else:
        verdict = "hold_v1_default"
        rationale = (
            "Keep v1 (+ optional v1/v2 draft blend) as the point engine. "
            "v3 simulation and/or calibration artifacts are incomplete."
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "verdict": verdict,
        "rationale": rationale,
        "gates": {
            "simulation_ready": simulation_ready,
            "calibration_within_5pp": calibration_ok,
            "mean_interval_coverage": coverage,
            "mean_interval_coverage_basis": forward.get("basis") or "missing",
            "mean_interval_coverage_n_scored": forward.get("n_scored"),
            "mean_interval_coverage_in_sample": coverage_in_sample,
            "v3_simulation_exists": v3_summary.exists(),
            "means_backtest_exists": bool(means_backtest),
            "interim_beats_v1_and_blend": interim_ok,
            "generative_beats_v1_and_blend": generative_ok,
            "blend_arm_usable": blend_usable,
            "blend_unusable_folds": summary.get("blend_unusable_folds") or [],
            "promote_v3_means": means_ready,
            "holdout_2025": fold_metrics,
        },
        "architecture_rule": (
            "v1 compose_board owns point rates/totals; optional v2 blend owns draft mean "
            "points when enabled; v3 owns distributional overlay until promote_v3_means."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "promotion_gate.json")
    args = parser.parse_args()
    report = evaluate_promotion_gate(args.season)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verdict: {report['verdict']}")
    print(report["rationale"])
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
