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
    calibration_path = REPO_ROOT / "output" / "backtest" / "v3_fantasy_interval_calibration.json"
    uncertainty_path = REPO_ROOT / "models" / "v3" / "uncertainty" / "manifest.json"
    simulation_manifest_path = OUT_DIR / f"simulation_manifest_{season}.json"
    calibration = _read_json(calibration_path) or {}
    uncertainty = _read_json(uncertainty_path) or {}
    simulation_manifest = _read_json(simulation_manifest_path) or {}
    means_backtest = _read_json(OUT_DIR / "means_backtest.json") or {}
    v3_summary = OUT_DIR / f"simulation_summary_{season}.csv"

    selected_mode = calibration.get("selected_distribution_mode")
    acceptance_key = {
        "generative_projection_uncertainty": "option_a_acceptance",
        "joint_bootstrap": "fallback_acceptance",
    }.get(selected_mode)
    selected_acceptance = calibration.get(acceptance_key) or {} if acceptance_key else {}
    selected_arm = {
        "generative_projection_uncertainty": "option_a",
        "joint_bootstrap": "joint_bootstrap",
    }.get(selected_mode)
    aggregate = calibration.get("aggregate") or {}
    selected_metrics = aggregate.get(selected_arm, {}).get("overall", {}) if selected_arm else {}
    coverage = selected_metrics.get("coverage")

    # These checks deliberately fail closed. The gate is valid only for the
    # exact season-fantasy-points production path, and only when the live full
    # simulation used the same selected distribution and uncertainty artifact.
    calibration_basis_ok = calibration.get("basis") == (
        "rolling_origin_exact_full_simulator_season_fantasy_points"
    ) and calibration.get("simulation_mode") == "full"
    calibration_ok = bool(selected_acceptance.get("pass") and calibration_basis_ok)
    uncertainty_matches = bool(
        uncertainty
        and uncertainty.get("selected_distribution_mode") == selected_mode
        and uncertainty.get("calibration_artifact") == str(calibration_path)
    )
    simulation_mode_ok = simulation_manifest.get("mode") == "full"
    simulation_distribution_ok = simulation_manifest.get("distribution_mode") == selected_mode
    simulation_hash_ok = bool(
        uncertainty.get("artifact_hash")
        and simulation_manifest.get("uncertainty_artifact_hash") == uncertainty.get("artifact_hash")
    )
    simulation_ready = bool(
        v3_summary.exists()
        and calibration_ok
        and uncertainty_matches
        and simulation_mode_ok
        and simulation_distribution_ok
        and simulation_hash_ok
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
        if calibration and selected_mode == "hold":
            rationale = (
                "Keep v1 (+ optional v1/v2 draft blend) as the point engine and do not "
                "publish a v3 distributional overlay. Neither the bounded generative "
                "calibration nor the exact-grain bootstrap fallback cleared every gate."
            )
        else:
            rationale = (
                "Keep v1 (+ optional v1/v2 draft blend) as the point engine. "
                "The exact season-distribution calibration or its matching full-simulation "
                "artifacts are missing, stale, or differently configured."
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "verdict": verdict,
        "rationale": rationale,
        "gates": {
            "simulation_ready": simulation_ready,
            "season_distribution_calibration_passed": calibration_ok,
            "selected_distribution_mode": selected_mode or "missing",
            "p10_p90_coverage": coverage,
            "season_distribution_basis": calibration.get("basis") or "missing",
            "season_distribution_n_scored": selected_metrics.get("n"),
            "season_distribution_acceptance": selected_acceptance.get("gates") or {},
            "uncertainty_manifest_matches_calibration": uncertainty_matches,
            "simulation_mode_full": simulation_mode_ok,
            "simulation_distribution_matches": simulation_distribution_ok,
            "simulation_uncertainty_hash_matches": simulation_hash_ok,
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
