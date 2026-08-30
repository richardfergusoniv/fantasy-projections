"""CLI for nested-prefix draw-count stability evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.projection.contracts import BACKTEST_DIR, MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import TOP_ADP, sha256_file
from src.projection.evaluation.draw_stability import (
    default_stability_checkpoint_dir,
    evaluate_draw_stability,
    evaluate_holdout_calibration_stability,
    load_intermediate_stability_config,
    load_stability_draws_from_checkpoint,
    stability_contract_hashes,
    write_draw_count_decision,
    write_draw_stability_artifacts,
)
from src.projection.inference.recenter import sha256_file as recenter_sha256_file
from src.projection.inference.simulation_config import load_simulation_config
from src.projection.inference.wr_calibration import ARTIFACT_PATH as WR_CALIBRATION_PATH
from src.draft_assistant.replacement_contract import default_selected_board_path


def _calibration_hash() -> str:
    path = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
    return sha256_file(str(path)) if path.exists() else ""


def _load_projections(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "projection_run_id" not in frame.columns:
        frame["projection_run_id"] = "draw_stability_frozen"
    return frame


def _cache_draw_path(season: int, reference_draws: int) -> Path:
    return (
        Path(MODEL_V3_DIR)
        / "draw_stability"
        / f"season={season}"
        / f"draws={reference_draws}"
        / "recentered_draws.parquet"
    )


def _evaluate_holdout_calibration(
    candidates: list[int],
    reference_draws: int,
) -> dict:
    """Optional holdout calibration comparison using cached 2025 holdout draws."""
    holdout_path = Path(OUTPUT_DIR) / "model_v3" / "holdout_draws_2025.parquet"
    if not holdout_path.exists():
        return {"status": "skipped", "reason": f"missing {holdout_path}"}

    from scripts.evaluate_recentered_distribution import load_holdout_frame
    from src.projection.inference.wr_calibration import load_wr_calibration

    draws = pd.read_parquet(holdout_path)
    holdout = load_holdout_frame()
    wr_scale = float((load_wr_calibration() or {}).get("selected_wr_scale", 1.0))
    result = evaluate_holdout_calibration_stability(
        holdout_draws=draws,
        holdout_frame=holdout,
        candidates=candidates,
        reference_draws=reference_draws,
        wr_scale=wr_scale,
    )
    result["holdout_season"] = 2025
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate nested draw-count stability")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--projections",
        type=Path,
        default=None,
        help="Frozen projections CSV (defaults to output/projections_<season>.csv)",
    )
    parser.add_argument(
        "--selected-board",
        type=Path,
        default=None,
        help="Accuracy-first board CSV",
    )
    parser.add_argument("--reference-draws", type=int, default=10000)
    parser.add_argument("--candidates", type=str, default="1000,2000,5000")
    parser.add_argument("--top-adp", type=int, default=TOP_ADP)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument(
        "--recentered-draws",
        type=Path,
        default=None,
        help="Use existing recentered draw parquet instead of regenerating",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Raw checkpoint directory for prefix-only recenter loading",
    )
    parser.add_argument(
        "--skip-generate-reference",
        action="store_true",
        help="Load reference draws from checkpoint via prefix-only recenter",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["legacy", "production_v20k"],
        default="legacy",
    )
    parser.add_argument(
        "--sweep-phase",
        type=str,
        default=None,
        help="Artifact suffix for sweep phase (e.g. intermediate_v20k)",
    )
    parser.add_argument("--holdout-season", type=int, default=None)
    parser.add_argument("--evaluate-calibration", action="store_true")
    parser.add_argument("--write-decision", action="store_true")
    args = parser.parse_args()

    sim_config = load_simulation_config()
    intermediate_config = load_intermediate_stability_config(sim_config)
    candidates = [int(x) for x in args.candidates.split(",") if x.strip()]
    if args.reference_draws not in candidates:
        candidates.append(args.reference_draws)
    candidates = sorted(set(candidates))

    projections_path = args.projections or Path(OUTPUT_DIR) / f"projections_{args.season}.csv"
    board_path = args.selected_board or default_selected_board_path(args.season)
    if not projections_path.exists():
        raise FileNotFoundError(f"Missing projections: {projections_path}")
    if not board_path.exists():
        raise FileNotFoundError(f"Missing selected board: {board_path}")

    projections = _load_projections(projections_path)
    selected_board = pd.read_csv(board_path)
    selected_board["player_id"] = selected_board["player_id"].astype(str)
    selected_board_hash = sha256_file(str(board_path))
    wr_calibration_hash = (
        recenter_sha256_file(str(WR_CALIBRATION_PATH))
        if WR_CALIBRATION_PATH.exists()
        else ""
    )
    run_ids = projections["projection_run_id"].dropna().unique()
    canonical_run_id = str(run_ids[0]) if len(run_ids) == 1 else "draw_stability_frozen"

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.skip_generate_reference:
        checkpoint_dir = default_stability_checkpoint_dir(
            args.season,
            suffix=str(intermediate_config.get("checkpoint_dir_suffix", "draws=10000")),
        )

    cache_path = _cache_draw_path(args.season, args.reference_draws)
    provenance_verdict = None
    if args.skip_generate_reference:
        if checkpoint_dir is None:
            raise ValueError("--skip-generate-reference requires --checkpoint-dir")
        recentered_draws = load_stability_draws_from_checkpoint(
            checkpoint_dir,
            selected_board,
            max_draws=args.reference_draws,
        )
        diagnostics_path = Path(MODEL_V3_DIR) / f"decision_change_diagnostics_{args.season}.json"
        if diagnostics_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            provenance_verdict = diagnostics.get("verdict")
            if intermediate_config.get("require_provenance_ok") and provenance_verdict != "ok":
                raise ValueError(
                    f"Decision-change diagnostics verdict is {provenance_verdict!r}; "
                    "cannot run intermediate sweep"
                )
    elif args.recentered_draws:
        recentered_draws = pd.read_parquet(args.recentered_draws)
    elif cache_path.exists() and not args.force_regenerate:
        recentered_draws = pd.read_parquet(cache_path)
    else:
        from src.projection.evaluation.draw_stability import generate_recentered_reference_draws
        from src.projection.models.uncertainty import load_uncertainty_manifest

        recentered_draws = generate_recentered_reference_draws(
            projections,
            season=args.season,
            selected_board=selected_board,
            selected_board_hash=selected_board_hash,
            reference_draws=args.reference_draws,
            canonical_projection_run_id=canonical_run_id,
            calibration_hash=_calibration_hash(),
            wr_calibration_hash=wr_calibration_hash,
            uncertainty_manifest=load_uncertainty_manifest(),
            checkpoint_dir=cache_path.parent,
        )

    contract_hashes = stability_contract_hashes(
        season=args.season,
        selected_board_hash=selected_board_hash,
        selected_board=selected_board,
        wr_calibration_hash=wr_calibration_hash,
        canonical_projection_run_id=canonical_run_id,
    )

    report = evaluate_draw_stability(
        recentered_draws,
        season=args.season,
        selected_board=selected_board,
        selected_board_hash=selected_board_hash,
        wr_calibration_hash=wr_calibration_hash,
        candidates=candidates,
        reference_draws=args.reference_draws,
        top_adp=args.top_adp,
        contract_hashes=contract_hashes,
        gate_mode=args.gate_mode,
    )
    report["cache_path"] = str(cache_path)
    report["checkpoint_dir"] = str(checkpoint_dir) if checkpoint_dir else None
    report["projections_path"] = str(projections_path)
    report["selected_board_path"] = str(board_path)
    if args.sweep_phase:
        report["sweep_phase"] = args.sweep_phase

    holdout_calibration = None
    if args.evaluate_calibration and args.holdout_season:
        holdout_calibration = _evaluate_holdout_calibration(
            [c for c in candidates if c != args.reference_draws],
            args.reference_draws,
        )
        report["holdout_calibration"] = holdout_calibration

    write_draw_stability_artifacts(
        report,
        season=args.season,
        sweep_phase=args.sweep_phase,
    )
    if args.write_decision:
        rationale = (
            f"Smallest passing nested-prefix draw count among {candidates} "
            f"vs {args.reference_draws} reference (gate_mode={args.gate_mode})."
        )
        write_draw_count_decision(
            season=args.season,
            stability_report=report,
            holdout_calibration=holdout_calibration,
            rationale=rationale,
            sweep_phase=args.sweep_phase,
            provenance_verdict=provenance_verdict,
            prior_sweep_reference_draws=int(
                intermediate_config.get("prior_sweep_reference_draws", 10000)
            )
            if args.gate_mode == "production_v20k"
            else None,
        )

    print(
        json.dumps(
            {
                "recommended_draw_count": report.get("recommended_draw_count"),
                "reference_draws": report.get("reference_draws"),
                "gate_mode": report.get("gate_mode"),
                "candidates": [
                    {
                        "draw_count": row["draw_count"],
                        "passes_gate": row.get("passes_gate"),
                        "passes_numerical": row.get("passes_numerical"),
                        "material_decision_events": row.get("material_decision_events"),
                        "core_adp_decision_events": row.get("core_adp_decision_events"),
                        "decision_changes": row.get("decision_changes", {}).get("total"),
                    }
                    for row in report.get("candidates", [])
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
