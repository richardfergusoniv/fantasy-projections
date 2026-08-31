"""CLI for decision-change diagnostics with 20k reference validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.draft_assistant.replacement_contract import default_selected_board_path
from src.projection.contracts import BACKTEST_DIR, MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.evaluation.decision_change_diagnostics import (
    build_decision_change_diagnostic_report,
    extend_diagnostic_reference_draws,
    load_decision_diagnostic_config,
    write_decision_change_diagnostics_artifacts,
)
from src.projection.inference.recenter import sha256_file as recenter_sha256_file
from src.projection.inference.simulation_config import load_simulation_config
from src.projection.inference.wr_calibration import ARTIFACT_PATH as WR_CALIBRATION_PATH
from src.projection.models.uncertainty import load_uncertainty_manifest


def _calibration_hash() -> str:
    path = Path(BACKTEST_DIR) / "v3_fantasy_interval_calibration.json"
    return sha256_file(str(path)) if path.exists() else ""


def _load_projections(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "projection_run_id" not in frame.columns:
        frame["projection_run_id"] = "draw_stability_frozen"
    return frame


def _primary_cache_path(season: int, primary_reference: int) -> Path:
    return (
        Path(MODEL_V3_DIR)
        / "draw_stability"
        / f"season={season}"
        / f"draws={primary_reference}"
        / "recentered_draws.parquet"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run decision-change diagnostics with 20k reference validation",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--projections", type=Path, default=None)
    parser.add_argument("--selected-board", type=Path, default=None)
    parser.add_argument("--primary-reference", type=int, default=None)
    parser.add_argument("--diagnostic-reference", type=int, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory for extending 10k to 20k (defaults to primary cache dir)",
    )
    parser.add_argument("--recentered-draws-10k", type=Path, default=None)
    parser.add_argument("--recentered-draws-20k", type=Path, default=None)
    parser.add_argument("--skip-generate-20k", action="store_true")
    parser.add_argument("--force-regenerate-20k", action="store_true")
    parser.add_argument("--top-adp", type=int, default=120)
    args = parser.parse_args()

    sim_config = load_simulation_config()
    diag_config = load_decision_diagnostic_config(sim_config)
    primary_reference = int(args.primary_reference or diag_config["primary_reference_draws"])
    diagnostic_reference = int(
        args.diagnostic_reference or diag_config["diagnostic_reference_draws"]
    )
    if args.candidates:
        diag_config["candidate_draw_counts"] = [
            int(x) for x in args.candidates.split(",") if x.strip()
        ]

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
    calibration_hash = _calibration_hash()

    primary_path = args.recentered_draws_10k or _primary_cache_path(args.season, primary_reference)
    checkpoint_dir = args.checkpoint_dir or primary_path.parent
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    checkpoint_meta = None
    if meta_path.exists():
        checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    completed_draws = int((checkpoint_meta or {}).get("completed_draws", 0))
    has_full_diagnostic = completed_draws >= diagnostic_reference
    raw_parts = sorted(checkpoint_dir.glob("raw_part_*.parquet"))
    if not raw_parts:
        raise FileNotFoundError(f"Missing raw checkpoint parts in {checkpoint_dir}")

    recentered_primary = None
    recentered_diagnostic = None

    if args.skip_generate_20k:
        if not has_full_diagnostic:
            raise FileNotFoundError(
                f"Cannot skip generation: checkpoint has {completed_draws} draws, "
                f"need {diagnostic_reference}"
            )
    elif not has_full_diagnostic and not args.force_regenerate_20k:
        raise FileNotFoundError(
            f"Checkpoint has {completed_draws} draws; run without --skip-generate-20k "
            f"to extend to {diagnostic_reference}"
        )
    elif not has_full_diagnostic or args.force_regenerate_20k:
        recentered_diagnostic, checkpoint_meta = extend_diagnostic_reference_draws(
            projections,
            season=args.season,
            selected_board=selected_board,
            selected_board_hash=selected_board_hash,
            canonical_projection_run_id=canonical_run_id,
            calibration_hash=calibration_hash,
            wr_calibration_hash=wr_calibration_hash,
            primary_reference_draws=primary_reference,
            diagnostic_reference_draws=diagnostic_reference,
            checkpoint_dir=checkpoint_dir,
            uncertainty_manifest=load_uncertainty_manifest(),
        )

    report = build_decision_change_diagnostic_report(
        season=args.season,
        recentered_primary=recentered_primary,
        recentered_diagnostic=recentered_diagnostic,
        selected_board=selected_board,
        projections=projections,
        selected_board_hash=selected_board_hash,
        wr_calibration_hash=wr_calibration_hash,
        canonical_projection_run_id=canonical_run_id,
        calibration_hash=calibration_hash,
        checkpoint_meta=checkpoint_meta,
        checkpoint_dir=checkpoint_dir,
        config=diag_config,
        top_adp=args.top_adp,
    )
    artifact_paths = write_decision_change_diagnostics_artifacts(report, season=args.season)

    print(
        json.dumps(
            {
                "verdict": report.get("verdict"),
                "reason": report.get("reason"),
                "unique_players_affected": report.get("unique_players_affected"),
                "changes_by_candidate": report.get("changes_by_candidate"),
                "changes_by_category": report.get("changes_by_category"),
                "core_player_events_requiring_review": report.get(
                    "core_player_events_requiring_review"
                ),
                "artifact_paths": artifact_paths,
            },
            indent=2,
        )
    )
    return 0 if report.get("verdict") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
