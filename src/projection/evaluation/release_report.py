"""Two-stage release monitoring reports for publish and draft export."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.evaluation.finish_probability_gate import FINISH_GATE_PATH

REPORT_SCHEMA_VERSION = "release_report_v1"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_miss_rates(scored: pd.DataFrame) -> dict[str, float]:
    if scored.empty:
        return {"miss_rate_below_p10": float("nan"), "miss_rate_above_p90": float("nan")}
    actual = pd.to_numeric(scored["actual_points"], errors="coerce")
    p10_col = "p10" if "p10" in scored.columns else "recentered_p10"
    p90_col = "p90" if "p90" in scored.columns else "recentered_p90"
    if p10_col not in scored.columns or p90_col not in scored.columns:
        return {"miss_rate_below_p10": float("nan"), "miss_rate_above_p90": float("nan")}
    p10 = pd.to_numeric(scored[p10_col], errors="coerce")
    p90 = pd.to_numeric(scored[p90_col], errors="coerce")
    valid = actual.notna() & p10.notna() & p90.notna()
    if not valid.any():
        return {"miss_rate_below_p10": float("nan"), "miss_rate_above_p90": float("nan")}
    below = (actual[valid] < p10[valid]).mean()
    above = (actual[valid] > p90[valid]).mean()
    return {
        "miss_rate_below_p10": float(below),
        "miss_rate_above_p90": float(above),
    }


def build_release_report_simulation(
    *,
    season: int,
    projection_run: dict | None = None,
    simulation_manifest: dict | None = None,
) -> dict[str, Any]:
    model_v3 = Path(MODEL_V3_DIR)
    projection_run = projection_run or _read_json(Path(OUTPUT_DIR) / f"projection_run_{season}.json")
    simulation_manifest = simulation_manifest or _read_json(
        model_v3 / f"simulation_manifest_{season}.json"
    )
    finish_gate = _read_json(FINISH_GATE_PATH)
    promotion_gate = _read_json(model_v3 / "promotion_gate.json")
    draw_stability = _read_json(model_v3 / f"draw_stability_{season}.json")
    draw_count_decision = _read_json(model_v3 / "draw_count_decision.json")
    decision_change_diagnostics = _read_json(
        model_v3 / f"decision_change_diagnostics_{season}.json"
    )
    segment_summary = _read_json(
        Path(OUTPUT_DIR) / "evaluation" / f"season={season}" / "calibration_segment_summary.json"
    )
    holdout = _read_json(model_v3 / "recentered_holdout_2025.json")
    wr_calibration = _read_json(model_v3 / "wr_calibration.json")

    risks: list[str] = []
    if draw_stability is None:
        risks.append("draw_stability: not_run")
    if decision_change_diagnostics is None:
        risks.append("decision_change_diagnostics: not_run")
    elif decision_change_diagnostics.get("verdict") == "hold":
        risks.append(
            f"decision_change_diagnostics: {decision_change_diagnostics.get('reason')}"
        )
    elif (decision_change_diagnostics.get("changes_by_category") or {}).get(
        "material"
    ) or (decision_change_diagnostics.get("changes_by_category") or {}).get(
        "reference_instability"
    ):
        risks.append("decision_change_diagnostics: material_or_reference_instability")
    if finish_gate is None:
        risks.append("finish_probability_gate: missing")
    elif (finish_gate.get("state") or finish_gate.get("verdict")) not in {
        "finish_probability_ready",
    }:
        risks.append(
            f"finish_probability_gate: {(finish_gate.get('state') or finish_gate.get('verdict'))}"
        )
    if promotion_gate is None:
        risks.append("promotion_gate: missing")

    draw_count_rollout = _read_json(model_v3 / "draw_count_rollout_decision.json")
    draw_count_policy = None
    if isinstance(draw_count_rollout, dict):
        strict_gate = draw_count_rollout.get("strict_gate_promotion")
        if strict_gate is False:
            # Lazy import avoids prepare ↔ release_report ↔ draw_count_rollout cycles.
            from src.projection.evaluation.draw_count_rollout import DRAW_COUNT_RISK_FLAG_10K

            risk_flag = str(
                draw_count_rollout.get("release_report_risk_flag") or DRAW_COUNT_RISK_FLAG_10K
            )
            if risk_flag not in risks:
                risks.append(risk_flag)
            draw_count_policy = {
                "profile": draw_count_rollout.get("current_production_profile")
                or draw_count_rollout.get("operational_policy"),
                "draw_count": draw_count_rollout.get("chosen_production_draw_count")
                or draw_count_rollout.get("current_production_draw_count"),
                "phase_2_status": draw_count_rollout.get("phase_2_status"),
                "strict_gate_promotion": False,
            }

    tail_rates = {}
    scored_path = model_v3 / "holdout_scored_top120_2025.parquet"
    if scored_path.exists():
        tail_rates = _tail_miss_rates(pd.read_parquet(scored_path))

    tail_diag_path = model_v3 / f"player_stability_diagnostics_{season}.parquet"
    tail_instability_core_rows = 0
    if tail_diag_path.exists():
        tail_df = pd.read_parquet(tail_diag_path)
        if "core_player_flag" in tail_df.columns and "tail_instability_flag" in tail_df.columns:
            tail_instability_core_rows = int(
                tail_df[
                    (tail_df["core_player_flag"] == True)  # noqa: E712
                    & (tail_df["tail_instability_flag"] == True)  # noqa: E712
                ]["player_id"]
                .nunique()
            )

    core_player_tail_stability = {
        "status": "monitor",
        "affected_output": "mostly p90 / upper-tail range",
        "decision_threshold_impact": "none",
        "tail_instability_core_rows": tail_instability_core_rows,
        "required_action": (
            "re-evaluate at selected production draw count and at next model refresh"
        ),
        "source": str(tail_diag_path) if tail_diag_path.exists() else None,
    }

    intermediate_sweep = None
    if draw_count_decision and draw_count_decision.get("schema_version") == "draw_count_decision_v2":
        intermediate_sweep = {
            "sweep_phase": draw_count_decision.get("sweep_phase"),
            "reference_draws": draw_count_decision.get("reference_draws"),
            "selected_draw_count": draw_count_decision.get("selected_draw_count"),
            "production_recommendation": draw_count_decision.get("production_recommendation"),
            "policy_matrix_outcome": draw_count_decision.get("policy_matrix_outcome"),
            "candidates_evaluated": draw_count_decision.get("candidates_evaluated"),
            "provenance_verdict": draw_count_decision.get("provenance_verdict"),
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "simulation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "provenance": {
            "projection_run_id": (projection_run or {}).get("run_id"),
            "canonical_projection_run_id": (simulation_manifest or {}).get(
                "canonical_projection_run_id"
            ),
            "simulation_run_id": (simulation_manifest or {}).get("source_projection_run_id"),
            "selected_board_hash": (simulation_manifest or {}).get("selected_board_hash"),
            "selected_board_model_id": (simulation_manifest or {}).get("selected_board_model_id"),
            "deterministic_board_version": (projection_run or {}).get("composition_version"),
        },
        "simulation": {
            "draw_count": (simulation_manifest or {}).get("draw_count"),
            "deterministic_seed": (simulation_manifest or {}).get("deterministic_seed"),
            "transform_version": (simulation_manifest or {}).get("transform_version"),
            "wr_calibration_version": (simulation_manifest or {}).get("wr_calibration_version"),
            "wr_residual_scale": (simulation_manifest or {}).get("wr_residual_scale"),
            "wr_calibration_artifact_hash": (simulation_manifest or {}).get(
                "wr_calibration_artifact_hash"
            ),
            "partition_hashes": (simulation_manifest or {}).get("partition_hashes"),
            **({"draw_count_policy": draw_count_policy} if draw_count_policy else {}),
            **({"summary_risks": list(risks)} if draw_count_policy else {}),
        },
        "gates": {
            "finish_probability": finish_gate,
            "promotion": promotion_gate,
            "draw_stability_recommended": (draw_stability or {}).get("recommended_draw_count"),
            "draw_count_decision": (draw_count_decision or {}).get("selected_draw_count"),
        },
        "quality_diagnostics": {
            "segment_summary": segment_summary,
            "holdout_acceptance": (holdout or {}).get("acceptance"),
            "tail_miss_rates": tail_rates,
            "wr_calibration": wr_calibration,
            "draw_stability_summary": {
                "recommended_draw_count": (draw_stability or {}).get("recommended_draw_count"),
                "reference_draws": (draw_stability or {}).get("reference_draws"),
            },
            "decision_change_diagnostics": {
                "path": str(model_v3 / f"decision_change_diagnostics_{season}.json"),
                "verdict": (decision_change_diagnostics or {}).get("verdict"),
                "reason": (decision_change_diagnostics or {}).get("reason"),
                "category_summary": (decision_change_diagnostics or {}).get(
                    "changes_by_category"
                ),
                "core_player_events_requiring_review": (
                    decision_change_diagnostics or {}
                ).get("core_player_events_requiring_review"),
                "prominent_risk": bool(
                    decision_change_diagnostics
                    and (
                        decision_change_diagnostics.get("verdict") == "hold"
                        or (decision_change_diagnostics.get("changes_by_category") or {}).get(
                            "material"
                        )
                        or (
                            decision_change_diagnostics.get("changes_by_category") or {}
                        ).get("reference_instability")
                    )
                ),
            },
            "core_player_tail_stability": core_player_tail_stability,
            "intermediate_draw_sweep": intermediate_sweep,
        },
        "summary_risks": risks,
    }


def build_release_report_board(
    *,
    season: int,
    draft_value_meta: dict | None,
    v3_sim_meta: dict | None,
    exported_board_path: Path,
    players_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    finish_cols = [c for c in players_df.columns if c.startswith("p_finish_top")] if players_df is not None else []
    vorp_cols = [
        c
        for c in (players_df.columns if players_df is not None else [])
        if c.startswith("sim_vorp_") or c in {"p_vorp_positive", "expected_pos_rank", "median_pos_rank"}
    ]
    finish_attached = int(players_df[finish_cols].notna().any(axis=1).sum()) if finish_cols else 0
    vorp_attached = int(players_df[vorp_cols].notna().any(axis=1).sum()) if vorp_cols else 0

    risks: list[str] = []
    if draft_value_meta:
        if not draft_value_meta.get("finish_probabilities", {}).get("attached"):
            risks.append("p_finish_*: not_attached")
        if not draft_value_meta.get("simulated_vorp", {}).get("attached"):
            risks.append("sim_vorp_*: not_attached")
    else:
        risks.append("draft_value_overlay: metadata_missing")

    board_hash = sha256_file(str(exported_board_path)) if exported_board_path.exists() else None
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "board",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "exported_board_hash": board_hash,
        "exported_board_path": str(exported_board_path),
        "overlay_attachment": {
            "finish_probabilities": draft_value_meta.get("finish_probabilities") if draft_value_meta else None,
            "simulated_vorp": draft_value_meta.get("simulated_vorp") if draft_value_meta else None,
            "v3_percentiles": v3_sim_meta,
            "players_with_finish_probs": finish_attached,
            "players_with_sim_vorp": vorp_attached,
        },
        "summary_risks": risks,
    }


def merge_release_reports(
    simulation_report: dict[str, Any],
    board_report: dict[str, Any] | None,
) -> dict[str, Any]:
    risks = list(simulation_report.get("summary_risks") or [])
    if board_report is None:
        risks.append("board_report: missing")
    else:
        risks.extend(board_report.get("summary_risks") or [])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "merged",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": simulation_report.get("season"),
        "simulation": simulation_report,
        "board": board_report,
        "summary_risks": sorted(set(risks)),
    }


def write_release_report_simulation(
    report: dict[str, Any], *, season: int, out_dir: Path | None = None
) -> Path:
    base = out_dir or Path(MODEL_V3_DIR)
    path = base / f"release_report_simulation_{season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_release_report_board(
    report: dict[str, Any], *, season: int, out_dir: Path | None = None
) -> Path:
    base = out_dir or Path(MODEL_V3_DIR)
    path = base / f"release_report_board_{season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_merged_release_report(
    report: dict[str, Any], *, season: int, out_dir: Path | None = None
) -> Path:
    base = out_dir or Path(MODEL_V3_DIR)
    path = base / f"release_report_{season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
