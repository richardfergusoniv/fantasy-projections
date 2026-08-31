"""Decision-change diagnostics with 20k reference validation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import TOP_ADP
from src.projection.evaluation.draw_stability import (
    ACTIVE_DECISION_METRIC_KEYS,
    STABILITY_METRICS,
    enumerate_decision_changes,
    filter_draw_prefix,
    generate_recentered_reference_draws,
    prefix_is_nested_subset,
    qualifies,
    resolve_decision_threshold_registry,
    signed_distance_from_threshold,
    stability_contract_hashes,
    summarize_candidate_metrics,
)
from src.projection.inference.recenter import TRANSFORM_VERSION, board_points_series
from src.projection.inference.simulate import slim_draw_frame
from src.projection.inference.simulation_config import (
    load_simulation_config,
    stability_simulation_seed,
)

DIAGNOSTIC_SCHEMA_VERSION = "decision_change_diagnostics_v1"

PLAYER_DIAGNOSTIC_METRICS = (
    "p10",
    "p50",
    "p90",
    "sim_vorp_p50",
    "expected_pos_rank",
    "p_finish_top6",
    "p_finish_top12",
    "p_finish_top24",
    "p_finish_top36",
    "p_finish_top48",
    "p_vorp_positive",
)

SUPPORTING_METRIC_COLUMNS = (
    "p10",
    "p50",
    "p90",
    "sim_vorp_p50",
    "p_finish_top6",
    "p_finish_top48",
)


def load_decision_diagnostic_config(config: dict | None = None) -> dict[str, Any]:
    sim_config = config or load_simulation_config()
    defaults = {
        "primary_reference_draws": 10000,
        "diagnostic_reference_draws": 20000,
        "candidate_draw_counts": [1000, 2000, 5000],
        "boundary_margin_probability": 0.02,
        "boundary_margin_rank": 0.25,
        "reference_instability_probability": 0.015,
        "reference_instability_rank": 0.25,
        "tail_p50_stable_threshold": 0.25,
        "tail_finish_probability_threshold": 0.015,
        "tail_p10_p90_threshold": 1.0,
        "core_adp_threshold": 36,
        "core_player_max_difference_requires_review": True,
        "core_player_decision_change_requires_review": True,
    }
    return {**defaults, **(sim_config.get("decision_diagnostic") or {})}


def slim_draw_frame_hash(draws: pd.DataFrame) -> str:
    slim = slim_draw_frame(draws).sort_values(["draw", "player_id"]).reset_index(drop=True)
    payload = slim.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_draw_id_uniqueness(draws: pd.DataFrame, *, max_draw_id: int) -> dict[str, Any]:
    prefix = filter_draw_prefix(draws, max_draw_id)
    if prefix.empty:
        return {"passes": False, "reason": "empty_prefix"}
    draw_ids = prefix["draw"].astype(int)
    expected = set(range(max_draw_id))
    actual = set(draw_ids.unique().tolist())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    counts = draw_ids.value_counts()
    per_draw_counts = counts.value_counts()
    uniform_rows_per_draw = len(per_draw_counts) == 1
    rows_per_draw = int(per_draw_counts.index[0]) if uniform_rows_per_draw else None
    passes = not missing and not extra and uniform_rows_per_draw
    return {
        "passes": passes,
        "missing_draw_ids": missing[:10],
        "extra_draw_ids": extra[:10],
        "rows_per_draw": rows_per_draw,
        "n_draw_ids": int(len(actual)),
        "n_rows": int(len(prefix)),
    }


def load_raw_draws_from_checkpoint(checkpoint_dir: Path) -> pd.DataFrame:
    parts = sorted(Path(checkpoint_dir).glob("raw_part_*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in parts]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_recentered_draws_from_checkpoint(
    checkpoint_dir: Path,
    selected_board: pd.DataFrame,
    *,
    max_draws: int,
) -> pd.DataFrame:
    from src.projection.inference.wr_calibration import load_wr_calibration, recenter_draws_wr_scaled

    raw = load_raw_draws_from_checkpoint(checkpoint_dir)
    if raw.empty:
        return pd.DataFrame()
    raw_prefix = filter_draw_prefix(raw, max_draws)
    wr_calibration = load_wr_calibration()
    wr_scale = float((wr_calibration or {}).get("selected_wr_scale", 1.0))
    selected_points = board_points_series(selected_board)
    return recenter_draws_wr_scaled(raw_prefix, selected_points, wr_scale=wr_scale)


def verify_nested_prefix_provenance(
    *,
    season: int,
    primary_reference_draws: int,
    diagnostic_reference_draws: int,
    primary_raw: pd.DataFrame,
    diagnostic_raw: pd.DataFrame,
    contract_hashes: dict[str, str],
    simulation_seed: int,
    checkpoint_meta: dict[str, Any] | None,
    selected_board_model_id: str,
    canonical_projection_run_id: str,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failures: list[str] = []

    checks["season"] = {"expected": season, "actual": season, "passes": True}
    if checkpoint_meta is not None:
        expected_seed = int(simulation_seed)
        actual_seed = int(checkpoint_meta.get("seed", -1))
        seed_passes = actual_seed == expected_seed
        checks["simulation_seed"] = {
            "expected": expected_seed,
            "actual": actual_seed,
            "passes": seed_passes,
        }
        if not seed_passes:
            failures.append("simulation_seed_mismatch")

    checks["selected_board_model_id"] = {
        "expected": selected_board_model_id,
        "actual": selected_board_model_id,
        "passes": True,
    }
    checks["canonical_projection_run_id"] = {
        "expected": canonical_projection_run_id,
        "actual": canonical_projection_run_id,
        "passes": True,
    }
    checks["transform_version"] = {
        "expected": TRANSFORM_VERSION,
        "actual": TRANSFORM_VERSION,
        "passes": True,
    }
    for key, expected in contract_hashes.items():
        checks[key] = {"expected": expected, "actual": expected, "passes": True}

    primary_raw_prefix = filter_draw_prefix(primary_raw, primary_reference_draws)
    diagnostic_raw_prefix = filter_draw_prefix(diagnostic_raw, primary_reference_draws)

    draw_uniqueness = verify_draw_id_uniqueness(
        diagnostic_raw,
        max_draw_id=primary_reference_draws,
    )
    checks["draw_ids_0_to_primary_minus_one"] = draw_uniqueness
    if not draw_uniqueness["passes"]:
        failures.append("draw_id_uniqueness_failed")

    nested_subset = prefix_is_nested_subset(
        primary_raw_prefix,
        diagnostic_raw,
        max_draw_id=primary_reference_draws,
    )
    checks["nested_prefix_subset"] = {"passes": nested_subset, "scope": "raw_draws"}
    if not nested_subset:
        failures.append("nested_prefix_subset_failed")

    primary_hash = slim_draw_frame_hash(primary_raw_prefix)
    diagnostic_prefix_hash = slim_draw_frame_hash(diagnostic_raw_prefix)
    hash_passes = primary_hash == diagnostic_prefix_hash
    checks["prefix_hash_match"] = {
        "primary_hash": primary_hash,
        "diagnostic_prefix_hash": diagnostic_prefix_hash,
        "passes": hash_passes,
        "scope": "raw_draws",
    }
    if not hash_passes:
        failures.append("prefix_hash_mismatch")

    return {
        "passes": not failures,
        "failures": failures,
        "checks": checks,
        "primary_reference_draws": primary_reference_draws,
        "diagnostic_reference_draws": diagnostic_reference_draws,
    }


def _reference_instability_tolerance(metric_kind: str, config: dict[str, Any]) -> float:
    if metric_kind == "rank":
        return float(config["reference_instability_rank"])
    return float(config["reference_instability_probability"])


def _boundary_margin(metric_kind: str, config: dict[str, Any]) -> float:
    if metric_kind == "rank":
        return float(config["boundary_margin_rank"])
    return float(config["boundary_margin_probability"])


def classify_decision_event(row: dict[str, Any], *, config: dict[str, Any]) -> str:
    metric_kind = str(row["metric_kind"])
    if row["reference_10k_vs_20k_crossing_disagrees"]:
        return "reference_instability"
    if row["reference_10k_to_20k_abs_diff"] > _reference_instability_tolerance(
        metric_kind,
        config,
    ):
        return "reference_instability"

    margin = _boundary_margin(metric_kind, config)
    if abs(row["distance_from_threshold_10k"]) <= margin and not row["change_survives_at_20k"]:
        return "boundary_noise"

    if metric_kind == "rank":
        return "rank_instability"

    return "material"


def _review_required(row: dict[str, Any], *, config: dict[str, Any]) -> bool:
    if not row["is_core_adp_player"]:
        return False
    category = row["category"]
    if category == "material" and config.get("core_player_decision_change_requires_review", True):
        return True
    if category == "boundary_noise" and config.get("core_player_decision_change_requires_review", True):
        return True
    if category == "reference_instability":
        return True
    return False


def _player_metadata(
    selected_board: pd.DataFrame,
    projections: pd.DataFrame,
    *,
    core_adp_threshold: int,
) -> pd.DataFrame:
    board = selected_board.copy()
    board["player_id"] = board["player_id"].astype(str)
    adp_col = "adp" if "adp" in board.columns else None
    if adp_col:
        board["adp"] = pd.to_numeric(board[adp_col], errors="coerce")
    else:
        board["adp"] = np.nan

    names = (
        projections[["player_id", "display_name"]]
        .drop_duplicates("player_id")
        .assign(player_id=lambda df: df["player_id"].astype(str))
    )
    points = board_points_series(board).rename("selected_fantasy_point_forecast")
    meta = board.set_index("player_id")[["position", "adp"]].join(points, how="left")
    meta = meta.join(names.set_index("player_id")[["display_name"]], how="left")
    meta = meta.reset_index()
    meta["player_name"] = meta["display_name"].fillna(meta["player_id"])
    meta["is_core_adp_player"] = meta["adp"].notna() & (meta["adp"] <= core_adp_threshold)
    return meta


def _metric_lookup(metrics: pd.DataFrame, player_id: str, metric: str) -> float | None:
    row = metrics.loc[metrics["player_id"] == player_id]
    if row.empty or metric not in row.columns:
        return None
    value = row.iloc[0][metric]
    if pd.isna(value):
        return None
    return float(value)


def build_decision_change_event_rows(
    *,
    season: int,
    selected_board_hash: str,
    simulation_seed: int,
    candidate_draw_count: int,
    primary_reference_draws: int,
    diagnostic_reference_draws: int,
    candidate_metrics: pd.DataFrame,
    reference_10k_metrics: pd.DataFrame,
    reference_20k_metrics: pd.DataFrame,
    player_meta: pd.DataFrame,
    threshold_registry: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    base_events = enumerate_decision_changes(
        candidate_metrics,
        reference_10k_metrics,
        threshold_registry=threshold_registry,
    )
    meta_by_player = player_meta.set_index("player_id")
    rows: list[dict[str, Any]] = []
    for event in base_events:
        player_id = event["player_id"]
        metric = event["metric"]
        spec = threshold_registry[metric]
        threshold = float(spec["threshold"])
        direction = str(spec["direction"])
        candidate_value = float(event["candidate_value"])
        reference_10k_value = float(event["reference_value"])
        reference_20k_value = _metric_lookup(reference_20k_metrics, player_id, metric)
        if reference_20k_value is None:
            continue

        candidate_qualifies = qualifies(candidate_value, threshold, direction)
        reference_10k_qualifies = qualifies(reference_10k_value, threshold, direction)
        reference_20k_qualifies = qualifies(reference_20k_value, threshold, direction)
        distance_10k = signed_distance_from_threshold(reference_10k_value, threshold, direction)
        distance_20k = signed_distance_from_threshold(reference_20k_value, threshold, direction)
        reference_10k_to_20k_abs_diff = abs(reference_10k_value - reference_20k_value)
        reference_10k_to_20k_signed_distance_change = distance_20k - distance_10k

        meta = meta_by_player.loc[player_id] if player_id in meta_by_player.index else None
        supporting = {
            col: _metric_lookup(reference_10k_metrics, player_id, col)
            for col in SUPPORTING_METRIC_COLUMNS
        }
        row = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "season": season,
            "selected_board_hash": selected_board_hash,
            "simulation_seed": simulation_seed,
            "candidate_draw_count": candidate_draw_count,
            "primary_reference_draw_count": primary_reference_draws,
            "diagnostic_reference_draw_count": diagnostic_reference_draws,
            "player_id": player_id,
            "player_name": None if meta is None else meta.get("player_name", player_id),
            "position": None if meta is None else meta.get("position"),
            "adp": None if meta is None else (
                float(meta["adp"]) if pd.notna(meta.get("adp")) else None
            ),
            "is_core_adp_player": False if meta is None else bool(meta.get("is_core_adp_player")),
            "selected_fantasy_point_forecast": None if meta is None else (
                float(meta["selected_fantasy_point_forecast"])
                if pd.notna(meta.get("selected_fantasy_point_forecast"))
                else None
            ),
            "metric": metric,
            "metric_kind": event["metric_kind"],
            "threshold": threshold,
            "threshold_direction": direction,
            "candidate_value": candidate_value,
            "reference_10k_value": reference_10k_value,
            "reference_20k_value": reference_20k_value,
            "candidate_qualifies": candidate_qualifies,
            "reference_10k_qualifies": reference_10k_qualifies,
            "reference_20k_qualifies": reference_20k_qualifies,
            "abs_diff_vs_10k": float(event["abs_diff"]),
            "distance_from_threshold_10k": distance_10k,
            "distance_from_threshold_20k": distance_20k,
            "reference_10k_to_20k_abs_diff": reference_10k_to_20k_abs_diff,
            "reference_10k_to_20k_signed_distance_change": reference_10k_to_20k_signed_distance_change,
            "change_survives_at_20k": candidate_qualifies != reference_20k_qualifies,
            "reference_10k_vs_20k_crossing_disagrees": (
                reference_10k_qualifies != reference_20k_qualifies
            ),
            "p10_reference_10k": supporting["p10"],
            "p50_reference_10k": supporting["p50"],
            "p90_reference_10k": supporting["p90"],
            "sim_vorp_p50_reference_10k": supporting["sim_vorp_p50"],
            "p_finish_top6_reference_10k": supporting["p_finish_top6"],
            "p_finish_top48_reference_10k": supporting["p_finish_top48"],
        }
        row["category"] = classify_decision_event(row, config=config)
        row["review_required"] = _review_required(row, config=config)
        rows.append(row)
    return rows


def _tail_instability_flag(
    diffs: dict[str, float],
    *,
    config: dict[str, Any],
) -> bool:
    p50_diff = diffs.get("p50", 0.0)
    if p50_diff > float(config["tail_p50_stable_threshold"]):
        return False
    tail_checks = [
        diffs.get("p10", 0.0) > float(config["tail_p10_p90_threshold"]),
        diffs.get("p90", 0.0) > float(config["tail_p10_p90_threshold"]),
        diffs.get("p_finish_top6", 0.0) > float(config["tail_finish_probability_threshold"]),
        diffs.get("p_finish_top48", 0.0) > float(config["tail_finish_probability_threshold"]),
    ]
    return any(tail_checks)


def build_player_stability_diagnostic_rows(
    *,
    candidate_draw_count: int,
    candidate_metrics: pd.DataFrame,
    reference_10k_metrics: pd.DataFrame,
    player_meta: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    merged = candidate_metrics.merge(
        reference_10k_metrics,
        on="player_id",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    rows: list[dict[str, Any]] = []
    meta_by_player = player_meta.set_index("player_id")
    for _, row in merged.iterrows():
        player_id = str(row["player_id"])
        diffs: dict[str, float] = {}
        for metric in PLAYER_DIAGNOSTIC_METRICS:
            cand_col = f"{metric}_candidate"
            ref_col = f"{metric}_reference"
            if cand_col in row and ref_col in row and pd.notna(row[cand_col]) and pd.notna(row[ref_col]):
                diffs[metric] = abs(float(row[cand_col]) - float(row[ref_col]))

        worst_metric = None
        worst_metric_abs_diff = 0.0
        if diffs:
            worst_metric = max(diffs, key=diffs.get)
            worst_metric_abs_diff = float(diffs[worst_metric])

        meta = meta_by_player.loc[player_id] if player_id in meta_by_player.index else None
        rows.append(
            {
                "candidate_draw_count": candidate_draw_count,
                "player_id": player_id,
                "position": None if meta is None else meta.get("position"),
                "adp": None if meta is None else (
                    float(meta["adp"]) if pd.notna(meta.get("adp")) else None
                ),
                "selected_fantasy_point_forecast": None if meta is None else (
                    float(meta["selected_fantasy_point_forecast"])
                    if pd.notna(meta.get("selected_fantasy_point_forecast"))
                    else None
                ),
                "p10_abs_diff_vs_10k": diffs.get("p10"),
                "p50_abs_diff_vs_10k": diffs.get("p50"),
                "p90_abs_diff_vs_10k": diffs.get("p90"),
                "sim_vorp_p50_abs_diff_vs_10k": diffs.get("sim_vorp_p50"),
                "expected_pos_rank_abs_diff_vs_10k": diffs.get("expected_pos_rank"),
                "p_finish_top6_abs_diff_vs_10k": diffs.get("p_finish_top6"),
                "p_finish_top12_abs_diff_vs_10k": diffs.get("p_finish_top12"),
                "p_finish_top24_abs_diff_vs_10k": diffs.get("p_finish_top24"),
                "p_finish_top36_abs_diff_vs_10k": diffs.get("p_finish_top36"),
                "p_finish_top48_abs_diff_vs_10k": diffs.get("p_finish_top48"),
                "p_vorp_positive_abs_diff_vs_10k": diffs.get("p_vorp_positive"),
                "tail_instability_flag": _tail_instability_flag(diffs, config=config),
                "core_player_flag": False if meta is None else bool(meta.get("is_core_adp_player")),
                "worst_metric": worst_metric,
                "worst_metric_abs_diff": worst_metric_abs_diff,
            }
        )
    return rows


def build_decision_change_diagnostic_report(
    *,
    season: int,
    recentered_primary: pd.DataFrame | None,
    recentered_diagnostic: pd.DataFrame | None,
    selected_board: pd.DataFrame,
    projections: pd.DataFrame,
    selected_board_hash: str,
    wr_calibration_hash: str,
    canonical_projection_run_id: str,
    selected_board_model_id: str = "accuracy_first_ensemble",
    calibration_hash: str,
    checkpoint_meta: dict[str, Any] | None,
    checkpoint_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    top_adp: int = TOP_ADP,
) -> dict[str, Any]:
    config = load_decision_diagnostic_config(config)
    primary_reference_draws = int(config["primary_reference_draws"])
    diagnostic_reference_draws = int(config["diagnostic_reference_draws"])
    candidate_draw_counts = [int(x) for x in config["candidate_draw_counts"]]

    contract_hashes = stability_contract_hashes(
        season=season,
        selected_board_hash=selected_board_hash,
        selected_board=selected_board,
        wr_calibration_hash=wr_calibration_hash,
        canonical_projection_run_id=canonical_projection_run_id,
        selected_board_model_id=selected_board_model_id,
    )
    sim_config = load_simulation_config()
    configured_seed = int(sim_config.get("random_seed") or 2026)
    simulation_seed = stability_simulation_seed(
        season=season,
        board_hash=selected_board_hash,
        calibration_hash=calibration_hash,
        configured_seed=configured_seed,
        canonical_projection_run_id=canonical_projection_run_id,
        transform_version=TRANSFORM_VERSION,
        wr_calibration_hash=wr_calibration_hash,
    )
    threshold_registry = resolve_decision_threshold_registry(
        sim_config.get("decision_thresholds") or {},
    )

    report: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "verdict": "hold",
        "reason": "diagnostic_reference_missing",
        "provenance_checks": None,
        "unique_players_affected": 0,
        "changes_by_candidate": {},
        "changes_by_metric": {},
        "changes_by_category": {},
        "core_player_events_requiring_review": 0,
        "resolution_hints": {
            "option_1_display_rules": False,
            "option_2_increase_draws": False,
            "option_3_variance_reduction": False,
        },
        "artifact_paths": {},
        "event_rows": [],
        "player_diagnostic_rows": [],
    }

    if checkpoint_dir is None and recentered_diagnostic is None:
        return report

    if checkpoint_dir is not None:
        recentered_primary = build_recentered_draws_from_checkpoint(
            checkpoint_dir,
            selected_board,
            max_draws=primary_reference_draws,
        )
        recentered_diagnostic = build_recentered_draws_from_checkpoint(
            checkpoint_dir,
            selected_board,
            max_draws=diagnostic_reference_draws,
        )
        primary_raw = filter_draw_prefix(
            load_raw_draws_from_checkpoint(checkpoint_dir),
            primary_reference_draws,
        )
        diagnostic_raw = load_raw_draws_from_checkpoint(checkpoint_dir)
    else:
        primary_raw = filter_draw_prefix(recentered_primary, primary_reference_draws)
        diagnostic_raw = recentered_diagnostic

    provenance = verify_nested_prefix_provenance(
        season=season,
        primary_reference_draws=primary_reference_draws,
        diagnostic_reference_draws=diagnostic_reference_draws,
        primary_raw=primary_raw,
        diagnostic_raw=diagnostic_raw,
        contract_hashes=contract_hashes,
        simulation_seed=simulation_seed,
        checkpoint_meta=checkpoint_meta,
        selected_board_model_id=selected_board_model_id,
        canonical_projection_run_id=canonical_projection_run_id,
    )
    report["provenance_checks"] = provenance
    if not provenance["passes"]:
        report["reason"] = "nested_prefix_invariant_failed"
        return report

    adp_col = "adp" if "adp" in selected_board.columns else None
    top_players = selected_board.copy()
    if adp_col:
        top_players = top_players[
            pd.to_numeric(top_players[adp_col], errors="coerce") <= top_adp
        ]
    top_ids = set(top_players["player_id"].astype(str))
    player_meta = _player_metadata(
        selected_board,
        projections,
        core_adp_threshold=int(config["core_adp_threshold"]),
    )
    player_meta = player_meta[player_meta["player_id"].isin(top_ids)].copy()

    reference_10k_metrics = summarize_candidate_metrics(
        recentered_primary,
        board=selected_board,
        max_draws=primary_reference_draws,
    )
    reference_10k_metrics = reference_10k_metrics[
        reference_10k_metrics["player_id"].isin(top_ids)
    ].copy()
    reference_20k_metrics = summarize_candidate_metrics(
        recentered_diagnostic,
        board=selected_board,
        max_draws=diagnostic_reference_draws,
    )
    reference_20k_metrics = reference_20k_metrics[
        reference_20k_metrics["player_id"].isin(top_ids)
    ].copy()

    event_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for draw_count in candidate_draw_counts:
        candidate_metrics = summarize_candidate_metrics(
            recentered_primary,
            board=selected_board,
            max_draws=draw_count,
        )
        candidate_metrics = candidate_metrics[candidate_metrics["player_id"].isin(top_ids)].copy()
        event_rows.extend(
            build_decision_change_event_rows(
                season=season,
                selected_board_hash=selected_board_hash,
                simulation_seed=simulation_seed,
                candidate_draw_count=draw_count,
                primary_reference_draws=primary_reference_draws,
                diagnostic_reference_draws=diagnostic_reference_draws,
                candidate_metrics=candidate_metrics,
                reference_10k_metrics=reference_10k_metrics,
                reference_20k_metrics=reference_20k_metrics,
                player_meta=player_meta,
                threshold_registry=threshold_registry,
                config=config,
            )
        )
        player_rows.extend(
            build_player_stability_diagnostic_rows(
                candidate_draw_count=draw_count,
                candidate_metrics=candidate_metrics,
                reference_10k_metrics=reference_10k_metrics,
                player_meta=player_meta,
                config=config,
            )
        )

    changes_by_candidate: dict[str, int] = {}
    changes_by_metric: dict[str, int] = {}
    changes_by_category: dict[str, int] = {}
    for row in event_rows:
        cand_key = str(row["candidate_draw_count"])
        changes_by_candidate[cand_key] = changes_by_candidate.get(cand_key, 0) + 1
        changes_by_metric[row["metric"]] = changes_by_metric.get(row["metric"], 0) + 1
        changes_by_category[row["category"]] = changes_by_category.get(row["category"], 0) + 1

    unique_players = len({row["player_id"] for row in event_rows})
    core_reviews = sum(1 for row in event_rows if row.get("review_required"))

    material_count = changes_by_category.get("material", 0)
    boundary_count = changes_by_category.get("boundary_noise", 0)
    reference_instability_count = changes_by_category.get("reference_instability", 0)

    report.update(
        {
            "verdict": "ok",
            "reason": None,
            "unique_players_affected": unique_players,
            "changes_by_candidate": changes_by_candidate,
            "changes_by_metric": changes_by_metric,
            "changes_by_category": changes_by_category,
            "core_player_events_requiring_review": core_reviews,
            "resolution_hints": {
                "option_1_display_rules": (
                    material_count == 0
                    and reference_instability_count == 0
                    and boundary_count > 0
                ),
                "option_2_increase_draws": (
                    material_count > 0 and reference_instability_count == 0
                ),
                "option_3_variance_reduction": (
                    material_count > 0 and reference_instability_count == 0
                ),
            },
            "event_rows": event_rows,
            "player_diagnostic_rows": player_rows,
            "active_decision_metrics": list(ACTIVE_DECISION_METRIC_KEYS),
        }
    )
    return report


def write_decision_change_diagnostics_artifacts(
    report: dict[str, Any],
    *,
    season: int,
) -> dict[str, str]:
    out_dir = Path(MODEL_V3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_rows = report.pop("event_rows", [])
    player_rows = report.pop("player_diagnostic_rows", [])

    events_path = out_dir / f"decision_change_events_{season}.parquet"
    player_path = out_dir / f"player_stability_diagnostics_{season}.parquet"
    summary_path = out_dir / f"decision_change_diagnostics_{season}.json"

    if event_rows:
        pd.DataFrame(event_rows).to_parquet(events_path, index=False)
    else:
        pd.DataFrame(columns=["schema_version"]).to_parquet(events_path, index=False)

    if player_rows:
        pd.DataFrame(player_rows).to_parquet(player_path, index=False)
    else:
        pd.DataFrame(columns=["candidate_draw_count", "player_id"]).to_parquet(
            player_path,
            index=False,
        )

    report["artifact_paths"] = {
        "events_parquet": str(events_path),
        "player_diagnostics_parquet": str(player_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["artifact_paths"]


def extend_diagnostic_reference_draws(
    projections: pd.DataFrame,
    *,
    season: int,
    selected_board: pd.DataFrame,
    selected_board_hash: str,
    canonical_projection_run_id: str,
    calibration_hash: str,
    wr_calibration_hash: str,
    primary_reference_draws: int,
    diagnostic_reference_draws: int,
    checkpoint_dir: Path,
    uncertainty_manifest: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    checkpoint_meta = None
    if meta_path.exists():
        checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    recentered = generate_recentered_reference_draws(
        projections,
        season=season,
        selected_board=selected_board,
        selected_board_hash=selected_board_hash,
        reference_draws=diagnostic_reference_draws,
        canonical_projection_run_id=canonical_projection_run_id,
        calibration_hash=calibration_hash,
        wr_calibration_hash=wr_calibration_hash,
        uncertainty_manifest=uncertainty_manifest,
        checkpoint_dir=checkpoint_dir,
    )
    if meta_path.exists():
        checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return recentered, checkpoint_meta
