"""Nested-prefix draw-count stability evaluation for simulation overlays."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.draft_assistant.draft_value_simulation import (
    FINISH_CUTOFFS,
    compute_finish_probabilities,
    compute_simulated_vorp_metrics,
)
from src.draft_assistant.replacement_contract import (
    build_replacement_contract,
    load_roster_configuration,
    roster_configuration_hash,
)
from src.projection.contracts import MODEL_V3_DIR, OUTPUT_DIR
from src.projection.evaluation.accuracy_first import TOP_ADP, sha256_file
from src.projection.evaluation.finish_probability_gate import FINISH_GATE_PATH
from src.projection.inference.recenter import TRANSFORM_VERSION, board_points_series
from src.projection.inference.simulate import (
    simulate_season_draw_range,
    slim_draw_frame,
    summarize_simulations,
)
from src.projection.inference.simulation_config import (
    load_simulation_config,
    stability_simulation_seed,
)
from src.projection.inference.wr_calibration import (
    ARTIFACT_PATH as WR_CALIBRATION_PATH,
    load_wr_calibration,
    recenter_draws_wr_scaled,
)

STABILITY_METRICS = (
    "p10",
    "p50",
    "p90",
    "p_finish_top6",
    "p_finish_top12",
    "p_finish_top24",
    "p_finish_top36",
    "p_finish_top48",
    "sim_vorp_p10",
    "sim_vorp_p50",
    "sim_vorp_p90",
    "p_vorp_positive",
    "expected_pos_rank",
    "median_pos_rank",
)

METRIC_TOLERANCE_KEYS = {
    "p50": "p50_abs",
    "p10": "p10_p90_abs",
    "p90": "p10_p90_abs",
    "sim_vorp_p50": "sim_vorp_p50_abs",
    "p_finish_top6": "probability_abs",
    "p_finish_top12": "probability_abs",
    "p_finish_top24": "probability_abs",
    "p_finish_top36": "probability_abs",
    "p_finish_top48": "probability_abs",
    "p_vorp_positive": "probability_abs",
    "expected_pos_rank": "expected_pos_rank_abs",
    "median_pos_rank": "expected_pos_rank_abs",
    "sim_vorp_p10": "p10_p90_abs",
    "sim_vorp_p90": "p10_p90_abs",
}

DECISION_THRESHOLDS: dict[str, dict[str, Any]] = {
    "p_finish_top6": {
        "threshold": 0.50,
        "direction": "ge",
        "kind": "probability",
    },
    "p_finish_top12": {
        "threshold": 0.50,
        "direction": "ge",
        "kind": "probability",
    },
    "p_finish_top24": {
        "threshold": 0.50,
        "direction": "ge",
        "kind": "probability",
    },
    "p_vorp_positive": {
        "threshold": 0.50,
        "direction": "ge",
        "kind": "probability",
    },
    "expected_pos_rank_top12": {
        "threshold": 12.0,
        "direction": "le",
        "kind": "rank",
    },
}

ACTIVE_DECISION_METRIC_KEYS = (
    "p_finish_top12",
    "p_finish_top24",
    "p_vorp_positive",
)

DECISION_METRICS = ACTIVE_DECISION_METRIC_KEYS


def qualifies(value: float, threshold: float, direction: str) -> bool:
    if direction == "ge":
        return float(value) >= float(threshold)
    if direction == "le":
        return float(value) <= float(threshold)
    raise ValueError(f"Unknown threshold direction: {direction}")


def signed_distance_from_threshold(
    value: float,
    threshold: float,
    direction: str,
) -> float:
    """Positive distance means the value qualifies; negative means it does not."""
    if direction == "ge":
        return float(value) - float(threshold)
    if direction == "le":
        return float(threshold) - float(value)
    raise ValueError(f"Unknown threshold direction: {direction}")


def resolve_decision_threshold_registry(
    config_overrides: dict[str, float] | None = None,
    *,
    active_keys: tuple[str, ...] = ACTIVE_DECISION_METRIC_KEYS,
) -> dict[str, dict[str, Any]]:
    registry = {key: dict(spec) for key, spec in DECISION_THRESHOLDS.items()}
    for key, value in (config_overrides or {}).items():
        if key in registry:
            registry[key]["threshold"] = float(value)
    return {key: registry[key] for key in active_keys if key in registry}


def filter_draw_prefix(draws: pd.DataFrame, max_draws: int) -> pd.DataFrame:
    """Return only draw IDs in [0, max_draws)."""
    if draws.empty:
        return draws.copy()
    return draws[draws["draw"] < int(max_draws)].copy()


def prefix_is_nested_subset(
    smaller: pd.DataFrame,
    larger: pd.DataFrame,
    *,
    max_draw_id: int,
) -> bool:
    """True when smaller is byte-identical to larger's draw prefix."""
    if smaller.empty and larger.empty:
        return True
    left = (
        filter_draw_prefix(smaller, max_draw_id)
        .sort_values(["draw", "player_id"])
        .reset_index(drop=True)
    )
    right = (
        filter_draw_prefix(larger, max_draw_id)
        .sort_values(["draw", "player_id"])
        .reset_index(drop=True)
    )
    if len(left) != len(right):
        return False
    if list(left.columns) != list(right.columns):
        return False
    for col in left.columns:
        if col in {"fantasy_pts_season", "sim_vorp_draw"}:
            if not np.allclose(left[col].to_numpy(), right[col].to_numpy(), equal_nan=True):
                return False
        elif not left[col].equals(right[col]):
            return False
    return True


def stability_contract_hashes(
    *,
    season: int,
    selected_board_hash: str,
    selected_board: pd.DataFrame,
    wr_calibration_hash: str,
    canonical_projection_run_id: str = "draw_stability_frozen",
    selected_board_model_id: str = "accuracy_first_ensemble",
) -> dict[str, str]:
    roster_config = load_roster_configuration()
    replacement = build_replacement_contract(
        selected_board,
        season=season,
        selected_board_hash=selected_board_hash,
        selected_board_model_id=selected_board_model_id,
        canonical_projection_run_id=canonical_projection_run_id,
        roster_config=roster_config,
    )
    finish_gate_hash = ""
    if FINISH_GATE_PATH.exists():
        gate_body = json.loads(FINISH_GATE_PATH.read_text(encoding="utf-8"))
        gate_body.pop("generated_at", None)
        finish_gate_hash = sha256_file(str(FINISH_GATE_PATH))
    return {
        "selected_board_hash": selected_board_hash,
        "roster_configuration_hash": roster_configuration_hash(roster_config),
        "replacement_contract_hash": replacement["contract_hash"],
        "finish_probability_gate_hash": finish_gate_hash,
        "wr_calibration_artifact_hash": wr_calibration_hash,
    }


def verify_contract_match(
    candidate: dict[str, str],
    reference: dict[str, str],
) -> list[str]:
    mismatches: list[str] = []
    for key, ref_val in reference.items():
        cand_val = candidate.get(key)
        if cand_val != ref_val:
            mismatches.append(f"{key}: candidate={cand_val!r} reference={ref_val!r}")
    return mismatches


def _metric_tolerance(metric: str, tolerances: dict[str, float]) -> float:
    key = METRIC_TOLERANCE_KEYS.get(metric)
    if key and key in tolerances:
        return float(tolerances[key])
    return float(tolerances.get("probability_abs", 0.015))


def summarize_candidate_metrics(
    recentered_draws: pd.DataFrame,
    *,
    board: pd.DataFrame,
    max_draws: int,
) -> pd.DataFrame:
    """Compute overlay metrics from a nested draw prefix."""
    prefix = filter_draw_prefix(recentered_draws, max_draws)
    percentiles = summarize_simulations(prefix)
    finish = compute_finish_probabilities(prefix)
    vorp = compute_simulated_vorp_metrics(prefix, board)
    out = percentiles.merge(finish, on="player_id", how="outer")
    out = out.merge(vorp, on="player_id", how="outer")
    out["player_id"] = out["player_id"].astype(str)
    return out


def compare_metric_distributions(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = STABILITY_METRICS,
    tolerances: dict[str, float],
    adp_by_player: pd.Series | None = None,
    core_adp_max: int = 36,
) -> dict[str, Any]:
    merged = candidate.merge(
        reference,
        on="player_id",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    per_metric: dict[str, Any] = {}
    player_diffs: list[dict[str, Any]] = []

    for metric in metrics:
        cand_col = f"{metric}_candidate"
        ref_col = f"{metric}_reference"
        if cand_col not in merged.columns or ref_col not in merged.columns:
            continue
        diff = (merged[cand_col] - merged[ref_col]).abs()
        tol = _metric_tolerance(metric, tolerances)
        median_diff = float(diff.median()) if len(diff) else float("nan")
        p95_diff = float(diff.quantile(0.95)) if len(diff) else float("nan")
        max_diff = float(diff.max()) if len(diff) else float("nan")
        max_idx = diff.idxmax() if len(diff) else None
        max_player = str(merged.loc[max_idx, "player_id"]) if max_idx is not None else None
        max_adp = None
        if max_player and adp_by_player is not None and max_player in adp_by_player.index:
            max_adp = float(adp_by_player[max_player])
        per_metric[metric] = {
            "median_abs_diff": round(median_diff, 6),
            "p95_abs_diff": round(p95_diff, 6),
            "max_abs_diff": round(max_diff, 6),
            "tolerance": tol,
            "median_passes": bool(median_diff <= tol) if not np.isnan(median_diff) else False,
            "p95_passes": bool(p95_diff <= tol) if not np.isnan(p95_diff) else False,
            "max_player_id": max_player,
            "max_player_adp": max_adp,
            "max_is_core_drafted": (
                max_adp is not None and max_adp <= core_adp_max
            ),
        }
        for idx, row in merged.iterrows():
            player_diffs.append(
                {
                    "player_id": str(row["player_id"]),
                    "metric": metric,
                    "abs_diff": float(abs(row[cand_col] - row[ref_col])),
                    "candidate": float(row[cand_col]),
                    "reference": float(row[ref_col]),
                }
            )

    volatile = (
        pd.DataFrame(player_diffs)
        .sort_values("abs_diff", ascending=False)
        .head(20)
        .to_dict(orient="records")
        if player_diffs
        else []
    )
    return {"per_metric": per_metric, "volatile_players": volatile, "n_players": len(merged)}


def enumerate_decision_changes(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    threshold_registry: dict[str, dict[str, Any]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    registry = threshold_registry or resolve_decision_threshold_registry(thresholds)
    merged = candidate.merge(reference, on="player_id", how="inner", suffixes=("_c", "_r"))
    events: list[dict[str, Any]] = []
    for metric, spec in registry.items():
        cand_col = f"{metric}_c"
        ref_col = f"{metric}_r"
        if cand_col not in merged.columns or ref_col not in merged.columns:
            continue
        threshold = float(spec["threshold"])
        direction = str(spec["direction"])
        kind = str(spec.get("kind", "probability"))
        for _, row in merged.iterrows():
            candidate_value = float(row[cand_col])
            reference_value = float(row[ref_col])
            candidate_qualifies = qualifies(candidate_value, threshold, direction)
            reference_qualifies = qualifies(reference_value, threshold, direction)
            if candidate_qualifies == reference_qualifies:
                continue
            events.append(
                {
                    "player_id": str(row["player_id"]),
                    "metric": metric,
                    "metric_kind": kind,
                    "threshold": threshold,
                    "threshold_direction": direction,
                    "candidate_value": candidate_value,
                    "reference_value": reference_value,
                    "abs_diff": abs(candidate_value - reference_value),
                    "distance_from_threshold_reference": signed_distance_from_threshold(
                        reference_value,
                        threshold,
                        direction,
                    ),
                    "candidate_qualifies": candidate_qualifies,
                    "reference_qualifies": reference_qualifies,
                }
            )
    return events


def count_decision_changes(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    threshold_registry: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    registry = threshold_registry or resolve_decision_threshold_registry(thresholds)
    events = enumerate_decision_changes(
        candidate,
        reference,
        threshold_registry=registry,
    )
    changes: dict[str, int] = {metric: 0 for metric in registry}
    for event in events:
        changes[event["metric"]] = changes.get(event["metric"], 0) + 1
    total = len(events)
    threshold_values = {metric: spec["threshold"] for metric, spec in registry.items()}
    return {"by_metric": changes, "total": int(total), "thresholds": threshold_values}


def load_intermediate_stability_config(config: dict | None = None) -> dict[str, Any]:
    sim_config = config or load_simulation_config()
    defaults = {
        "reference_draws": 20000,
        "candidate_draw_counts": [7500, 10000, 15000],
        "checkpoint_dir_suffix": "draws=10000",
        "require_provenance_ok": True,
        "prior_sweep_reference_draws": 10000,
    }
    return {**defaults, **(sim_config.get("intermediate_stability") or {})}


def default_stability_checkpoint_dir(season: int, *, suffix: str = "draws=10000") -> Path:
    return Path(MODEL_V3_DIR) / "draw_stability" / f"season={season}" / suffix


def read_checkpoint_meta(checkpoint_dir: Path) -> dict[str, Any]:
    meta_path = Path(checkpoint_dir) / "checkpoint_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def verify_checkpoint_draws(checkpoint_dir: Path, *, reference_draws: int) -> None:
    meta = read_checkpoint_meta(checkpoint_dir)
    completed = int(meta.get("completed_draws", 0))
    if completed < reference_draws:
        raise ValueError(
            f"Checkpoint {checkpoint_dir} has {completed} draws; need {reference_draws}"
        )


def load_stability_draws_from_checkpoint(
    checkpoint_dir: Path,
    selected_board: pd.DataFrame,
    *,
    max_draws: int,
) -> pd.DataFrame:
    from src.projection.evaluation.decision_change_diagnostics import (
        build_recentered_draws_from_checkpoint,
    )

    verify_checkpoint_draws(checkpoint_dir, reference_draws=max_draws)
    return build_recentered_draws_from_checkpoint(
        checkpoint_dir,
        selected_board,
        max_draws=max_draws,
    )


def _boundary_margin(metric_kind: str, config: dict[str, Any]) -> float:
    if metric_kind == "rank":
        return float(config.get("boundary_margin_rank", 0.25))
    return float(config.get("boundary_margin_probability", 0.02))


def classify_candidate_vs_reference_event(
    event: dict[str, Any],
    *,
    config: dict[str, Any],
) -> str:
    metric_kind = str(event.get("metric_kind", "probability"))
    ref_distance = float(event["distance_from_threshold_reference"])
    if abs(ref_distance) <= _boundary_margin(metric_kind, config):
        return "boundary_noise"
    return "material"


def evaluate_production_decision_events(
    candidate_metrics: pd.DataFrame,
    reference_metrics: pd.DataFrame,
    *,
    threshold_registry: dict[str, dict[str, Any]],
    player_meta: pd.DataFrame,
    diagnostic_config: dict[str, Any],
) -> dict[str, Any]:
    events = enumerate_decision_changes(
        candidate_metrics,
        reference_metrics,
        threshold_registry=threshold_registry,
    )
    meta_by_player = player_meta.set_index("player_id")
    by_category: dict[str, int] = {}
    material_count = 0
    core_count = 0
    classified_events: list[dict[str, Any]] = []
    for event in events:
        player_id = event["player_id"]
        category = classify_candidate_vs_reference_event(event, config=diagnostic_config)
        by_category[category] = by_category.get(category, 0) + 1
        if category == "material":
            material_count += 1
        is_core = False
        if player_id in meta_by_player.index:
            is_core = bool(meta_by_player.loc[player_id].get("is_core_adp_player"))
        if is_core:
            core_count += 1
        classified_events.append({**event, "category": category, "is_core_adp_player": is_core})
    return {
        "total": len(events),
        "material_decision_events": material_count,
        "core_adp_decision_events": core_count,
        "decision_events_by_category": by_category,
        "events": classified_events,
    }


def _build_player_meta_for_stability(
    selected_board: pd.DataFrame,
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
    meta = board[["player_id", "position", "adp"]].copy()
    meta["is_core_adp_player"] = meta["adp"].notna() & (meta["adp"] <= core_adp_threshold)
    return meta


def passes_numerical_stability(comparison: dict[str, Any]) -> bool:
    for metric_info in comparison.get("per_metric", {}).values():
        if not metric_info.get("median_passes") or not metric_info.get("p95_passes"):
            return False
    return True


def candidate_passes_gate(
    comparison: dict[str, Any],
    *,
    decision_changes: dict[str, Any],
    max_decision_changes: int = 0,
    gate_mode: str = "legacy",
    production_decision: dict[str, Any] | None = None,
) -> bool:
    if gate_mode == "production_v20k":
        if production_decision is None:
            return False
        if not passes_numerical_stability(comparison):
            return False
        if production_decision.get("material_decision_events", 0) > 0:
            return False
        if production_decision.get("core_adp_decision_events", 0) > 0:
            return False
        return True
    if decision_changes.get("total", 0) > max_decision_changes:
        return False
    return passes_numerical_stability(comparison)


def select_smallest_passing_draw_count(
    candidate_results: list[dict[str, Any]],
    *,
    reference_draws: int | None = None,
) -> int | None:
    passing = [
        row
        for row in candidate_results
        if row.get("passes_gate")
        and (reference_draws is None or int(row["draw_count"]) < int(reference_draws))
    ]
    if not passing:
        return None
    return int(min(row["draw_count"] for row in passing))


def generate_recentered_reference_draws(
    projections: pd.DataFrame,
    *,
    season: int,
    selected_board: pd.DataFrame,
    selected_board_hash: str,
    reference_draws: int,
    canonical_projection_run_id: str,
    calibration_hash: str,
    wr_calibration_hash: str,
    uncertainty_manifest: dict | None = None,
    checkpoint_dir: Path | None = None,
    batch_size: int = 500,
) -> pd.DataFrame:
    sim_config = load_simulation_config()
    configured_seed = int(sim_config.get("random_seed") or 2026)
    seed = stability_simulation_seed(
        season=season,
        board_hash=selected_board_hash,
        calibration_hash=calibration_hash,
        configured_seed=configured_seed,
        canonical_projection_run_id=canonical_projection_run_id,
        transform_version=TRANSFORM_VERSION,
        wr_calibration_hash=wr_calibration_hash,
    )
    checkpoint_root = checkpoint_dir or (
        Path(MODEL_V3_DIR)
        / "draw_stability"
        / f"season={season}"
        / f"draws={reference_draws}"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoint_root / "checkpoint_meta.json"
    rng_path = checkpoint_root / "checkpoint_rng.pkl"
    raw_parts = sorted(checkpoint_root.glob("raw_part_*.parquet"))

    completed = 0
    rng = None
    if meta_path.exists() and rng_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        completed = int(meta.get("completed_draws", 0))
        with rng_path.open("rb") as handle:
            rng = pickle.load(handle)

    progress_every = max(batch_size, 100)
    while completed < reference_draws:
        batch_end = min(completed + batch_size, reference_draws)
        part_path = checkpoint_root / f"raw_part_{completed:05d}.parquet"
        if part_path.exists():
            completed = batch_end
            continue
        batch_draws, rng = simulate_season_draw_range(
            projections,
            start_draw=completed,
            end_draw=batch_end,
            seed=seed,
            uncertainty_manifest=uncertainty_manifest,
            progress_every=progress_every,
            total_draws=reference_draws,
            rng=rng,
        )
        batch_draws.to_parquet(part_path, index=False)
        meta_path.write_text(
            json.dumps({"completed_draws": batch_end, "seed": seed}),
            encoding="utf-8",
        )
        with rng_path.open("wb") as handle:
            pickle.dump(rng, handle)
        print(f"checkpoint saved draws 0-{batch_end}/{reference_draws}", flush=True)
        completed = batch_end

    raw_frames = [pd.read_parquet(path) for path in sorted(checkpoint_root.glob("raw_part_*.parquet"))]
    draws = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    wr_calibration = load_wr_calibration()
    wr_scale = float((wr_calibration or {}).get("selected_wr_scale", 1.0))
    selected_points = board_points_series(selected_board)
    recentered = recenter_draws_wr_scaled(draws, selected_points, wr_scale=wr_scale)
    cache_path = checkpoint_root / "recentered_draws.parquet"
    slim_draw_frame(recentered).to_parquet(cache_path, index=False)
    return recentered


def evaluate_draw_stability(
    recentered_draws: pd.DataFrame,
    *,
    season: int,
    selected_board: pd.DataFrame,
    selected_board_hash: str,
    wr_calibration_hash: str,
    candidates: list[int],
    reference_draws: int,
    top_adp: int = TOP_ADP,
    tolerances: dict[str, float] | None = None,
    decision_thresholds: dict[str, float] | None = None,
    core_adp_max: int = 36,
    contract_hashes: dict[str, str] | None = None,
    gate_mode: str = "legacy",
    diagnostic_config: dict[str, Any] | None = None,
    projections: pd.DataFrame | None = None,
) -> dict[str, Any]:
    sim_config = load_simulation_config()
    tolerances = {**(sim_config.get("stability_tolerances") or {}), **(tolerances or {})}
    decision_thresholds = {
        **(sim_config.get("decision_thresholds") or {}),
        **(decision_thresholds or {}),
    }
    core_adp_max = int(sim_config.get("core_drafted_adp_max") or core_adp_max)
    if diagnostic_config is None:
        from src.projection.evaluation.decision_change_diagnostics import (
            load_decision_diagnostic_config,
        )

        diagnostic_config = load_decision_diagnostic_config(sim_config)
    core_adp_threshold = int(diagnostic_config.get("core_adp_threshold", core_adp_max))

    if contract_hashes is None:
        contract_hashes = stability_contract_hashes(
            season=season,
            selected_board_hash=selected_board_hash,
            selected_board=selected_board,
            wr_calibration_hash=wr_calibration_hash,
        )

    adp_col = "adp" if "adp" in selected_board.columns else None
    top_players = selected_board.copy()
    if adp_col:
        top_players = top_players[pd.to_numeric(top_players[adp_col], errors="coerce") <= top_adp]
    top_ids = set(top_players["player_id"].astype(str))
    adp_by_player = None
    if adp_col:
        adp_by_player = (
            top_players.set_index(top_players["player_id"].astype(str))[adp_col]
            .astype(float)
        )

    reference_metrics = summarize_candidate_metrics(
        recentered_draws,
        board=selected_board,
        max_draws=reference_draws,
    )
    reference_metrics = reference_metrics[
        reference_metrics["player_id"].isin(top_ids)
    ].copy()

    player_meta = _build_player_meta_for_stability(
        selected_board,
        core_adp_threshold=core_adp_threshold,
    )
    player_meta = player_meta[player_meta["player_id"].isin(top_ids)].copy()
    threshold_registry = resolve_decision_threshold_registry(decision_thresholds)

    candidate_rows: list[dict[str, Any]] = []
    for draw_count in sorted(candidates):
        if draw_count > reference_draws:
            continue
        candidate_contract = dict(contract_hashes)
        mismatches = verify_contract_match(candidate_contract, contract_hashes)
        if mismatches:
            candidate_rows.append(
                {
                    "draw_count": draw_count,
                    "passes_gate": False,
                    "contract_mismatches": mismatches,
                }
            )
            continue

        candidate_metrics = summarize_candidate_metrics(
            recentered_draws,
            board=selected_board,
            max_draws=draw_count,
        )
        candidate_metrics = candidate_metrics[
            candidate_metrics["player_id"].isin(top_ids)
        ].copy()
        comparison = compare_metric_distributions(
            candidate_metrics,
            reference_metrics,
            tolerances=tolerances,
            adp_by_player=adp_by_player,
            core_adp_max=core_adp_max,
        )
        decision_changes = count_decision_changes(
            candidate_metrics,
            reference_metrics,
            threshold_registry=threshold_registry,
        )
        production_decision = None
        if gate_mode == "production_v20k":
            production_decision = evaluate_production_decision_events(
                candidate_metrics,
                reference_metrics,
                threshold_registry=threshold_registry,
                player_meta=player_meta,
                diagnostic_config=diagnostic_config,
            )
        passes = candidate_passes_gate(
            comparison,
            decision_changes=decision_changes,
            gate_mode=gate_mode,
            production_decision=production_decision,
        )
        row = {
            "draw_count": draw_count,
            "passes_gate": passes,
            "passes_numerical": passes_numerical_stability(comparison),
            "contract_hashes": candidate_contract,
            "comparison": comparison,
            "decision_changes": decision_changes,
        }
        if production_decision is not None:
            row.update(
                {
                    "material_decision_events": production_decision["material_decision_events"],
                    "core_adp_decision_events": production_decision["core_adp_decision_events"],
                    "decision_events_by_category": production_decision[
                        "decision_events_by_category"
                    ],
                }
            )
        candidate_rows.append(row)

    recommended = select_smallest_passing_draw_count(
        candidate_rows,
        reference_draws=reference_draws,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "reference_draws": reference_draws,
        "top_adp": top_adp,
        "tolerances": tolerances,
        "decision_thresholds": decision_thresholds,
        "contract_hashes": contract_hashes,
        "recommended_draw_count": recommended,
        "candidates": candidate_rows,
        "nested_prefix_design": True,
        "gate_mode": gate_mode,
    }


def _import_holdout_helpers():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.evaluate_recentered_distribution import (
        build_finish_probability_frame,
        evaluate_holdout,
    )
    from src.projection.evaluation.finish_probability_calibration import (
        evaluate_finish_probability_report,
    )
    from src.projection.inference.wr_calibration import recenter_draws_wr_scaled

    return {
        "evaluate_holdout": evaluate_holdout,
        "build_finish_probability_frame": build_finish_probability_frame,
        "evaluate_finish_probability_report": evaluate_finish_probability_report,
        "recenter_draws_wr_scaled": recenter_draws_wr_scaled,
    }


def evaluate_holdout_calibration_stability(
    *,
    holdout_draws: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    candidates: list[int],
    reference_draws: int,
    wr_scale: float = 1.0,
) -> dict[str, Any]:
    """Optional holdout calibration comparison across nested draw prefixes."""
    helpers = _import_holdout_helpers()
    evaluate_holdout = helpers["evaluate_holdout"]
    build_finish_probability_frame = helpers["build_finish_probability_frame"]
    evaluate_finish_probability_report = helpers["evaluate_finish_probability_report"]
    recenter_draws_wr_scaled = helpers["recenter_draws_wr_scaled"]
    selected = holdout_frame.set_index("player_id")["selected_pred"].astype(float)

    rows = []
    max_available = int(holdout_draws["draw"].max()) + 1 if not holdout_draws.empty else 0
    effective_reference = min(reference_draws, max_available)
    for draw_count in sorted(candidates):
        if draw_count > effective_reference:
            continue
        prefix = filter_draw_prefix(holdout_draws, draw_count)
        report = evaluate_holdout(prefix, holdout_frame, wr_scale=wr_scale)
        recentered = recenter_draws_wr_scaled(prefix, selected, wr_scale=wr_scale)
        finish_scored, _ = build_finish_probability_frame(
            recentered,
            holdout_frame,
            training_seasons=(2024,),
        )
        finish_calibration = evaluate_finish_probability_report(finish_scored)
        rows.append(
            {
                "draw_count": draw_count,
                "coverage_80": report["recentered_metrics"]["overall"].get("coverage"),
                "wr_coverage": report["recentered_metrics"]["by_position"]
                .get("WR", {})
                .get("coverage"),
                "interval_score": report["recentered_metrics"]["overall"].get("interval_score"),
                "p50_mae": report["recentered_metrics"]["overall"].get("p50_mae"),
                "p50_spearman": report["recentered_metrics"]["overall"].get("p50_spearman"),
                "finish_brier_top12": _finish_brier(finish_calibration, 12),
                "finish_brier_top24": _finish_brier(finish_calibration, 24),
                "finish_brier_top36": _finish_brier(finish_calibration, 36),
                "reliability_slope_top12": _finish_slope(finish_calibration, 12),
            }
        )
    return {
        "status": "ok",
        "reference_draws_requested": reference_draws,
        "reference_draws_available": effective_reference,
        "candidates": rows,
    }


def _finish_brier(finish_calibration: dict, cutoff: int) -> float | None:
    for check in finish_calibration.get("checks") or []:
        if int(check.get("cutoff", 0)) == cutoff:
            return check.get("brier")
    return None


def _finish_slope(finish_calibration: dict, cutoff: int) -> float | None:
    for check in finish_calibration.get("checks") or []:
        if int(check.get("cutoff", 0)) == cutoff:
            return check.get("calibration_slope")
    return None


def write_draw_stability_artifacts(
    report: dict[str, Any],
    *,
    season: int,
    player_diffs: pd.DataFrame | None = None,
    sweep_phase: str | None = None,
) -> dict[str, str]:
    out_dir = Path(MODEL_V3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sweep_phase:
        json_path = out_dir / f"draw_stability_{sweep_phase}_{season}.json"
    else:
        json_path = out_dir / f"draw_stability_{season}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths = {"draw_stability_json": str(json_path)}
    if player_diffs is not None and not player_diffs.empty:
        parquet_name = (
            f"draw_stability_{sweep_phase}_{season}.parquet"
            if sweep_phase
            else f"draw_stability_{season}.parquet"
        )
        parquet_path = out_dir / parquet_name
        player_diffs.to_parquet(parquet_path, index=False)
        paths["draw_stability_parquet"] = str(parquet_path)
    return paths


def _policy_matrix_outcome(
    candidate_rows: list[dict[str, Any]],
    *,
    recommended: int | None,
) -> str:
    if any(row.get("core_adp_decision_events", 0) > 0 for row in candidate_rows):
        return "escalate_core_adp_events"
    if recommended is not None:
        return f"{recommended}_pass"
    return "all_candidates_failed"


def _production_recommendation(
    stability_report: dict[str, Any],
    *,
    candidate_rows: list[dict[str, Any]],
) -> str:
    if any(row.get("core_adp_decision_events", 0) > 0 for row in candidate_rows):
        return "escalate_core_adp_events"
    recommended = stability_report.get("recommended_draw_count")
    if recommended is not None:
        return "candidate_ready_pending_publish_rollout"
    if stability_report.get("gate_mode") == "production_v20k":
        return "decision_stable_sub20k_numerical_fail_schedule_rc_or_20k"
    return "no_candidate_passed_increase_reference_or_review_tolerances"


def write_draw_count_decision(
    *,
    season: int,
    stability_report: dict[str, Any],
    holdout_calibration: dict[str, Any] | None = None,
    rationale: str | None = None,
    sweep_phase: str | None = None,
    provenance_verdict: str | None = None,
    prior_sweep_reference_draws: int | None = None,
) -> Path:
    out_dir = Path(OUTPUT_DIR) / "model_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = stability_report.get("candidates") or []
    recommended = stability_report.get("recommended_draw_count")
    gate_mode = stability_report.get("gate_mode", "legacy")
    schema_version = "draw_count_decision_v2" if gate_mode == "production_v20k" else "draw_count_decision_v1"
    stability_report_name = (
        f"draw_stability_{sweep_phase}_{season}.json"
        if sweep_phase
        else f"draw_stability_{season}.json"
    )
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "selected_draw_count": recommended,
        "reference_draws": stability_report.get("reference_draws"),
        "production_recommendation": _production_recommendation(
            stability_report,
            candidate_rows=candidate_rows,
        ),
        "tolerances": stability_report.get("tolerances"),
        "nested_prefix_design": True,
        "stability_report_path": str(out_dir / stability_report_name),
        "holdout_calibration": holdout_calibration,
        "rationale": rationale,
    }
    if gate_mode == "production_v20k":
        payload.update(
            {
                "sweep_phase": sweep_phase,
                "gate_mode": gate_mode,
                "candidates_evaluated": [row["draw_count"] for row in candidate_rows],
                "provenance_verdict": provenance_verdict,
                "prior_sweep_reference_draws": prior_sweep_reference_draws,
                "policy_matrix_outcome": _policy_matrix_outcome(
                    candidate_rows,
                    recommended=recommended,
                ),
            }
        )
    elif recommended is None:
        payload["production_recommendation"] = (
            "no_candidate_passed_increase_reference_or_review_tolerances"
        )
    else:
        payload["production_recommendation"] = "retain_current_draw_count"
    path = out_dir / "draw_count_decision.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
