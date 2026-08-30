"""Paired E2 evaluation: with-QB-context vs without-QB-context candidates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.projection.contracts import OUTPUT_DIR, REPO_ROOT
from src.projection.data_prep import get_conn
from src.projection.evaluation.accuracy_first import sha256_file
from src.projection.evaluation.calibration_segments import MINIMUM_N_FOR_GATE
from src.projection.evaluation.decision_quality import DEFAULT_FOLDS
from src.projection.fantasy_evaluation import run_evaluation
from src.projection.features import build_player_season_features
from src.projection.fantasy_evaluation import build_leakage_safe_long_board
from src.projection.qb_context import (
    QB_CONTEXT_FEATURES,
    assert_temporal_invariance,
    build_team_qb_context,
    model_artifact_manifest,
)
from src.projection.shadow.evaluate_0a import evaluate_shadow_0a_on_long_board

QB_CONTEXT_EVAL_DIR = Path(OUTPUT_DIR) / "evaluation" / "qb_context"
FROZEN_BASELINE_ID = "qb_context_baseline_v1"


def _cohort_labels(frame: pd.DataFrame, qb_context: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ctx = qb_context.set_index("team")
    out["qb_changed_cohort"] = out["preseason_team"].map(ctx.get("qb_changed", pd.Series(dtype=float))).fillna(0).astype(int)
    epa_delta = out["preseason_team"].map(ctx.get("qb_epa_change_vs_prev", pd.Series(dtype=float)))
    out["qb_quality_cohort"] = np.select(
        [
            out["preseason_team"].map(ctx.get("qb_rookie_or_unknown", pd.Series(dtype=float))).fillna(0).astype(bool),
            epa_delta.gt(0.05),
            epa_delta.lt(-0.05),
        ],
        ["unknown", "upgrade", "downgrade"],
        default="continuity",
    )
    prior_team = out.get("prior_team", out.get("team"))
    out["team_change_cohort"] = np.where(
        out.get("is_rookie", False),
        "rookie_or_no_prior_team",
        np.where(out["preseason_team"].astype(str).eq(prior_team.astype(str)), "same_team", "changed_team"),
    )
    out["adp_band"] = _assign_adp_band(out.get("adp", pd.Series(index=out.index)))
    out["preseason_team_cohort"] = out["preseason_team"].astype(str)
    return out


def evaluate_qb_context_fold(
    source_season: int,
    target_season: int,
) -> dict[str, Any]:
    """Compare baseline and QB-context candidates on one fold."""
    baseline_rows, baseline_summary, baseline_meta = run_evaluation(
        source_season, target_season, use_qb_context=False
    )
    candidate_rows, candidate_summary, candidate_meta = run_evaluation(
        source_season, target_season, use_qb_context=True
    )

    conn = get_conn()
    try:
        feat = build_player_season_features(conn)
        qb_ctx = build_team_qb_context(
            conn, feat, source_season=source_season, target_season=target_season
        )
        baseline_long = build_leakage_safe_long_board(
            conn, feat, source_season, target_season, use_qb_context=False
        )
        candidate_long = build_leakage_safe_long_board(
            conn, feat, source_season, target_season, use_qb_context=True
        )
        shadow_baseline = evaluate_shadow_0a_on_long_board(baseline_long)
        shadow_candidate = evaluate_shadow_0a_on_long_board(candidate_long)
    finally:
        conn.close()

    cohort_frame = _cohort_labels(baseline_rows, qb_ctx)
    segment_rows: list[dict[str, Any]] = []
    for label, col in (
        ("pure_model", "model_points_end_to_end"),
    ):
        for cohort_col in (
            "qb_changed_cohort",
            "qb_quality_cohort",
            "team_change_cohort",
            "adp_band",
            "preseason_team_cohort",
        ):
            work = cohort_frame.copy()
            work["segment_value"] = work[cohort_col].astype(str)
            for value, grp in work.groupby("segment_value", observed=True):
                if grp.empty:
                    continue
                merged = grp.merge(
                    baseline_rows[["player_id", "model_points_end_to_end"]],
                    on="player_id",
                    how="left",
                    suffixes=("", "_base"),
                ).merge(
                    candidate_rows[["player_id", "model_points_end_to_end"]],
                    on="player_id",
                    how="left",
                    suffixes=("", "_cand"),
                )
                actual = pd.to_numeric(merged["actual_points"], errors="coerce")
                base_pred = pd.to_numeric(merged["model_points_end_to_end_base"], errors="coerce")
                cand_pred = pd.to_numeric(merged["model_points_end_to_end_cand"], errors="coerce")
                n = int(len(grp))
                segment_rows.append({
                    "source_season": source_season,
                    "target_season": target_season,
                    "cohort_type": cohort_col,
                    "cohort_value": str(value),
                    "n": n,
                    "eligible_for_gate": n >= MINIMUM_N_FOR_GATE,
                    "baseline_points_mae": float((base_pred - actual).abs().mean()),
                    "candidate_points_mae": float((cand_pred - actual).abs().mean()),
                    "baseline_spearman": float(base_pred.corr(actual, method="spearman")) if n >= 3 else float("nan"),
                    "candidate_spearman": float(cand_pred.corr(actual, method="spearman")) if n >= 3 else float("nan"),
                })

    return {
        "source_season": source_season,
        "target_season": target_season,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "baseline_metadata": baseline_meta,
        "candidate_metadata": candidate_meta,
        "cohort_metrics": pd.DataFrame(segment_rows),
        "shadow_baseline_realized": shadow_baseline,
        "shadow_candidate_realized": shadow_candidate,
        "shadow_incumbent_agreement_baseline": shadow_baseline.get("target_mae_top_adp"),
        "shadow_incumbent_agreement_candidate": shadow_candidate.get("target_mae_top_adp"),
        "qb_context_manifest": model_artifact_manifest(consumes_qb_context=True),
    }


def evaluate_qb_context_rolling(
    folds: tuple[int, ...] = DEFAULT_FOLDS,
) -> dict[str, Any]:
    results = [evaluate_qb_context_fold(target - 1, target) for target in folds]
    return aggregate_qb_context_results(results)


def aggregate_qb_context_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    cohort = pd.concat(
        [r["cohort_metrics"] for r in results if not r["cohort_metrics"].empty],
        ignore_index=True,
    ) if results else pd.DataFrame()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": results,
        "cohort_metrics": cohort,
        "contract_hashes": vorp_tier_contract_hashes(),
        "qb_context_features": list(QB_CONTEXT_FEATURES),
    }


def write_qb_context_evidence(
    payload: dict[str, Any],
    *,
    bundle_id: str,
    output_dir: Path | None = None,
) -> Path:
    root = Path(output_dir or QB_CONTEXT_EVAL_DIR) / bundle_id
    root.mkdir(parents=True, exist_ok=True)
    sha256: dict[str, str] = {}
    paths: dict[str, str] = {}

    cohort = payload.get("cohort_metrics")
    if isinstance(cohort, pd.DataFrame) and not cohort.empty:
        path = root / "cohort_metrics.parquet"
        cohort.to_parquet(path, index=False)
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        paths["cohort_metrics.parquet"] = rel
        sha256["cohort_metrics.parquet"] = sha256_file(path)

    summary = {
        "bundle_id": bundle_id,
        "bundle_type": "qb_context_evidence",
        "generated_at": payload.get("generated_at"),
        "folds": [
            {
                "source_season": f["source_season"],
                "target_season": f["target_season"],
                "shadow_baseline_realized": f.get("shadow_baseline_realized"),
                "shadow_candidate_realized": f.get("shadow_candidate_realized"),
            }
            for f in payload.get("folds", [])
        ],
        "qb_context_features": payload.get("qb_context_features"),
        "contract_hashes": payload.get("contract_hashes"),
        "source_paths": paths,
        "sha256": sha256,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["sha256"]["manifest.json"] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path


def run_temporal_mutation_check(
    conn,
    feature_table: pd.DataFrame,
    *,
    source_season: int,
    target_season: int,
) -> bool:
    """Prove target-season outcome mutations do not alter preseason QB context."""
    baseline = build_team_qb_context(
        conn, feature_table, source_season=source_season, target_season=target_season
    )
    mutated_table = feature_table.copy()
    target_mask = mutated_table["season"].eq(target_season)
    for col in ("passing_yards", "passing_tds", "attempts", "games_played"):
        if col in mutated_table.columns:
            mutated_table.loc[target_mask, col] = mutated_table.loc[target_mask, col].fillna(0) * 2 + 100
    mutated = build_team_qb_context(
        conn, mutated_table, source_season=source_season, target_season=target_season
    )
    return assert_temporal_invariance(baseline, mutated)
