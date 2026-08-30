"""Calibration report and summary artifacts for segmented evaluation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.projection.evaluation.accuracy_first import canonical_json_hash
from src.projection.evaluation.calibration_segments import (
    MINIMUM_N_FOR_GATE,
    evaluate_all_segments,
)

DEFAULT_THRESHOLDS = {
    "nominal_coverage": 0.80,
    "aggregate_coverage_min": 0.75,
    "aggregate_coverage_max": 0.85,
    "segment_coverage_min": 0.70,
    "segment_coverage_max": 0.90,
    "minimum_n_for_gate": MINIMUM_N_FOR_GATE,
    "max_failed_eligible_segments": 0,
}


def _segment_passes(row: pd.Series, thresholds: dict) -> bool:
    if not bool(row.get("eligible_for_gate")):
        return True
    cov = row.get("coverage_80")
    if pd.isna(cov):
        return False
    return (
        thresholds["segment_coverage_min"]
        <= float(cov)
        <= thresholds["segment_coverage_max"]
    )


def build_segment_summary(
    segments: pd.DataFrame,
    *,
    thresholds: dict | None = None,
    artifact_hashes: dict | None = None,
) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    eligible = segments[segments["eligible_for_gate"].fillna(False)]
    failed = eligible[~eligible.apply(_segment_passes, axis=1, thresholds=thresholds)]

    overall = segments[segments["segment_type"].eq("overall")]
    overall_cov = float(overall["coverage_80"].iloc[0]) if not overall.empty else float("nan")
    aggregate_pass = (
        thresholds["aggregate_coverage_min"]
        <= overall_cov
        <= thresholds["aggregate_coverage_max"]
    )

    position_rows = segments[segments["segment_type"].eq("position_gate")]
    position_failed = position_rows[
        position_rows["eligible_for_gate"].fillna(False)
        & ~position_rows.apply(_segment_passes, axis=1, thresholds=thresholds)
    ]

    worst_cov = eligible.sort_values("coverage_80", ascending=True).head(1)
    worst_width = eligible.sort_values("mean_interval_width_80", ascending=False).head(1)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "n": int(overall["n"].iloc[0]) if not overall.empty else 0,
            "coverage_80": overall_cov,
            "passes": aggregate_pass,
        },
        "per_position": position_rows[
            ["segment_value", "n", "coverage_80", "median_absolute_error", "spearman_rank_correlation"]
        ].to_dict(orient="records"),
        "worst_eligible_by_coverage": (
            worst_cov.iloc[0].to_dict() if not worst_cov.empty else None
        ),
        "worst_eligible_by_interval_width": (
            worst_width.iloc[0].to_dict() if not worst_width.empty else None
        ),
        "failed_eligible_segment_count": int(len(failed)),
        "failed_position_gate_count": int(len(position_failed)),
        "passes_segment_gate": (
            aggregate_pass
            and len(position_failed) == 0
            and len(failed) <= int(thresholds["max_failed_eligible_segments"])
        ),
        "thresholds": thresholds,
        "artifact_hashes": artifact_hashes or {},
        "summary_hash": None,
    }


def finalize_summary(summary: dict) -> dict:
    payload = dict(summary)
    payload.pop("summary_hash", None)
    payload["summary_hash"] = canonical_json_hash(payload)
    return payload


def write_calibration_artifacts(
    scored_frame: pd.DataFrame,
    out_dir: Path,
    *,
    season: int,
    run_id: str,
    artifact_hashes: dict | None = None,
    thresholds: dict | None = None,
) -> dict:
    out_dir = Path(out_dir) / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = evaluate_all_segments(scored_frame)
    segments_path = out_dir / "calibration_segments.parquet"
    segments.to_parquet(segments_path, index=False)
    summary = finalize_summary(
        build_segment_summary(segments, thresholds=thresholds, artifact_hashes=artifact_hashes)
    )
    summary["season"] = int(season)
    summary["run_id"] = str(run_id)
    summary_path = out_dir / "calibration_segment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "segments_path": str(segments_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }
