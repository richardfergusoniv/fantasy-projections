"""Provenance helpers for weekly_props decisions and publication."""

from __future__ import annotations

from typing import Any


def decision_provenance(
    *,
    requested_source: str,
    effective_source: str,
    run_id: str | None,
    season: int,
    week: int | None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "requested_source": requested_source,
        "effective_source": effective_source,
        "projection_run_id": run_id,
        "season": season,
        "week": week,
        "fallback_reason": fallback_reason,
    }


def is_weekly_props_run(run: Any) -> bool:
    if run is None:
        return False
    model_version = str(getattr(run, "model_version", "") or "")
    artifact_mode = str(getattr(run, "artifact_mode", "") or "")
    return model_version.startswith("weekly_props") or artifact_mode == "market"
