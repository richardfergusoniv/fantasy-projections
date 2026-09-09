"""Provenance helpers for weekly_props decisions and publication."""

from __future__ import annotations

from typing import Any

#: Decisions currently consume weekly_props as points/quantile draws only.
WEEKLY_PROPS_SCORING_FIDELITY = "weekly_props_points_only"
WEEKLY_PROPS_CAPABILITY_MODE = "weekly_props_market"


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


def weekly_props_context_fields(
    *,
    run: Any | None = None,
    week: int | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Scoring fidelity / capability / caveats for an active weekly_props run."""
    caveats = [
        "weekly_props_points_only_draws",
        "no_component_rescore_for_weekly_props",
        "no_joint_stat_draws_for_weekly_props",
    ]
    if fallback_reason:
        caveats.append(f"source_fallback:{fallback_reason}")
    fidelity = WEEKLY_PROPS_SCORING_FIDELITY
    capability = WEEKLY_PROPS_CAPABILITY_MODE
    if run is not None and is_weekly_props_run(run):
        meta = getattr(run, "metadata_json", None) or {}
        if isinstance(meta, dict):
            if meta.get("scoring_fidelity"):
                fidelity = str(meta["scoring_fidelity"])
            if meta.get("capability_mode"):
                capability = str(meta["capability_mode"])
            if meta.get("joint_draws") or meta.get("component_draws"):
                # Only claim richer fidelity when the artifact itself declares it.
                caveats = [c for c in caveats if "no_joint" not in c and "no_component" not in c]
    return {
        "scoring_fidelity": fidelity,
        "capability_mode": capability,
        "caveats": caveats,
        "week": week,
        "projection_run_id": getattr(run, "id", None) if run is not None else None,
        "model_version": getattr(run, "model_version", None) if run is not None else None,
        "artifact_mode": getattr(run, "artifact_mode", None) if run is not None else None,
    }
