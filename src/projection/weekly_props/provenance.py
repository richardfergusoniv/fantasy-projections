"""Provenance helpers for weekly_props decisions and publication."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.projection.weekly_props.config import DEFAULT_WEEKLY_POLICY

#: Decisions currently consume weekly_props as points/quantile draws only.
WEEKLY_PROPS_SCORING_FIDELITY = "weekly_props_points_only"
WEEKLY_PROPS_CAPABILITY_MODE = "weekly_props_market"
#: Dedicated pointer so promoting Vegas Props does not replace weekly-v2 R&D.
WEEKLY_PROPS_POINTER_MODE = "weekly_props"


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
    mode = str(getattr(run, "mode", "") or "")
    return (
        mode == WEEKLY_PROPS_POINTER_MODE
        or model_version.startswith("weekly_props")
        or artifact_mode == "market"
    )


def load_promoted_weekly_props_run(session: Any, *, season: int, week: int) -> Any | None:
    """Return the active weekly_props run for ``season``/``week``, if any.

    Prefers the dedicated ``weekly_props`` pointer so a weekly-v2 R&D pointer
    occupying ``mode=weekly`` cannot masquerade as Vegas Props. Falls back to a
    ``weekly`` pointer that actually is a weekly_props artifact (older tests /
    the pre-split contract).
    """
    from src.app.persistence.models import ProjectionRun
    from src.app.releases.publication import active_pointer

    for mode in (WEEKLY_PROPS_POINTER_MODE, "weekly"):
        pointer = active_pointer(session, mode=mode, season=season, week=week)
        if pointer is None:
            continue
        run = (
            session.query(ProjectionRun)
            .filter(ProjectionRun.id == pointer.run_id)
            .one_or_none()
        )
        if run is not None and is_weekly_props_run(run):
            return run
    return None


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
    snapshot_age_hours: float | None = None
    as_of = getattr(run, "as_of", None) if run is not None else None
    if isinstance(as_of, datetime):
        clock = datetime.now(UTC)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        snapshot_age_hours = round((clock - as_of.astimezone(UTC)).total_seconds() / 3600.0, 3)
        caveats.append(f"snapshot_age_hours:{snapshot_age_hours:.1f}")
        if snapshot_age_hours > DEFAULT_WEEKLY_POLICY.max_snapshot_age_hours:
            caveats.append(f"stale_closing_line:{snapshot_age_hours:.1f}h")
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
        "snapshot_age_hours": snapshot_age_hours,
    }
