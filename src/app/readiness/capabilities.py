"""Capability-specific GO/NO-GO readiness matrix.

Replaces a single global readiness verdict with independent capability decisions.
A weekly R&D NO-GO must not mark production capabilities unhealthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app.projections.loader import ReleaseBundleLoader, ReleaseBundleLoadError
from src.app.projections.source import ProjectionSource, configured_projection_source, weekly_rnd_enabled
from src.app.projections.weekly_v2_bridge import weekly_v2_readiness
from src.projection.active_release import read_active_pointer


@dataclass(frozen=True)
class CapabilityStatus:
    capability: str
    verdict: str  # GO | NO-GO | DEGRADED
    source: str
    detail: str
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "verdict": self.verdict,
            "source": self.source,
            "detail": self.detail,
            "caveats": list(self.caveats),
        }


@dataclass
class CapabilityMatrix:
    capabilities: list[CapabilityStatus] = field(default_factory=list)
    production_healthy: bool = False
    production_degraded: list[str] = field(default_factory=list)
    weekly_rnd_healthy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "production_healthy": self.production_healthy,
            "production_degraded": list(self.production_degraded),
            "weekly_rnd_healthy": self.weekly_rnd_healthy,
        }

    def by_name(self) -> dict[str, CapabilityStatus]:
        return {cap.capability: cap for cap in self.capabilities}


def _verdict(passed: bool, *, degraded: bool = False) -> str:
    if passed:
        return "GO"
    if degraded:
        return "DEGRADED"
    return "NO-GO"


def build_capability_matrix(
    session: Session | None,
    *,
    season: int = 2026,
    week: int = 1,
) -> CapabilityMatrix:
    settings = get_settings()
    matrix = CapabilityMatrix()
    caveats: list[str] = []

    # --- Sealed production release ---
    sealed_go = False
    sealed_detail = "no active pointer"
    try:
        loader = ReleaseBundleLoader(season=season)
        bundle = loader.load_bundle()
        if bundle is not None and bundle.players:
            sealed_go = bundle.validation_passed
            sealed_detail = (
                f"namespace={bundle.namespace} release={bundle.release_id[:8]}… "
                f"players={len(bundle.players)}"
            )
            if bundle.caveats:
                caveats.extend(bundle.caveats)
        elif bundle is not None:
            sealed_detail = "bundle loaded but empty player index"
    except ReleaseBundleLoadError as exc:
        sealed_detail = str(exc)

    matrix.capabilities.append(
        CapabilityStatus(
            capability="draft_rankings_and_roster_values",
            verdict=_verdict(sealed_go),
            source=ProjectionSource.SEALED_RELEASE.value,
            detail=sealed_detail,
            caveats=tuple(caveats),
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="dynasty_values_and_trade_foundations",
            verdict=_verdict(sealed_go and session is not None),
            source=ProjectionSource.SEALED_RELEASE.value,
            detail=sealed_detail,
            caveats=tuple(caveats),
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="league_specific_season_projections",
            verdict=_verdict(sealed_go and session is not None),
            source="sealed_component_rescore",
            detail="requires sealed bundle + live league rule snapshots",
            caveats=tuple(caveats),
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="waiver_baseline",
            verdict=_verdict(sealed_go),
            source=ProjectionSource.SEALED_RELEASE.value,
            detail="production values + roster availability; breakout signal limited",
        )
    )

    # --- Status overlay ---
    overlay_go = False
    overlay_detail = "no active overlay pointer"
    overlay_pub_go = False
    overlay_pub_detail = "overlay publication not configured"
    if session is not None:
        from src.app.projections.status_overlay import active_overlay_status

        overlay_status = active_overlay_status(session, season=season)
        overlay_go = overlay_status.get("active", False)
        overlay_detail = overlay_status.get("detail", overlay_detail)
        overlay_pub_go = overlay_status.get("publication_allowed", False)
        overlay_pub_detail = overlay_status.get("publication_detail", overlay_pub_detail)

    matrix.capabilities.append(
        CapabilityStatus(
            capability="daily_injury_depth_adjustments",
            verdict=_verdict(overlay_go, degraded=not overlay_go and sealed_go),
            source=ProjectionSource.STATUS_ADJUSTED_RELEASE.value,
            detail=overlay_detail,
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="status_overlay_auto_publication",
            verdict=_verdict(overlay_pub_go),
            source="status_overlay_gate",
            detail=overlay_pub_detail,
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="season_baseline_start_sit_advisory",
            verdict=_verdict(sealed_go, degraded=overlay_go),
            source=(
                ProjectionSource.STATUS_ADJUSTED_RELEASE.value
                if overlay_go
                else ProjectionSource.SEALED_RELEASE.value
            ),
            detail="labeled season-baseline comparison; not matchup-specific weekly model",
            caveats=("not_matchup_specific_weekly_projection",),
        )
    )

    # --- Weekly R&D ---
    readiness = weekly_v2_readiness(season, week)
    weekly_pub_go = readiness.auto_publish_allowed
    matchup_go = readiness.state == "trained" and readiness.auto_publish_allowed

    matrix.capabilities.append(
        CapabilityStatus(
            capability="matchup_specific_weekly_start_sit_win_probability",
            verdict=_verdict(matchup_go),
            source="weekly_v2_rnd",
            detail=f"state={readiness.state} auto_publish={readiness.auto_publish_allowed}",
            caveats=tuple(readiness.reasons),
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="weekly_v2_event_joint_auto_publication",
            verdict=_verdict(weekly_pub_go),
            source="weekly_v2_rnd",
            detail=f"state={readiness.state}",
            caveats=tuple(readiness.reasons),
        )
    )
    matrix.capabilities.append(
        CapabilityStatus(
            capability="weekly_v2_rnd_access",
            verdict=_verdict(weekly_rnd_enabled(), degraded=not weekly_rnd_enabled()),
            source="weekly_v2_rnd",
            detail=f"WEEKLY_RND_ENABLED={weekly_rnd_enabled()}",
        )
    )

    # --- Infrastructure ---
    private_go = sealed_go
    if session is not None:
        try:
            from src.app.persistence.models import League

            league_count = session.query(League).filter(League.season == season).count()
            private_go = sealed_go and league_count > 0
        except Exception:  # noqa: BLE001
            private_go = False

    matrix.capabilities.append(
        CapabilityStatus(
            capability="private_core_app_beta",
            verdict=_verdict(private_go),
            source=configured_projection_source().value,
            detail="sealed bundle + league sync + auth + persistence",
        )
    )

    public_problems = settings.production_config_problems()
    matrix.capabilities.append(
        CapabilityStatus(
            capability="public_internet_deployment",
            verdict=_verdict(not public_problems and settings.app_env == "production"),
            source="infrastructure",
            detail=(
                "all production config checks pass"
                if not public_problems
                else f"{len(public_problems)} blocking config issue(s)"
            ),
            caveats=tuple(public_problems[:5]),
        )
    )

    production_caps = {
        "draft_rankings_and_roster_values",
        "dynasty_values_and_trade_foundations",
        "league_specific_season_projections",
        "waiver_baseline",
        "season_baseline_start_sit_advisory",
        "private_core_app_beta",
    }
    production_statuses = [
        cap for cap in matrix.capabilities if cap.capability in production_caps
    ]
    matrix.production_degraded = [
        cap.capability for cap in production_statuses if cap.verdict == "DEGRADED"
    ]
    matrix.production_healthy = all(cap.verdict == "GO" for cap in production_statuses)
    matrix.weekly_rnd_healthy = readiness.state == "trained" and readiness.auto_publish_allowed

    pointer = read_active_pointer(season)
    if pointer is not None:
        matrix.capabilities.append(
            CapabilityStatus(
                capability="sealed_season_projection_source",
                verdict=_verdict(sealed_go),
                source=pointer.get("namespace", "unknown"),
                detail=sealed_detail,
            )
        )

    return matrix
