"""Unified projection service — production integration facade.

Every feature resolves projections through this service so source boundaries,
overlay versions, and scoring fidelity stay explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from src.app.projections.loader import PlayerSummary, ReleaseBundleLoader, ReleaseBundleLoadError
from src.app.projections.source import (
    ProjectionSource,
    configured_projection_source,
    resolve_effective_source,
    weekly_rnd_enabled,
)
from src.app.projections.status_overlay import load_overlay_players, read_active_overlay


@dataclass(frozen=True)
class ProjectionContext:
    source: ProjectionSource
    season: int
    base_release_id: str | None
    base_manifest_sha256: str | None
    overlay_hash: str | None
    as_of: str
    scoring_fidelity: str
    capability_mode: str
    caveats: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_source": self.source.value,
            "base_release_id": self.base_release_id,
            "base_manifest_sha256": self.base_manifest_sha256,
            "overlay_version": self.overlay_hash,
            "as_of": self.as_of,
            "scoring_fidelity": self.scoring_fidelity,
            "capability_mode": self.capability_mode,
            "caveats": list(self.caveats),
            "provenance": self.provenance,
        }


class ProjectionService:
    """Resolve player projections from the configured production source."""

    def __init__(self, session: Session | None = None, *, season: int = 2026) -> None:
        self.session = session
        self.season = season
        self._loader = ReleaseBundleLoader(season=season)

    def effective_source(self, requested: ProjectionSource | None = None) -> ProjectionSource:
        return resolve_effective_source(requested)

    def context(
        self,
        *,
        requested_source: ProjectionSource | None = None,
        league_id: str | None = None,
    ) -> ProjectionContext:
        source = self.effective_source(requested_source)
        bundle = self._loader.load_bundle()
        overlay_pointer = read_active_overlay(self.season)

        base_release_id = bundle.release_id if bundle else None
        base_manifest = bundle.manifest_sha256 if bundle else None
        overlay_hash = None
        caveats = list(bundle.caveats) if bundle else []
        scoring_fidelity = "baseline_points_only"
        capability_mode = "season_baseline"

        if source == ProjectionSource.STATUS_ADJUSTED_RELEASE and overlay_pointer:
            overlay_hash = str(overlay_pointer.get("overlay_hash") or "")
            caveats.append("status_adjusted_season_baseline")
            scoring_fidelity = "status_adjusted_baseline"
            capability_mode = "status_adjusted_advisory"
        elif source == ProjectionSource.WEEKLY_V2_RND:
            caveats.append("weekly_v2_experimental_not_production")
            scoring_fidelity = "weekly_v2_rnd"
            capability_mode = "weekly_rnd"

        if league_id and self.session is not None and bundle and bundle.component_projections_path:
            from src.app.persistence.models import League, LeagueRuleSnapshot
            from src.app.projections.league_rescore import load_component_projections, rescore_league
            from src.app.scoring.compiler import scoring_settings_from_snapshot

            snapshot = (
                self.session.query(LeagueRuleSnapshot)
                .filter(LeagueRuleSnapshot.league_id == league_id)
                .order_by(LeagueRuleSnapshot.fetched_at.desc())
                .first()
            )
            league = self.session.query(League).filter(League.league_id == league_id).one_or_none()
            if snapshot is not None and league is not None:
                components = load_component_projections(bundle.component_projections_path)
                if components:
                    raw = snapshot.raw_json or {}
                    result = rescore_league(
                        league_id=league_id,
                        display_name=league.name or league_id,
                        scoring_settings=scoring_settings_from_snapshot(raw),
                        roster_positions=list((league.raw_json or {}).get("roster_positions") or []),
                        components_by_player=components,
                    )
                    scoring_fidelity = result.scoring_fidelity
                    if result.approximate_rules:
                        caveats.extend(result.approximate_rules)

        provenance = bundle.provenance() if bundle else {"validation_passed": False}
        provenance["requested_source"] = (requested_source or configured_projection_source()).value

        return ProjectionContext(
            source=source,
            season=self.season,
            base_release_id=base_release_id,
            base_manifest_sha256=base_manifest,
            overlay_hash=overlay_hash,
            as_of=self._loader.as_of(),
            scoring_fidelity=scoring_fidelity,
            capability_mode=capability_mode,
            caveats=tuple(caveats),
            provenance=provenance,
        )

    def players(
        self,
        *,
        requested_source: ProjectionSource | None = None,
    ) -> dict[str, PlayerSummary]:
        source = self.effective_source(requested_source)
        if source == ProjectionSource.WEEKLY_V2_RND:
            if not weekly_rnd_enabled():
                raise ValueError("weekly_v2_rnd is not enabled")
            raise NotImplementedError(
                "weekly_v2_rnd player resolution is experimental; use sealed_release"
            )

        if source == ProjectionSource.STATUS_ADJUSTED_RELEASE:
            overlay_players = load_overlay_players(self.season)
            if overlay_players:
                return overlay_players

        bundle = self._loader.load_bundle()
        if bundle is None:
            return {}
        return bundle.players

    def get(self, player_id: str, *, requested_source: ProjectionSource | None = None) -> PlayerSummary | None:
        return self.players(requested_source=requested_source).get(player_id)

    def matchup_win_probability_allowed(self, *, season: int | None = None, week: int = 1) -> bool:
        from src.app.projections.weekly_v2_bridge import weekly_v2_readiness

        readiness = weekly_v2_readiness(season or self.season, week)
        return readiness.state == "trained" and readiness.auto_publish_allowed

    @staticmethod
    def ensure_bundle_available(season: int = 2026) -> None:
        """Fail fast when the sealed bundle cannot be loaded (startup probe)."""
        loader = ReleaseBundleLoader(season=season)
        try:
            bundle = loader.load_bundle()
        except ReleaseBundleLoadError:
            bundle = None
        if bundle is None or not bundle.players:
            return
