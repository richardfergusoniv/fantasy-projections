"""Apply Sleeper player status payloads to availability lifecycle."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.app.availability.identity import PlayerIdentityResolver
from src.app.availability.service import AvailabilityService
from src.app.persistence.models import PlayerStatusSnapshot, SourceSnapshot

logger = logging.getLogger(__name__)

INJURY_PLAY_PROBABILITY = {
    "Out": 0.0,
    "Doubtful": 0.25,
    "Questionable": 0.65,
    "Probable": 0.85,
}


class AvailabilitySyncService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.availability = AvailabilityService(session)
        self.resolver = PlayerIdentityResolver(session)

    def resolve_player_id(self, sleeper_id: str, payload: dict[str, Any]) -> str:
        """Stable id for a Sleeper row.

        Resolution is by identifier only. Names are never used here: a namesake
        would otherwise merge two players' availability.
        """

        gsis_id = payload.get("gsis_id")
        resolution = self.resolver.resolve(gsis_id=gsis_id, sleeper_id=sleeper_id)
        if resolution.status == "resolved" and resolution.player_id:
            return resolution.player_id
        if resolution.status == "ambiguous":
            logger.warning(
                "player_identity_ambiguous",
                extra={"candidate_count": len(resolution.candidates), "sleeper_id": sleeper_id},
            )
        return str(gsis_id or sleeper_id)

    def _already_active(self, player_id: str, injury_status: str) -> bool:
        return any(
            event.event_type == "injury_status"
            and (event.policy_json or {}).get("injury_status") == injury_status
            for event in self.availability.repo.active_events(player_id=player_id)
        )

    def sync_from_players_payload(self, players: dict[str, Any], snapshot: SourceSnapshot) -> dict[str, int]:
        activated = 0
        unchanged = 0
        cleared = 0
        clearance = self.availability.clearance_check(snapshot, record_count=len(players))
        for sleeper_id, payload in players.items():
            if not isinstance(payload, dict):
                continue
            player_id = self.resolve_player_id(str(sleeper_id), payload)
            injury_status = payload.get("injury_status")
            self.session.add(
                PlayerStatusSnapshot(
                    player_id=player_id,
                    fetched_at=datetime.now(UTC),
                    status=payload.get("status"),
                    injury_status=injury_status,
                    practice=payload.get("practice_participation"),
                    raw_json={
                        **payload,
                        "resolved_player_id": player_id,
                        "sleeper_id": str(sleeper_id),
                    },
                )
            )
            if injury_status in INJURY_PLAY_PROBABILITY:
                if self._already_active(player_id, injury_status):
                    unchanged += 1
                    continue
                self.availability.activate_event(
                    player_id=player_id,
                    event_type="injury_status",
                    source_snapshot_id=snapshot.id,
                    evidence_ids=[],
                    policy={
                        "play_probability": INJURY_PLAY_PROBABILITY[injury_status],
                        "injury_status": injury_status,
                        "source_endpoint": snapshot.endpoint,
                    },
                )
                activated += 1
            elif not injury_status and clearance.allowed:
                cleared += self.availability.try_clear_for_player(
                    player_id,
                    snapshot,
                    player_count=len(players),
                )
        self.session.flush()
        return {
            "activated": activated,
            "unchanged": unchanged,
            "cleared": cleared,
            "player_count": len(players),
            "clearing_allowed": clearance.allowed,
            "clearing_reason": clearance.reason,
        }
