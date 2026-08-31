"""Resolve availability evidence to a stable player id.

Name matching is the main way evidence gets attached to the wrong player, so a
name that matches more than one identity resolves to ``ambiguous`` and the
caller must quarantine rather than guess.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.app.persistence.models import PlayerIdentity

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")


def normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _PUNCTUATION.sub(" ", ascii_only.lower())
    parts = [part for part in cleaned.split() if part and part not in _SUFFIXES]
    return " ".join(parts)


@dataclass(frozen=True)
class IdentityResolution:
    status: str  # resolved | ambiguous | unknown
    player_id: str | None = None
    reason: str | None = None
    candidates: list[str] = field(default_factory=list)


class PlayerIdentityResolver:
    def __init__(self, session: Session) -> None:
        self.session = session

    def resolve(
        self,
        *,
        player_id: str | None = None,
        gsis_id: str | None = None,
        sleeper_id: str | None = None,
        name: str | None = None,
        team: str | None = None,
        position: str | None = None,
    ) -> IdentityResolution:
        for stable_id, column in ((player_id, PlayerIdentity.player_id), (gsis_id, PlayerIdentity.gsis_id), (sleeper_id, PlayerIdentity.sleeper_id)):
            if not stable_id:
                continue
            rows = self.session.query(PlayerIdentity).filter(column == str(stable_id)).all()
            if len(rows) == 1:
                return IdentityResolution(status="resolved", player_id=rows[0].player_id)
            if len(rows) > 1:
                return IdentityResolution(
                    status="ambiguous",
                    reason="stable_id_maps_to_multiple_identities",
                    candidates=sorted(row.player_id for row in rows),
                )
        if not name:
            return IdentityResolution(status="unknown", reason="no_identifier_supplied")

        target = normalize_name(name)
        if not target:
            return IdentityResolution(status="unknown", reason="empty_name")
        matches = [row for row in self.session.query(PlayerIdentity).all() if normalize_name(row.name) == target]
        if not matches:
            return IdentityResolution(status="unknown", reason="name_not_in_identity_registry")
        narrowed = matches
        if team:
            narrowed = [row for row in narrowed if (row.team or "").upper() == team.upper()]
        if position:
            narrowed = [row for row in narrowed if (row.position or "").upper() == position.upper()]
        if len(narrowed) == 1:
            return IdentityResolution(status="resolved", player_id=narrowed[0].player_id)
        if not narrowed:
            return IdentityResolution(
                status="unknown",
                reason="name_matched_but_team_or_position_conflicts",
                candidates=sorted(row.player_id for row in matches),
            )
        return IdentityResolution(
            status="ambiguous",
            reason="namesake_requires_team_or_position",
            candidates=sorted(row.player_id for row in narrowed),
        )
