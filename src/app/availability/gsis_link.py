"""Link Sleeper-keyed identities to GSIS ids used by projection releases.

Sleeper's ``players/nfl`` payload omits ``gsis_id`` for many active players. When
that happens, ``upsert_player_identities`` stores the Sleeper id as
``player_identity.player_id``. Projection releases and ``player_projection`` rows
are keyed by GSIS, so decisions that look up by the stored id find nothing and
fail closed with ``no_projected_players_on_roster``.

This module fills ``gsis_id`` by an unambiguous name+position(+team) match
against the sealed release roster. Ambiguous namesakes are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.app.availability.identity import normalize_name
from src.app.logging import get_logger
from src.app.persistence.models import PlayerIdentity

logger = get_logger(__name__)


@dataclass(frozen=True)
class GsisLinkStats:
    linked: int = 0
    already_linked: int = 0
    ambiguous: int = 0
    unmatched: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "linked": self.linked,
            "already_linked": self.already_linked,
            "ambiguous": self.ambiguous,
            "unmatched": self.unmatched,
        }


def _is_gsis_shaped(player_id: str | None) -> bool:
    return bool(player_id) and str(player_id).startswith("00-")


def projection_id_for_identity(row: PlayerIdentity) -> str:
    """Return the id space projections use for this identity row."""
    if row.gsis_id:
        return str(row.gsis_id)
    if _is_gsis_shaped(row.player_id):
        return str(row.player_id)
    return str(row.player_id)


def build_release_gsis_index(
    players: dict[str, object],
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str, str], list[str]]]:
    """Index release players by (name, position) and (name, position, team)."""
    by_name_pos: dict[tuple[str, str], list[str]] = {}
    by_name_pos_team: dict[tuple[str, str, str], list[str]] = {}
    for player_id, summary in players.items():
        name = normalize_name(str(getattr(summary, "name", "") or ""))
        position = str(getattr(summary, "position", "") or "").upper()
        team = str(getattr(summary, "team", "") or "").upper()
        if not name or not position:
            continue
        by_name_pos.setdefault((name, position), []).append(str(player_id))
        if team:
            by_name_pos_team.setdefault((name, position, team), []).append(str(player_id))
    return by_name_pos, by_name_pos_team


def match_gsis_for_identity(
    row: PlayerIdentity,
    *,
    by_name_pos: dict[tuple[str, str], list[str]],
    by_name_pos_team: dict[tuple[str, str], list[str]] | dict[tuple[str, str, str], list[str]],
) -> str | None:
    """Return a unique GSIS id for ``row``, or None when unmatched/ambiguous."""
    if row.gsis_id:
        return str(row.gsis_id)
    if _is_gsis_shaped(row.player_id):
        return str(row.player_id)

    name = normalize_name(row.name or "")
    position = str(row.position or "").upper()
    team = str(row.team or "").upper()
    if not name or not position:
        return None

    candidates: list[str] = []
    if team:
        candidates = list(by_name_pos_team.get((name, position, team), []))
    if len(candidates) != 1:
        candidates = list(by_name_pos.get((name, position), []))
    if len(candidates) == 1:
        return candidates[0]
    return None


def link_identities_to_release_players(
    session: Session,
    players: dict[str, object],
) -> GsisLinkStats:
    """Persist GSIS links onto Sleeper-keyed identity rows when unambiguous."""
    if not players:
        return GsisLinkStats()

    by_name_pos, by_name_pos_team = build_release_gsis_index(players)
    linked = already = ambiguous = unmatched = 0

    rows = session.query(PlayerIdentity).all()
    for row in rows:
        if row.gsis_id or _is_gsis_shaped(row.player_id):
            already += 1
            continue
        name = normalize_name(row.name or "")
        position = str(row.position or "").upper()
        team = str(row.team or "").upper()
        if not name or not position:
            unmatched += 1
            continue

        team_matches = (
            list(by_name_pos_team.get((name, position, team), [])) if team else []
        )
        name_matches = list(by_name_pos.get((name, position), []))
        if len(team_matches) == 1:
            chosen = team_matches[0]
        elif len(name_matches) == 1:
            chosen = name_matches[0]
        elif len(team_matches) > 1 or len(name_matches) > 1:
            ambiguous += 1
            continue
        else:
            unmatched += 1
            continue

        row.gsis_id = chosen
        # Prefer the GSIS row as the sleeper linkage target when it already exists.
        gsis_row = session.get(PlayerIdentity, chosen)
        if (
            gsis_row is not None
            and row.sleeper_id
            and (not gsis_row.sleeper_id or gsis_row.sleeper_id == row.sleeper_id)
        ):
            gsis_row.sleeper_id = row.sleeper_id
        linked += 1

    if linked:
        session.flush()
        logger.info(
            "sleeper_gsis_identity_linked",
            linked=linked,
            ambiguous=ambiguous,
            unmatched=unmatched,
        )
    return GsisLinkStats(
        linked=linked,
        already_linked=already,
        ambiguous=ambiguous,
        unmatched=unmatched,
    )
