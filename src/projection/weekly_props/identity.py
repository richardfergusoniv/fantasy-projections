"""Canonicalize provider quote identities before consensus grouping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.app.availability.identity import normalize_name
from src.draft_assistant.market_adp import canonicalize_player_name
from src.ingest.props.contracts import NormalizedQuote


def _lookup(
    identity_map: Mapping[str, dict[str, Any]], *keys: str | None
) -> dict[str, Any] | None:
    for key in keys:
        if not key:
            continue
        hit = identity_map.get(key)
        if hit:
            return hit
    return None


def resolve_quote_group_key(
    quote: NormalizedQuote,
    identity_map: Mapping[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Return ``(group_key, identity)`` for one quote.

    Quotes that resolve to the same canonical player id share a group even when
    one provider supplies only a name. Unresolved identities use a deterministic
    key that includes team/position so distinct same-name players do not merge.
    Ambiguous name-only identities stay separated rather than guessing.
    """
    canon = canonicalize_player_name(quote.player_name_raw or "")
    norm = normalize_name(quote.player_name_raw or "")
    team = str(quote.team or "").upper()
    position = str(getattr(quote, "position", None) or "").upper()

    if quote.player_id:
        ident = _lookup(identity_map, quote.player_id)
        if ident and ident.get("player_id"):
            return str(ident["player_id"]), ident
        # Provider id may already be canonical even if absent from the map.
        return str(quote.player_id), {
            "player_id": quote.player_id,
            "name": quote.player_name_raw,
            "team": quote.team,
            "opponent": quote.opponent,
            "position": getattr(quote, "position", None),
        }

    ident = _lookup(
        identity_map,
        f"{canon}|{team}|{position}" if canon and team and position else None,
        f"{canon}|{team}" if canon and team else None,
        f"{norm}|{team}" if norm and team else None,
        canon if canon else None,
        norm if norm else None,
        (quote.player_name_raw or "").strip().lower(),
    )
    if ident and ident.get("player_id"):
        # Name-only unique resolution: merge onto the canonical player.
        if not team or not ident.get("team") or str(ident.get("team")).upper() == team:
            return str(ident["player_id"]), ident
        # Team conflict → do not trust the name hit.
        ident = None

    # Deterministic unresolved key: same name+team(+pos) can still merge across
    # books, but different teams never collapse together.
    label = canon or norm or (quote.player_name_raw or "").strip().lower() or "unknown"
    parts = ["unresolved", label]
    if team:
        parts.append(team)
    if position:
        parts.append(position)
    return "|".join(parts), {
        "player_id": None,
        "name": quote.player_name_raw,
        "team": quote.team,
        "opponent": quote.opponent,
        "position": getattr(quote, "position", None),
    }
