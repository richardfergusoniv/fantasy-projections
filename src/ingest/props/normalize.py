"""Normalize provider payloads into full-fidelity weekly prop quotes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from src.ingest.props.contracts import PLAYER_MARKET_ALIASES, NormalizedQuote
from src.projection.market_quotes import no_vig_probs, scalar_line


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def canonicalize_market(name: str) -> str | None:
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return PLAYER_MARKET_ALIASES.get(key)


def raw_record_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_normalized_quote(
    *,
    source: str,
    sportsbook: str,
    player_name_raw: str,
    market: str,
    line: float,
    fetched_at: datetime,
    over_odds: float | None = None,
    under_odds: float | None = None,
    event_id: str | None = None,
    game_id: str | None = None,
    player_id: str | None = None,
    team: str | None = None,
    opponent: str | None = None,
    period: str = "game",
    event_start: datetime | str | None = None,
    source_url: str | None = None,
    kind: str = "book",
    raw: dict[str, Any] | None = None,
) -> NormalizedQuote:
    novig = no_vig_probs(over_odds, under_odds)
    raw_payload = raw or {
        "source": source,
        "sportsbook": sportsbook,
        "player_name_raw": player_name_raw,
        "market": market,
        "line": line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "event_id": event_id,
        "game_id": game_id,
    }
    return NormalizedQuote(
        source=source,
        sportsbook=sportsbook,
        event_id=event_id,
        game_id=game_id,
        player_name_raw=player_name_raw,
        player_id=player_id,
        team=team,
        opponent=opponent,
        market=market,
        period=period,
        line=float(line),
        over_odds=None if over_odds is None else float(over_odds),
        under_odds=None if under_odds is None else float(under_odds),
        fetched_at=fetched_at if fetched_at.tzinfo else fetched_at.replace(tzinfo=timezone.utc),
        event_start=_parse_dt(event_start),
        source_url=source_url,
        raw_record_hash=raw_record_hash(raw_payload),
        over_prob_novig=None if novig is None else novig[0],
        under_prob_novig=None if novig is None else novig[1],
        kind="projection" if kind == "projection" else "book",
    )


def quotes_from_legacy_player_row(
    *,
    source: str,
    row: dict[str, Any],
    fetched_at: datetime,
    source_url: str | None = None,
    period: str = "game",
    default_sportsbook: str | None = None,
) -> list[NormalizedQuote]:
    """Convert a vegas_raw-style player markets blob into NormalizedQuote rows."""
    name = str(row.get("name") or row.get("player_name") or "").strip()
    if not name:
        return []
    team = row.get("team")
    opponent = row.get("opponent")
    player_id = row.get("player_id") or row.get("gsis_id")
    event_id = row.get("event_id")
    game_id = row.get("game_id")
    event_start = row.get("event_start") or row.get("commence_time")
    markets = dict(row.get("markets") or {})
    quotes: list[NormalizedQuote] = []
    for raw_market, payload in markets.items():
        market = canonicalize_market(str(raw_market))
        if market is None:
            continue
        if not isinstance(payload, dict):
            line = scalar_line(payload)
            if line is None:
                continue
            quotes.append(
                build_normalized_quote(
                    source=source,
                    sportsbook=default_sportsbook or source,
                    player_name_raw=name,
                    market=market,
                    line=line,
                    fetched_at=fetched_at,
                    player_id=None if player_id is None else str(player_id),
                    team=None if team is None else str(team),
                    opponent=None if opponent is None else str(opponent),
                    event_id=None if event_id is None else str(event_id),
                    game_id=None if game_id is None else str(game_id),
                    period=period,
                    event_start=event_start,
                    source_url=source_url,
                    kind="book",
                    raw={"market": raw_market, "payload": payload, "player": name},
                )
            )
            continue
        books = payload.get("books")
        if isinstance(books, dict) and books:
            for book_name, book_raw in books.items():
                if not isinstance(book_raw, dict):
                    line = scalar_line(book_raw)
                    if line is None:
                        continue
                    over_odds = under_odds = None
                else:
                    line = scalar_line(
                        book_raw.get("line")
                        or book_raw.get("value")
                        or book_raw.get("ou")
                        or book_raw.get("total")
                    )
                    if line is None:
                        continue
                    over_odds = scalar_line(
                        book_raw.get("over_odds") or book_raw.get("over_odds_american")
                    )
                    under_odds = scalar_line(
                        book_raw.get("under_odds") or book_raw.get("under_odds_american")
                    )
                quotes.append(
                    build_normalized_quote(
                        source=source,
                        sportsbook=str(book_name),
                        player_name_raw=name,
                        market=market,
                        line=line,
                        over_odds=over_odds,
                        under_odds=under_odds,
                        fetched_at=fetched_at,
                        player_id=None if player_id is None else str(player_id),
                        team=None if team is None else str(team),
                        opponent=None if opponent is None else str(opponent),
                        event_id=None if event_id is None else str(event_id),
                        game_id=None if game_id is None else str(game_id),
                        period=period,
                        event_start=event_start,
                        source_url=source_url,
                        kind="book",
                        raw={
                            "market": raw_market,
                            "book": book_name,
                            "payload": book_raw,
                            "player": name,
                        },
                    )
                )
            continue
        line = scalar_line(
            payload.get("line") or payload.get("value") or payload.get("ou") or payload.get("total")
        )
        projection = scalar_line(
            payload.get("rotowire_proj")
            or payload.get("projection")
            or payload.get("proj")
            or payload.get("projected")
        )
        if line is not None:
            quotes.append(
                build_normalized_quote(
                    source=source,
                    sportsbook=default_sportsbook or source,
                    player_name_raw=name,
                    market=market,
                    line=line,
                    over_odds=scalar_line(
                        payload.get("over_odds") or payload.get("over_odds_american")
                    ),
                    under_odds=scalar_line(
                        payload.get("under_odds") or payload.get("under_odds_american")
                    ),
                    fetched_at=fetched_at,
                    player_id=None if player_id is None else str(player_id),
                    team=None if team is None else str(team),
                    opponent=None if opponent is None else str(opponent),
                    event_id=None if event_id is None else str(event_id),
                    game_id=None if game_id is None else str(game_id),
                    period=period,
                    event_start=event_start,
                    source_url=source_url,
                    kind="book",
                    raw={"market": raw_market, "payload": payload, "player": name},
                )
            )
        elif projection is not None:
            quotes.append(
                build_normalized_quote(
                    source=source,
                    sportsbook=default_sportsbook or source,
                    player_name_raw=name,
                    market=market,
                    line=projection,
                    fetched_at=fetched_at,
                    player_id=None if player_id is None else str(player_id),
                    team=None if team is None else str(team),
                    opponent=None if opponent is None else str(opponent),
                    event_id=None if event_id is None else str(event_id),
                    game_id=None if game_id is None else str(game_id),
                    period=period,
                    event_start=event_start,
                    source_url=source_url,
                    kind="projection",
                    raw={"market": raw_market, "payload": payload, "player": name},
                )
            )
    return quotes
