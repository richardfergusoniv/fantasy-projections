"""Live BettingPros NFL weekly player O/U fetch.

Uses the public ``api.bettingpros.com/v3`` JSON endpoints that the BettingPros
web UI calls (same capture surface as the draft-assistant vegas_raw dump).

Endpoints
---------
- ``GET /v3/events?sport=NFL&season={season}&week={week}``
- ``GET /v3/offers?sport=NFL&market_id={ids}&event_id={id}&book_id=0&limit=10&page=N``
- ``GET /v3/markets?sport=NFL`` (catalog reference; market ids are pinned below)

Auth is the site's browser-embedded ``x-api-key`` (public client key, not a
repo secret — same pattern as FanDuel's ``_ak`` query param). ``api.bettingpros.com``
robots.txt disallows crawling; see ``docs/ops/BETTINGPROS_LIVE_WEEKLY_SCRAPE.md``.

Role 1 emits BettingPros **consensus** (``book_id=0``) as sportsbook
``bettingpros`` so we do not double-count DraftKings/FanDuel or blend
prediction markets into Role 1/2 means.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlencode

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.normalize import build_normalized_quote
from src.ingest.props.providers.http import LiveFetchError, fetch_json

BP_API_BASE = "https://api.bettingpros.com/v3"
BP_SITE = "https://www.bettingpros.com"
# Public client key embedded in the BettingPros web app (not a private secret).
BP_PUBLIC_API_KEY = "CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh"

# Weekly game player-prop market_id → canonical market.
# Intentionally omits market 334 ("touchdowns") — combined TD O/U, not rush/rec.
WEEKLY_MARKET_IDS: tuple[tuple[int, str], ...] = (
    (103, "pass_yards"),
    (102, "pass_tds"),
    (333, "pass_attempts"),
    (100, "pass_completions"),
    (101, "pass_ints"),
    (107, "rush_yards"),
    (106, "rush_attempts"),
    (105, "rec_yards"),
    (104, "receptions"),
)

WEEKLY_MARKET_ID_MAP: dict[int, str] = {mid: key for mid, key in WEEKLY_MARKET_IDS}
CONSENSUS_BOOK_ID = 0
OFFERS_PAGE_LIMIT = 10
# Closed/complete events are post-kickoff — skip for Role 1 boards.
_SKIP_EVENT_STATUSES = frozenset(
    {"closed", "complete", "completed", "final", "cancelled", "canceled", "postponed"}
)


def _bp_headers(*, referer: str | None = None) -> dict[str, str]:
    return {
        "Origin": BP_SITE,
        "Referer": referer or f"{BP_SITE}/nfl/odds/player-props/",
        "x-api-key": BP_PUBLIC_API_KEY,
    }


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # BettingPros often returns "YYYY-MM-DD HH:MM:SS" (naive UTC).
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1) + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _main_line(book_entry: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Return (line, american cost) for the main active line on a book entry."""
    if not book_entry:
        return None, None
    for line in book_entry.get("lines") or []:
        if not isinstance(line, dict):
            continue
        if line.get("is_off") or line.get("active") is False:
            continue
        if line.get("main") is False:
            continue
        try:
            value = float(line["line"])
        except (KeyError, TypeError, ValueError):
            continue
        cost_raw = line.get("cost")
        try:
            cost = None if cost_raw is None else float(cost_raw)
        except (TypeError, ValueError):
            cost = None
        return value, cost
    return None, None


def _selection_book(
    selection: dict[str, Any], *, book_id: int = CONSENSUS_BOOK_ID
) -> dict[str, Any] | None:
    for book in selection.get("books") or []:
        if not isinstance(book, dict):
            continue
        try:
            if int(book.get("id")) == book_id:
                return book
        except (TypeError, ValueError):
            continue
    return None


def parse_offer(
    offer: dict[str, Any],
    *,
    event: dict[str, Any] | None,
    fetched_at: datetime,
    source_url: str | None = None,
) -> NormalizedQuote | None:
    """Parse one BettingPros offer into a consensus NormalizedQuote."""
    try:
        market_id = int(offer.get("market_id"))
    except (TypeError, ValueError):
        return None
    market = WEEKLY_MARKET_ID_MAP.get(market_id)
    if market is None:
        return None
    if offer.get("active") is False:
        return None

    participants = offer.get("participants") or []
    player_name = None
    team = None
    player_id = offer.get("player_id")
    for part in participants:
        if not isinstance(part, dict):
            continue
        player_name = str(part.get("name") or "").strip() or player_name
        player = part.get("player") or {}
        if isinstance(player, dict):
            team = player.get("team") or team
            if player_id is None:
                player_id = player.get("id") or part.get("id")
        if player_name:
            break
    if not player_name:
        return None

    over_line = under_line = None
    over_odds = under_odds = None
    for sel in offer.get("selections") or []:
        if not isinstance(sel, dict) or sel.get("active") is False:
            continue
        side = str(sel.get("selection") or sel.get("label") or "").strip().lower()
        book = _selection_book(sel, book_id=CONSENSUS_BOOK_ID)
        line, cost = _main_line(book)
        if line is None:
            continue
        if side.startswith("over"):
            over_line, over_odds = line, cost
        elif side.startswith("under"):
            under_line, under_odds = line, cost

    if over_line is None and under_line is None:
        return None
    if (
        over_line is not None
        and under_line is not None
        and abs(over_line - under_line) > 1e-9
    ):
        # Consensus sometimes posts mismatched O/U rungs — skip rather than guess.
        return None
    line = over_line if over_line is not None else under_line
    assert line is not None

    home = away = None
    event_id = offer.get("event_id")
    event_start = None
    if event:
        home = event.get("home")
        away = event.get("visitor") or event.get("away")
        event_start = _parse_dt(event.get("scheduled"))
        if event_id is None:
            event_id = event.get("id")

    opponent = None
    if team and home and away:
        team_u = str(team).upper()
        home_u = str(home).upper()
        away_u = str(away).upper()
        if team_u == home_u:
            opponent = away_u
        elif team_u == away_u:
            opponent = home_u

    return build_normalized_quote(
        source="bettingpros",
        sportsbook="bettingpros",
        player_name_raw=player_name,
        market=market,
        line=float(line),
        over_odds=over_odds,
        under_odds=under_odds,
        fetched_at=fetched_at,
        event_id=None if event_id is None else str(event_id),
        game_id=None if event_id is None else str(event_id),
        player_id=None if player_id is None else str(player_id),
        team=None if team is None else str(team).upper(),
        opponent=opponent,
        period="game",
        event_start=event_start,
        source_url=source_url,
        kind="book",
        raw={
            "market_id": market_id,
            "offer_id": offer.get("id"),
            "book_id": CONSENSUS_BOOK_ID,
        },
    )


def parse_offers(
    offers: Iterable[dict[str, Any]],
    *,
    events_by_id: dict[str, dict[str, Any]],
    fetched_at: datetime,
    source_url: str | None = None,
) -> list[NormalizedQuote]:
    quotes: list[NormalizedQuote] = []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        event = None
        eid = offer.get("event_id")
        if eid is not None:
            event = events_by_id.get(str(eid))
        quote = parse_offer(
            offer, event=event, fetched_at=fetched_at, source_url=source_url
        )
        if quote is not None:
            quotes.append(quote)
    return quotes


def _events_url(*, season: int, week: int) -> str:
    return (
        f"{BP_API_BASE}/events?"
        + urlencode({"sport": "NFL", "season": season, "week": week})
    )


def _offers_url(
    *,
    event_id: int | str,
    market_ids: str,
    page: int = 1,
) -> str:
    return (
        f"{BP_API_BASE}/offers?"
        + urlencode(
            {
                "sport": "NFL",
                "market_id": market_ids,
                "event_id": str(event_id),
                "book_id": str(CONSENSUS_BOOK_ID),
                "limit": str(OFFERS_PAGE_LIMIT),
                "page": str(page),
            }
        )
    )


def fetch_events(*, season: int, week: int) -> list[dict[str, Any]]:
    url = _events_url(season=season, week=week)
    payload = fetch_json(url, headers=_bp_headers(), prefer_curl_cffi=True)
    if not isinstance(payload, dict):
        raise LiveFetchError(f"unexpected events payload type from {url}")
    events = payload.get("events") or []
    if not isinstance(events, list):
        raise LiveFetchError(f"events is not a list from {url}")
    return [e for e in events if isinstance(e, dict)]


def fetch_offers_for_event(
    event_id: int | str,
    *,
    market_ids: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Paginate consensus offers for one event across weekly market ids."""
    mid = market_ids or ":".join(str(m) for m, _ in WEEKLY_MARKET_IDS)
    urls: list[str] = []
    offers: list[dict[str, Any]] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        url = _offers_url(event_id=event_id, market_ids=mid, page=page)
        urls.append(url)
        payload = fetch_json(
            url,
            headers=_bp_headers(
                referer=f"{BP_SITE}/nfl/odds/player-props/passing-yards/"
            ),
            prefer_curl_cffi=True,
        )
        if not isinstance(payload, dict):
            raise LiveFetchError(f"unexpected offers payload type from {url}")
        batch = payload.get("offers") or []
        if isinstance(batch, list):
            offers.extend(o for o in batch if isinstance(o, dict))
        pagination = payload.get("_pagination") or {}
        try:
            total_pages = max(1, int(pagination.get("total_pages") or 1))
        except (TypeError, ValueError):
            total_pages = 1
        page += 1
        # Safety cap against runaway pagination.
        if page > 50:
            break
    return offers, urls


def fetch_weekly_snapshot(
    *,
    season: int,
    week: int,
    now: datetime | None = None,
) -> ProviderSnapshot:
    clock = now or datetime.now(timezone.utc)
    urls: list[str] = []
    quotes: list[NormalizedQuote] = []
    errors: list[str] = []

    events_url = _events_url(season=season, week=week)
    urls.append(events_url)
    try:
        events = fetch_events(season=season, week=week)
    except LiveFetchError as exc:
        return ProviderSnapshot(
            source="bettingpros",
            season=season,
            week=week,
            fetched_at=clock,
            urls=tuple(urls),
            quotes=(),
            success=False,
            error=str(exc),
            metadata={"live": True, "provider": "bettingpros", "period": "game"},
        )

    open_events = [
        e
        for e in events
        if str(e.get("status") or "").strip().lower() not in _SKIP_EVENT_STATUSES
    ]
    events_by_id = {str(e["id"]): e for e in open_events if e.get("id") is not None}

    for event in open_events:
        eid = event.get("id")
        if eid is None:
            continue
        try:
            offers, offer_urls = fetch_offers_for_event(eid)
            urls.extend(offer_urls)
            quotes.extend(
                parse_offers(
                    offers,
                    events_by_id=events_by_id,
                    fetched_at=clock,
                    source_url=offer_urls[0] if offer_urls else None,
                )
            )
        except LiveFetchError as exc:
            errors.append(f"event {eid}: {exc}")
        except Exception as exc:  # noqa: BLE001 — board isolation within provider
            errors.append(f"event {eid}: {exc}")

    success = len(quotes) > 0
    return ProviderSnapshot(
        source="bettingpros",
        season=season,
        week=week,
        fetched_at=clock,
        urls=tuple(urls),
        quotes=tuple(quotes),
        success=success,
        error=None if success else ("; ".join(errors) or "no_quotes"),
        metadata={
            "live": True,
            "provider": "bettingpros",
            "quote_count": len(quotes),
            "event_count": len(open_events),
            "board_errors": errors,
            "period": "game",
            "book_id": CONSENSUS_BOOK_ID,
            "markets": [key for _, key in WEEKLY_MARKET_IDS],
        },
    )
