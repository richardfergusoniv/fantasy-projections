"""Live BettingPros NFL weekly player O/U fetch.

Uses the public ``api.bettingpros.com/v3`` JSON endpoints that the BettingPros
web UI calls (same capture surface as the draft-assistant vegas_raw dump).

Endpoints
---------
- ``GET /v3/events?sport=NFL&season={season}&week={week}``
- ``GET /v3/offers?sport=NFL&market_id={ids}&event_id={id}&limit=10&page=N``
  (all books; we filter client-side — never Role-1-emit ``book_id=0`` consensus)
- ``GET /v3/books`` (id → sportsbook name)
- ``GET /v3/markets?sport=NFL`` (catalog reference; market ids are pinned below)

Auth uses the site's browser-embedded ``x-api-key``, supplied at runtime via
``BETTINGPROS_API_KEY`` (ops-owned env — not committed; gitleaks treats the
literal as ``generic-api-key``). Missing/empty key → ``LiveFetchError`` and
provider isolation skips BettingPros without aborting DK/FD.
``api.bettingpros.com`` robots.txt disallows crawling; see
``docs/ops/BETTINGPROS_LIVE_WEEKLY_SCRAPE.md``.

Role 1 / Role 2 emission
------------------------
Emits **per-book** quotes (``source=bettingpros``, ``sportsbook=<book name>``),
matching the season-path approach in ``draft_assistant/vegas_consensus.py``:

- Skip ``book_id=0`` (BettingPros Consensus) — that aggregate already blends
  DK/FD (and can include prediction markets); treating it as an independent
  book would inflate ``book_count`` / double-weight DK/FD in ``robust_median``.
- Skip prediction markets (Kalshi / Polymarket / …).
- Skip DraftKings / FanDuel — already scraped by primary live providers.
- Skip DFS / pick'em / exchange-style boards (PrizePicks, Underdog, …).

Surviving books (Caesars, BetMGM, …) are the genuine third+ books for Role 1.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.normalize import build_normalized_quote
from src.ingest.props.providers.http import LiveFetchError, fetch_json
from src.projection.market_quotes import is_prediction_market_book

BP_API_BASE = "https://api.bettingpros.com/v3"
BP_SITE = "https://www.bettingpros.com"
BP_API_KEY_ENV = "BETTINGPROS_API_KEY"

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

# Primary live scrapes already cover these; re-emitting from BP would
# double-weight their lines inside Role 1 robust_median.
PRIMARY_LIVE_SPORTSBOOKS = frozenset({"draftkings", "fanduel"})

# DFS / pick'em / exchange-style boards — not traditional sportsbook O/U.
NON_SPORTSBOOK_NAMES = frozenset(
    {
        "prizepicks",
        "underdog",
        "sleeper",
        "dabble",
        "fliff",
        "betr",
        "draftkings pick6",
        "fanduel picks",
        "draftkings predictions",
        "playsqor",
        "novig",
        "prophetx",
    }
)

# Closed/complete events are post-kickoff — skip for Role 1 boards.
_SKIP_EVENT_STATUSES = frozenset(
    {"closed", "complete", "completed", "final", "cancelled", "canceled", "postponed"}
)


def resolve_bettingpros_api_key() -> str:
    """Return the BettingPros public client key from the environment.

    The value is the browser-embedded ``x-api-key`` the site uses for
    ``api.bettingpros.com`` (not a private account credential). It must not be
    committed — set ``BETTINGPROS_API_KEY`` in local env / ``PRODUCTION_JOB_ENV``.
    """
    key = (os.environ.get(BP_API_KEY_ENV) or "").strip()
    if not key:
        raise LiveFetchError(
            f"{BP_API_KEY_ENV} is not set; BettingPros live scrape requires the "
            "site's public browser x-api-key (ops-owned env, not committed)"
        )
    return key


def _bp_headers(
    *,
    referer: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    key = api_key if api_key is not None else resolve_bettingpros_api_key()
    return {
        "Origin": BP_SITE,
        "Referer": referer or f"{BP_SITE}/nfl/odds/player-props/",
        "x-api-key": key,
    }


def normalize_sportsbook_name(name: str) -> str:
    return str(name or "").strip().lower()


def is_eligible_role1_sportsbook(name: str) -> bool:
    """Whether a BettingPros book name may enter Role 1 / Role 2 means."""
    key = normalize_sportsbook_name(name)
    if not key:
        return False
    if key in {"bettingpros", "bettingpros consensus", "consensus"}:
        return False
    if key in PRIMARY_LIVE_SPORTSBOOKS:
        return False
    if is_prediction_market_book(key):
        return False
    if key in NON_SPORTSBOOK_NAMES:
        return False
    if "pick6" in key or key.endswith(" predictions"):
        return False
    return True


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


def _books_on_selection(selection: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for book in selection.get("books") or []:
        if not isinstance(book, dict):
            continue
        try:
            bid = int(book.get("id"))
        except (TypeError, ValueError):
            continue
        out[bid] = book
    return out


def parse_offer(
    offer: dict[str, Any],
    *,
    event: dict[str, Any] | None,
    fetched_at: datetime,
    books_by_id: Mapping[int, str] | None = None,
    source_url: str | None = None,
) -> list[NormalizedQuote]:
    """Parse one BettingPros offer into per-book NormalizedQuote rows.

    Never emits ``book_id=0`` consensus or prediction-market / primary-live books.
    """
    try:
        market_id = int(offer.get("market_id"))
    except (TypeError, ValueError):
        return []
    market = WEEKLY_MARKET_ID_MAP.get(market_id)
    if market is None:
        return []
    if offer.get("active") is False:
        return []

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
        return []

    over_by_book: dict[int, tuple[float, float | None]] = {}
    under_by_book: dict[int, tuple[float, float | None]] = {}
    for sel in offer.get("selections") or []:
        if not isinstance(sel, dict) or sel.get("active") is False:
            continue
        side = str(sel.get("selection") or sel.get("label") or "").strip().lower()
        for bid, book in _books_on_selection(sel).items():
            line, cost = _main_line(book)
            if line is None:
                continue
            if side.startswith("over"):
                over_by_book[bid] = (line, cost)
            elif side.startswith("under"):
                under_by_book[bid] = (line, cost)

    book_ids = set(over_by_book) | set(under_by_book)
    catalog = dict(books_by_id or {})

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

    quotes: list[NormalizedQuote] = []
    for bid in sorted(book_ids):
        if bid == CONSENSUS_BOOK_ID:
            continue
        sportsbook = catalog.get(bid) or f"book_{bid}"
        if not is_eligible_role1_sportsbook(sportsbook):
            continue
        over = over_by_book.get(bid)
        under = under_by_book.get(bid)
        over_line = over[0] if over else None
        under_line = under[0] if under else None
        over_odds = over[1] if over else None
        under_odds = under[1] if under else None
        if over_line is None and under_line is None:
            continue
        if (
            over_line is not None
            and under_line is not None
            and abs(over_line - under_line) > 1e-9
        ):
            # Mismatched O/U rungs on the same book — skip rather than guess.
            continue
        line = over_line if over_line is not None else under_line
        assert line is not None
        quotes.append(
            build_normalized_quote(
                source="bettingpros",
                sportsbook=str(sportsbook),
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
                    "book_id": bid,
                    "sportsbook": sportsbook,
                },
            )
        )
    return quotes


def parse_offers(
    offers: Iterable[dict[str, Any]],
    *,
    events_by_id: dict[str, dict[str, Any]],
    fetched_at: datetime,
    books_by_id: Mapping[int, str] | None = None,
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
        quotes.extend(
            parse_offer(
                offer,
                event=event,
                fetched_at=fetched_at,
                books_by_id=books_by_id,
                source_url=source_url,
            )
        )
    return quotes


def _events_url(*, season: int, week: int) -> str:
    return (
        f"{BP_API_BASE}/events?"
        + urlencode({"sport": "NFL", "season": season, "week": week})
    )


def _books_url() -> str:
    return f"{BP_API_BASE}/books"


def _offers_url(
    *,
    event_id: int | str,
    market_ids: str,
    page: int = 1,
) -> str:
    # No book_id filter — response includes per-book lines; we drop consensus
    # and ineligible books in parse_offer.
    return (
        f"{BP_API_BASE}/offers?"
        + urlencode(
            {
                "sport": "NFL",
                "market_id": market_ids,
                "event_id": str(event_id),
                "limit": str(OFFERS_PAGE_LIMIT),
                "page": str(page),
            }
        )
    )


def fetch_books_catalog() -> dict[int, str]:
    url = _books_url()
    payload = fetch_json(url, headers=_bp_headers(), prefer_curl_cffi=True)
    if not isinstance(payload, dict):
        raise LiveFetchError(f"unexpected books payload type from {url}")
    books = payload.get("books") or []
    if not isinstance(books, list):
        raise LiveFetchError(f"books is not a list from {url}")
    out: dict[int, str] = {}
    for book in books:
        if not isinstance(book, dict):
            continue
        try:
            bid = int(book.get("id"))
        except (TypeError, ValueError):
            continue
        name = str(book.get("name") or book.get("slug") or "").strip()
        if name:
            out[bid] = name
    return out


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
    """Paginate all-book offers for one event across weekly market ids."""
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

    try:
        # Fail closed before any HTTP when the ops key is missing.
        resolve_bettingpros_api_key()
    except LiveFetchError as exc:
        return ProviderSnapshot(
            source="bettingpros",
            season=season,
            week=week,
            fetched_at=clock,
            urls=(),
            quotes=(),
            success=False,
            error=str(exc),
            metadata={
                "live": True,
                "provider": "bettingpros",
                "period": "game",
                "missing_api_key": True,
            },
        )

    books_url = _books_url()
    urls.append(books_url)
    try:
        books_by_id = fetch_books_catalog()
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
                    books_by_id=books_by_id,
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
            "emission": "per_book",
            "skipped_book_ids": [CONSENSUS_BOOK_ID],
            "skipped_sportsbooks": sorted(
                PRIMARY_LIVE_SPORTSBOOKS
                | NON_SPORTSBOOK_NAMES
                | {"bettingpros", "bettingpros consensus", "consensus"}
            ),
            "markets": [key for _, key in WEEKLY_MARKET_IDS],
            "sportsbooks": sorted({q.sportsbook.lower() for q in quotes}),
        },
    )
