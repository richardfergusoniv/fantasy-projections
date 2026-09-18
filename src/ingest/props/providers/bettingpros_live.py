"""Live BettingPros NFL weekly player-prop scrape (HTML bootstrap + optional API).

Public board / analyzer pages SSR-embed Island JSON. Direct ``api.bettingpros.com``
calls often 403 from cloud egress; we still attempt pagination soft-fail.
See ``docs/ops/WEEKLY_PROPS_BETTINGPROS_TOS_FEASIBILITY_2026-09-18.md``.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.normalize import build_normalized_quote
from src.ingest.props.providers.http import LiveFetchError, fetch_json, fetch_text

BP_SITE = "https://www.bettingpros.com"
BP_API = "https://api.bettingpros.com"
BP_CONSENSUS_BOOK_ID = 0
SOURCE = "bettingpros"

# (page slug, canonical market, BettingPros market_id)
WEEKLY_MARKET_BOARDS: tuple[tuple[str, str, int], ...] = (
    ("passing-yards", "pass_yards", 103),
    ("passing-touchdowns", "pass_tds", 102),
    ("passing-attempts", "pass_attempts", 333),
    ("passing-completions", "pass_completions", 100),
    ("rushing-yards", "rush_yards", 107),
    ("rushing-attempts", "rush_attempts", 106),
    ("receiving-yards", "rec_yards", 105),
    ("receptions", "receptions", 104),
)

# Documented absence: no weekly rush_tds / rec_tds O/U on BP player-props catalog.
WEEKLY_MARKETS_ABSENT: frozenset[str] = frozenset({"rush_tds", "rec_tds"})

_SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def board_url(slug: str) -> str:
    return f"{BP_SITE}/nfl/odds/player-props/{slug}/"


def analyzer_url(*, player_slug: str, market_slug: str) -> str:
    return f"{BP_SITE}/nfl/props/{player_slug}/{market_slug}/"


def extract_bootstrap_json(html: str) -> dict[str, Any]:
    """Pick the largest script JSON blob that looks like a BP odds bootstrap."""
    candidates: list[str] = []
    for body in _SCRIPT_RE.findall(html or ""):
        text = body.strip()
        if not text.startswith("{"):
            continue
        if '"offers"' not in text and '"participantPropOffer"' not in text:
            continue
        candidates.append(text)
    if not candidates:
        raise LiveFetchError("bettingpros_bootstrap_json_missing")
    # Prefer blobs that include offers / participantPropOffer and are largest.
    best = max(candidates, key=len)
    try:
        payload = json.loads(best)
    except json.JSONDecodeError as exc:
        raise LiveFetchError(f"bettingpros_bootstrap_json_invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise LiveFetchError("bettingpros_bootstrap_json_not_object")
    return payload


def _event_index(bootstrap: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    events_block = bootstrap.get("events")
    rows: Iterable[Any]
    if isinstance(events_block, dict):
        rows = events_block.get("events") or []
    elif isinstance(events_block, list):
        rows = events_block
    else:
        rows = []
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            eid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        out[eid] = row
    return out


def _event_matches_week(
    event: Mapping[str, Any] | None,
    *,
    season: int,
    week: int,
) -> bool:
    if not event:
        return True  # keep quote; week filter soft when event missing
    try:
        event_season = int(event.get("season")) if event.get("season") is not None else None
    except (TypeError, ValueError):
        event_season = None
    raw_week = event.get("week")
    try:
        event_week = int(raw_week) if raw_week is not None and str(raw_week).strip() else None
    except (TypeError, ValueError):
        event_week = None
    if event_season is not None and event_season != season:
        return False
    if event_week is not None and event_week != week:
        return False
    return True


def _main_consensus_line(selection: Mapping[str, Any]) -> tuple[float, float | None] | None:
    books = selection.get("books") or []
    for book in books:
        if not isinstance(book, dict):
            continue
        raw_id = book.get("id")
        if raw_id is None:
            continue
        try:
            book_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        # book_id 0 is Consensus — do not use `or` (0 is falsy).
        if book_id != BP_CONSENSUS_BOOK_ID:
            continue
        for line in book.get("lines") or []:
            if not isinstance(line, dict):
                continue
            if line.get("is_off") is True or line.get("active") is False:
                continue
            if line.get("main") is False:
                continue
            try:
                value = float(line["line"])
            except (KeyError, TypeError, ValueError):
                continue
            cost = line.get("cost")
            try:
                odds = float(cost) if cost is not None else None
            except (TypeError, ValueError):
                odds = None
            return value, odds
    return None


def _player_from_offer(offer: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    participants = offer.get("participants") or []
    if participants and isinstance(participants[0], dict):
        part = participants[0]
        name = str(part.get("name") or "").strip()
        player = part.get("player") if isinstance(part.get("player"), dict) else {}
        team = str(player.get("team") or "").strip().upper() or None
        slug = str(player.get("slug") or "").strip() or None
        if name:
            return name, team, slug
    return "", None, None


def parse_board_offers(
    bootstrap: Mapping[str, Any],
    *,
    market: str,
    market_id: int,
    fetched_at: datetime,
    season: int,
    week: int,
    source_url: str | None = None,
) -> list[NormalizedQuote]:
    """Parse Consensus O/U quotes from a board bootstrap ``offers`` array."""
    events = _event_index(bootstrap)
    quotes: list[NormalizedQuote] = []
    for offer in bootstrap.get("offers") or []:
        if not isinstance(offer, dict):
            continue
        try:
            oid_market = int(offer.get("market_id"))
        except (TypeError, ValueError):
            continue
        if oid_market != market_id:
            continue
        try:
            event_id = int(offer.get("event_id")) if offer.get("event_id") is not None else None
        except (TypeError, ValueError):
            event_id = None
        event = events.get(event_id) if event_id is not None else None
        if not _event_matches_week(event, season=season, week=week):
            continue
        player_name, team, _slug = _player_from_offer(offer)
        if not player_name:
            continue
        over_line = under_line = None
        over_odds = under_odds = None
        for sel in offer.get("selections") or []:
            if not isinstance(sel, dict):
                continue
            side = str(sel.get("selection") or "").strip().lower()
            parsed = _main_consensus_line(sel)
            if parsed is None:
                continue
            line, odds = parsed
            if side == "over":
                over_line, over_odds = line, odds
            elif side == "under":
                under_line, under_odds = line, odds
        line = over_line if over_line is not None else under_line
        if line is None:
            continue
        if under_line is not None and over_line is not None and under_line != over_line:
            # Prefer over main when sides disagree (rare); still emit one quote.
            line = over_line
        opponent = None
        if event and team:
            home = str(event.get("home") or "").upper() or None
            visitor = str(event.get("visitor") or "").upper() or None
            if team == home:
                opponent = visitor
            elif team == visitor:
                opponent = home
        event_start = None
        if event and event.get("scheduled"):
            event_start = event.get("scheduled")
        player_id = offer.get("player_id")
        quotes.append(
            build_normalized_quote(
                source=SOURCE,
                sportsbook=SOURCE,
                player_name_raw=player_name,
                market=market,
                line=float(line),
                fetched_at=fetched_at,
                over_odds=over_odds,
                under_odds=under_odds,
                event_id=str(event_id) if event_id is not None else None,
                player_id=str(player_id) if player_id is not None else None,
                team=team,
                opponent=opponent,
                period="game",
                event_start=event_start,
                source_url=source_url,
                kind="book",
                raw={
                    "offer_id": offer.get("id"),
                    "market_id": market_id,
                    "book_id": BP_CONSENSUS_BOOK_ID,
                },
            )
        )
    return quotes


def parse_participant_prop_offer(
    payload: Mapping[str, Any],
    *,
    market: str,
    market_id: int,
    fetched_at: datetime,
    source_url: str | None = None,
    event: Mapping[str, Any] | None = None,
) -> NormalizedQuote | None:
    """Parse Consensus line from an analyzer ``participantPropOffer`` object."""
    ppo = payload.get("participantPropOffer") if "participantPropOffer" in payload else payload
    if not isinstance(ppo, dict):
        return None
    try:
        ppo_market = int(ppo.get("market_id")) if ppo.get("market_id") is not None else None
    except (TypeError, ValueError):
        ppo_market = None
    if ppo_market is not None and ppo_market != market_id:
        return None
    participant = ppo.get("participant") if isinstance(ppo.get("participant"), dict) else {}
    player = participant.get("player") if isinstance(participant.get("player"), dict) else {}
    name = str(participant.get("name") or "").strip()
    if not name:
        return None
    team = str(player.get("team") or "").strip().upper() or None
    over = ppo.get("over") if isinstance(ppo.get("over"), dict) else {}
    under = ppo.get("under") if isinstance(ppo.get("under"), dict) else {}
    line_raw = over.get("consensus_line")
    if line_raw is None:
        line_raw = under.get("consensus_line")
    if line_raw is None:
        line_raw = over.get("line")
    if line_raw is None:
        return None
    try:
        line = float(line_raw)
    except (TypeError, ValueError):
        return None

    def _odds(side: Mapping[str, Any]) -> float | None:
        for key in ("consensus_odds", "odds"):
            if side.get(key) is None:
                continue
            try:
                return float(side[key])
            except (TypeError, ValueError):
                continue
        return None

    opponent = None
    event_start = None
    event_id = ppo.get("event_id")
    if event:
        event_start = event.get("scheduled")
        home = str(event.get("home") or "").upper() or None
        visitor = str(event.get("visitor") or "").upper() or None
        if team == home:
            opponent = visitor
        elif team == visitor:
            opponent = home
    return build_normalized_quote(
        source=SOURCE,
        sportsbook=SOURCE,
        player_name_raw=name,
        market=market,
        line=line,
        fetched_at=fetched_at,
        over_odds=_odds(over),
        under_odds=_odds(under),
        event_id=str(event_id) if event_id is not None else None,
        player_id=str(participant.get("id") or player.get("id") or "") or None,
        team=team,
        opponent=opponent,
        period="game",
        event_start=event_start,
        source_url=source_url,
        kind="book",
        raw={"market_id": market_id, "book_id": BP_CONSENSUS_BOOK_ID, "via": "participantPropOffer"},
    )


def participants_for_market(
    bootstrap: Mapping[str, Any],
    *,
    market_id: int,
) -> list[dict[str, Any]]:
    counts = bootstrap.get("offerCounts") or {}
    rows = counts.get("player-props") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("market_id")) != market_id:
                continue
        except (TypeError, ValueError):
            continue
        parts = row.get("participants") or []
        return [p for p in parts if isinstance(p, dict)]
    return []


def _quote_key(quote: NormalizedQuote) -> tuple[str, str]:
    return (quote.player_name_raw.strip().lower(), quote.market)


def _paginate_offers_api(
    pagination: Mapping[str, Any] | None,
    *,
    referer: str,
) -> list[dict[str, Any]]:
    """Follow ``offersPagination.next`` soft-failing on 403 / errors."""
    if not isinstance(pagination, dict):
        return []
    collected: list[dict[str, Any]] = []
    next_path = pagination.get("next")
    pages = 0
    while next_path and pages < 20:
        pages += 1
        url = urljoin(f"{BP_API}/", str(next_path).lstrip("/"))
        if not url.startswith(BP_API):
            url = f"{BP_API}{next_path}" if str(next_path).startswith("/") else urljoin(f"{BP_API}/", str(next_path))
        try:
            payload = fetch_json(
                url,
                headers={
                    "Accept": "application/json",
                    "Origin": BP_SITE,
                    "Referer": referer,
                },
                prefer_curl_cffi=True,
            )
        except LiveFetchError:
            break
        except Exception:
            break
        if not isinstance(payload, dict):
            break
        for offer in payload.get("offers") or []:
            if isinstance(offer, dict):
                collected.append(offer)
        pag = payload.get("offersPagination") or {}
        next_path = pag.get("next") if isinstance(pag, dict) else None
        if not next_path:
            break
    return collected


def fetch_weekly_snapshot(
    *,
    season: int,
    week: int,
    now: datetime | None = None,
    fill_participants: bool = True,
    participant_pause_s: float = 0.05,
    max_participants_per_market: int | None = None,
) -> ProviderSnapshot:
    """Scrape BP weekly game props into a ProviderSnapshot."""
    clock = now or datetime.now(timezone.utc)
    urls: list[str] = []
    quotes: list[NormalizedQuote] = []
    errors: list[str] = []
    api_pages = 0
    analyzer_fetches = 0
    seen: set[tuple[str, str]] = set()

    for slug, market, market_id in WEEKLY_MARKET_BOARDS:
        url = board_url(slug)
        urls.append(url)
        try:
            html = fetch_text(
                url,
                headers={"Referer": f"{BP_SITE}/nfl/odds/player-props/"},
                prefer_curl_cffi=True,
            )
            bootstrap = extract_bootstrap_json(html)
            board_quotes = parse_board_offers(
                bootstrap,
                market=market,
                market_id=market_id,
                fetched_at=clock,
                season=season,
                week=week,
                source_url=url,
            )
            for q in board_quotes:
                key = _quote_key(q)
                if key not in seen:
                    seen.add(key)
                    quotes.append(q)

            # Soft API pagination using embedded next link.
            extra_offers = _paginate_offers_api(
                bootstrap.get("offersPagination")
                if isinstance(bootstrap.get("offersPagination"), dict)
                else None,
                referer=url,
            )
            if extra_offers:
                api_pages += 1
                merged = dict(bootstrap)
                merged["offers"] = extra_offers
                for q in parse_board_offers(
                    merged,
                    market=market,
                    market_id=market_id,
                    fetched_at=clock,
                    season=season,
                    week=week,
                    source_url=url,
                ):
                    key = _quote_key(q)
                    if key not in seen:
                        seen.add(key)
                        quotes.append(q)

            if fill_participants:
                events = _event_index(bootstrap)
                parts = participants_for_market(bootstrap, market_id=market_id)
                if max_participants_per_market is not None:
                    parts = parts[: max(0, int(max_participants_per_market))]
                for part in parts:
                    player = part.get("player") if isinstance(part.get("player"), dict) else {}
                    player_slug = str(player.get("slug") or "").strip()
                    player_name = str(part.get("name") or "").strip()
                    if not player_slug:
                        continue
                    key = (player_name.lower(), market)
                    if key in seen:
                        continue
                    aurl = analyzer_url(player_slug=player_slug, market_slug=slug)
                    try:
                        ahtml = fetch_text(
                            aurl,
                            headers={"Referer": url},
                            prefer_curl_cffi=True,
                        )
                        analyzer_fetches += 1
                        apayload = extract_bootstrap_json(ahtml)
                        # Analyzer pages may only expose participantPropOffer.
                        quote = parse_participant_prop_offer(
                            apayload,
                            market=market,
                            market_id=market_id,
                            fetched_at=clock,
                            source_url=aurl,
                            event=None,
                        )
                        if quote is None and isinstance(apayload.get("participantPropOffer"), dict):
                            quote = parse_participant_prop_offer(
                                apayload["participantPropOffer"],
                                market=market,
                                market_id=market_id,
                                fetched_at=clock,
                                source_url=aurl,
                            )
                        # Attach opponent from events when event_id present.
                        if quote is not None and quote.event_id:
                            try:
                                ev = events.get(int(quote.event_id))
                            except (TypeError, ValueError):
                                ev = None
                            if ev and not _event_matches_week(ev, season=season, week=week):
                                continue
                        if quote is not None:
                            k2 = _quote_key(quote)
                            if k2 not in seen:
                                seen.add(k2)
                                quotes.append(quote)
                                urls.append(aurl)
                    except LiveFetchError as exc:
                        errors.append(f"{market}/{player_slug}: {exc}")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{market}/{player_slug}: {exc}")
                    if participant_pause_s > 0:
                        time.sleep(participant_pause_s)
        except LiveFetchError as exc:
            errors.append(f"{market}: {exc}")
        except Exception as exc:  # noqa: BLE001 — provider isolation
            errors.append(f"{market}: {exc}")

    success = len(quotes) > 0
    return ProviderSnapshot(
        source=SOURCE,
        season=season,
        week=week,
        fetched_at=clock,
        urls=tuple(dict.fromkeys(urls)),
        quotes=tuple(quotes),
        success=success,
        error=None if success else ("; ".join(errors) or "no_quotes"),
        metadata={
            "live": True,
            "provider": SOURCE,
            "quote_count": len(quotes),
            "board_errors": errors,
            "period": "game",
            "scrape_style": "html_bootstrap",
            "api_pagination_batches": api_pages,
            "analyzer_fetches": analyzer_fetches,
            "weekly_markets_absent": sorted(WEEKLY_MARKETS_ABSENT),
            "consensus_book_id": BP_CONSENSUS_BOOK_ID,
        },
    )
