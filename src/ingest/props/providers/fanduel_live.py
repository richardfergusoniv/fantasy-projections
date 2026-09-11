"""Live FanDuel NFL player O/U fetch (weekly game props + season futures).

Uses the public NJ SBAPI content-managed / event-page endpoints.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.normalize import build_normalized_quote, canonicalize_market
from src.ingest.props.providers.http import LiveFetchError, fetch_json

FD_AK = "FhMFpcPWXMeyZxOx"
FD_SBAPI = "https://sbapi.nj.sportsbook.fanduel.com"
FD_NFL_PAGE = (
    f"{FD_SBAPI}/api/content-managed-page?"
    + urlencode(
        {
            "currencyCode": "USD",
            "exchangeLocale": "en_US",
            "includePrices": "true",
            "language": "en",
            "regionCode": "NAMERICA",
            "timezone": "America/New_York",
            "_ak": FD_AK,
            "page": "CUSTOM",
            "customPageId": "nfl",
        }
    )
)

WEEKLY_TABS: tuple[str, ...] = (
    "popular",
    "passing-props",
    "rushing-props",
    "receiving-props",
)

# marketType substring → canonical market. Skip ALT ladders.
WEEKLY_MARKET_TYPE_MAP: tuple[tuple[str, str], ...] = (
    ("PASSING_YARDS", "pass_yards"),
    ("PASSING_TOUCHDOWNS", "pass_tds"),
    ("PASSING_ATTEMPTS", "pass_attempts"),
    ("PASSING_COMPLETIONS", "pass_completions"),
    ("INTERCEPTIONS", "pass_ints"),
    ("RUSHING_YARDS", "rush_yards"),
    ("RUSHING_TOUCHDOWNS", "rush_tds"),
    ("RUSHING_ATTEMPTS", "rush_attempts"),
    ("RECEIVING_YARDS", "rec_yards"),
    ("RECEIVING_TOUCHDOWNS", "rec_tds"),
    ("RECEPTIONS", "receptions"),
)

SEASON_STAT_ALIASES: dict[str, str] = {
    "passing yards": "pass_yards",
    "passing touchdowns": "pass_tds",
    "pass touchdowns": "pass_tds",
    "rushing yards": "rush_yards",
    "rushing touchdowns": "rush_tds",
    "receiving yards": "rec_yards",
    "receiving touchdowns": "rec_tds",
    "receptions": "receptions",
}

_SEASON_MARKET = re.compile(
    r"^(?P<player>.+?)\s+Regular Season\s+(?P<stat>.+?)(?:\s+20\d{2}-\d{2})?$",
    re.IGNORECASE,
)
_PLAYER_TOTAL = re.compile(
    r"^(?P<player>.+?)\s+-\s+Total\s+(?P<stat>.+)$",
    re.IGNORECASE,
)
_OVER_UNDER_NAME = re.compile(
    r"^(?P<player>.+?)\s+(?P<aside>Over|Under)\s+(?P<line>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)

TEAM_SLUG_TO_ABBR: dict[str, str] = {
    "arizona_cardinals": "ARI",
    "atlanta_falcons": "ATL",
    "baltimore_ravens": "BAL",
    "buffalo_bills": "BUF",
    "carolina_panthers": "CAR",
    "chicago_bears": "CHI",
    "cincinnati_bengals": "CIN",
    "cleveland_browns": "CLE",
    "dallas_cowboys": "DAL",
    "denver_broncos": "DEN",
    "detroit_lions": "DET",
    "green_bay_packers": "GB",
    "houston_texans": "HOU",
    "indianapolis_colts": "IND",
    "jacksonville_jaguars": "JAX",
    "kansas_city_chiefs": "KC",
    "las_vegas_raiders": "LV",
    "los_angeles_chargers": "LAC",
    "los_angeles_rams": "LA",
    "miami_dolphins": "MIA",
    "minnesota_vikings": "MIN",
    "new_england_patriots": "NE",
    "new_orleans_saints": "NO",
    "new_york_giants": "NYG",
    "new_york_jets": "NYJ",
    "philadelphia_eagles": "PHI",
    "pittsburgh_steelers": "PIT",
    "san_francisco_49ers": "SF",
    "seattle_seahawks": "SEA",
    "tampa_bay_buccaneers": "TB",
    "tennessee_titans": "TEN",
    "washington_commanders": "WAS",
}


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
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


def _american(runner: dict[str, Any]) -> float | None:
    odds = (
        ((runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get(
            "americanOdds"
        )
    )
    if odds is None:
        return None
    try:
        return float(odds)
    except (TypeError, ValueError):
        return None


def _team_from_logo(runner: dict[str, Any]) -> str | None:
    for key in ("logo", "secondaryLogo"):
        url = str(runner.get(key) or "")
        for slug, abbr in TEAM_SLUG_TO_ABBR.items():
            if slug in url:
                return abbr
    return None


def _event_opponent(event_name: str, team: str | None) -> str | None:
    if not team or "@" not in event_name:
        return None
    # FanDuel uses full names: "Buffalo Bills @ Houston Texans"
    # Without a full-name map, leave opponent unset when ambiguous.
    return None


def _weekly_market_key(market_type: str, market_name: str) -> str | None:
    mt = market_type.upper()
    if "ALT" in mt or mt.startswith("MOST_"):
        return None
    for needle, key in WEEKLY_MARKET_TYPE_MAP:
        if needle in mt:
            return key
    # Name fallback: "Dalton Kincaid - Total Receptions"
    match = _PLAYER_TOTAL.match(market_name.strip())
    if match:
        return canonicalize_market(match.group("stat"))
    return None


def _player_from_weekly_market(market_name: str, runners: list[dict[str, Any]]) -> str | None:
    match = _PLAYER_TOTAL.match(market_name.strip())
    if match:
        return match.group("player").strip()
    for runner in runners:
        name = str(runner.get("runnerName") or "")
        for suffix in (" Over", " Under"):
            if name.endswith(suffix):
                return name[: -len(suffix)].strip()
    return None


def parse_weekly_markets(
    payload: dict[str, Any],
    *,
    fetched_at: datetime,
    source_url: str | None = None,
) -> list[NormalizedQuote]:
    attachments = payload.get("attachments") or {}
    markets = attachments.get("markets") or {}
    events = attachments.get("events") or {}
    quotes: list[NormalizedQuote] = []
    seen: set[tuple[str, str, float]] = set()
    for market in markets.values():
        if not isinstance(market, dict):
            continue
        if str(market.get("marketStatus") or "").upper() not in {"", "OPEN"}:
            # Still accept missing status; skip explicitly closed/suspended.
            status = str(market.get("marketStatus") or "").upper()
            if status and status not in {"OPEN"}:
                continue
        market_type = str(market.get("marketType") or "")
        market_name = str(market.get("marketName") or "")
        key = _weekly_market_key(market_type, market_name)
        if key is None:
            continue
        runners = [r for r in (market.get("runners") or []) if isinstance(r, dict)]
        player = _player_from_weekly_market(market_name, runners)
        if not player:
            continue
        line: float | None = None
        over_odds = under_odds = None
        team = None
        for runner in runners:
            team = team or _team_from_logo(runner)
            result_type = str((runner.get("result") or {}).get("type") or "").upper()
            runner_name = str(runner.get("runnerName") or "")
            handicap = runner.get("handicap")
            try:
                if handicap is not None:
                    line = float(handicap)
            except (TypeError, ValueError):
                pass
            odds = _american(runner)
            if result_type == "OVER" or runner_name.endswith(" Over"):
                over_odds = odds
            elif result_type == "UNDER" or runner_name.endswith(" Under"):
                under_odds = odds
        if line is None:
            continue
        dedupe = (player.lower(), key, float(line))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        event = events.get(str(market.get("eventId"))) or events.get(market.get("eventId"))
        event_name = str((event or {}).get("name") or "")
        event_start = _parse_dt((event or {}).get("openDate") or market.get("marketTime"))
        quotes.append(
            build_normalized_quote(
                source="fanduel",
                sportsbook="fanduel",
                player_name_raw=player,
                market=key,
                line=float(line),
                over_odds=over_odds,
                under_odds=under_odds,
                fetched_at=fetched_at,
                team=team,
                opponent=_event_opponent(event_name, team),
                event_id=None if market.get("eventId") is None else str(market.get("eventId")),
                event_start=event_start,
                source_url=source_url,
                kind="book",
                raw={"market": market},
            )
        )
    return quotes


def parse_season_markets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    attachments = payload.get("attachments") or {}
    markets = attachments.get("markets") or {}
    merged: dict[str, dict[str, Any]] = {}
    for market in markets.values():
        if not isinstance(market, dict):
            continue
        name = str(market.get("marketName") or "")
        match = _SEASON_MARKET.match(name)
        if not match:
            continue
        player = match.group("player").strip()
        stat = match.group("stat").strip().lower()
        market_key = None
        for alias, key in SEASON_STAT_ALIASES.items():
            if alias in stat:
                market_key = key
                break
        if market_key is None:
            market_key = canonicalize_market(stat.replace(" ", "_"))
        if market_key is None:
            continue
        line: float | None = None
        over_odds = under_odds = None
        team = None
        for runner in market.get("runners") or []:
            if not isinstance(runner, dict):
                continue
            team = team or _team_from_logo(runner)
            rn = str(runner.get("runnerName") or "")
            m2 = _OVER_UNDER_NAME.match(rn)
            if m2:
                line = float(m2.group("line"))
                side = m2.group("aside").lower()
            else:
                m3 = re.match(r"^(Over|Under)\s+(\d+(?:\.\d+)?)$", rn, re.I)
                if not m3:
                    continue
                line = float(m3.group(2))
                side = m3.group(1).lower()
            odds = _american(runner)
            if side == "over":
                over_odds = odds
            else:
                under_odds = odds
        if line is None:
            continue
        key = player.lower()
        bucket = merged.setdefault(
            key, {"name": player, "team": team, "markets": {}}
        )
        if team and not bucket.get("team"):
            bucket["team"] = team
        bucket["markets"][market_key] = {
            "line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
        }
    return list(merged.values())


def _event_page_url(event_id: str | int, tab: str) -> str:
    return (
        f"{FD_SBAPI}/api/event-page?"
        + urlencode(
            {
                "eventId": str(event_id),
                "tab": tab,
                "_ak": FD_AK,
                "includePrices": "true",
                "currencyCode": "USD",
                "exchangeLocale": "en_US",
                "language": "en",
                "regionCode": "NAMERICA",
                "timezone": "America/New_York",
            }
        )
    )


def _game_event_ids(nfl_page: dict[str, Any]) -> list[str]:
    attachments = nfl_page.get("attachments") or {}
    events = attachments.get("events") or {}
    markets = attachments.get("markets") or {}
    # Prefer events that have game lines (moneyline / total).
    game_ids: set[str] = set()
    for market in markets.values():
        if not isinstance(market, dict):
            continue
        mt = str(market.get("marketType") or "")
        if mt in {"MONEY_LINE", "TOTAL_POINTS_(OVER/UNDER)", "MATCH_HANDICAP_(2-WAY)"}:
            if market.get("eventId") is not None:
                game_ids.add(str(market["eventId"]))
    if game_ids:
        return sorted(game_ids)
    # Fallback: events whose name looks like "Away @ Home".
    out: list[str] = []
    for eid, event in events.items():
        name = str((event or {}).get("name") or "")
        if "@" in name:
            out.append(str(eid))
    return out


def fetch_weekly_snapshot(
    *,
    season: int,
    week: int,
    now: datetime | None = None,
    max_events: int = 24,
) -> ProviderSnapshot:
    clock = now or datetime.now(timezone.utc)
    urls: list[str] = [FD_NFL_PAGE]
    quotes: list[NormalizedQuote] = []
    errors: list[str] = []
    try:
        nfl_page = fetch_json(
            FD_NFL_PAGE,
            prefer_curl_cffi=False,
            headers={"Referer": "https://sportsbook.fanduel.com/navigation/nfl"},
        )
    except LiveFetchError as exc:
        return ProviderSnapshot(
            source="fanduel",
            season=season,
            week=week,
            fetched_at=clock,
            urls=tuple(urls),
            quotes=(),
            success=False,
            error=str(exc),
            metadata={"live": True, "provider": "fanduel"},
        )

    event_ids = _game_event_ids(nfl_page)[:max_events]
    for event_id in event_ids:
        for tab in WEEKLY_TABS:
            url = _event_page_url(event_id, tab)
            urls.append(url)
            try:
                payload = fetch_json(
                    url,
                    prefer_curl_cffi=False,
                    headers={"Referer": "https://sportsbook.fanduel.com/"},
                )
                quotes.extend(
                    parse_weekly_markets(
                        payload, fetched_at=clock, source_url=url
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"event={event_id} tab={tab}: {exc}")

    # Dedupe quotes across overlapping tabs.
    deduped: dict[tuple[str, str, float], NormalizedQuote] = {}
    for quote in quotes:
        key = (quote.player_name_raw.lower(), quote.market, float(quote.line))
        deduped[key] = quote
    final = list(deduped.values())
    success = len(final) > 0
    return ProviderSnapshot(
        source="fanduel",
        season=season,
        week=week,
        fetched_at=clock,
        urls=tuple(dict.fromkeys(urls)),
        quotes=tuple(final),
        success=success,
        error=None if success else ("; ".join(errors) or "no_quotes"),
        metadata={
            "live": True,
            "provider": "fanduel",
            "quote_count": len(final),
            "event_count": len(event_ids),
            "board_errors": errors,
            "period": "game",
        },
    )


def fetch_season_raw(
    *,
    season: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = now or datetime.now(timezone.utc)
    payload = fetch_json(
        FD_NFL_PAGE,
        prefer_curl_cffi=False,
        headers={"Referer": "https://sportsbook.fanduel.com/navigation/nfl"},
    )
    players = parse_season_markets(payload)
    return {
        "source": "fanduel",
        "season": season,
        "fetched_at": clock.isoformat(),
        "urls": [FD_NFL_PAGE],
        "players": players,
        "notes": "Live FanDuel season Regular Season player O/U via SBAPI NFL page.",
        "live": True,
    }
