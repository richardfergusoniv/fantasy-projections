"""Live DraftKings NFL player O/U fetch (weekly game props + season futures).

Uses the public sportscontent feed under sportsbook-nash (requires curl_cffi
Chrome impersonation from most cloud egress paths).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.ingest.props.contracts import NormalizedQuote, ProviderSnapshot
from src.ingest.props.normalize import build_normalized_quote
from src.ingest.props.providers.http import LiveFetchError, fetch_json

DK_LEAGUE_ID = "88808"
DK_API_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusoh/v1"
DK_SITE = "https://sportsbook.draftkings.com"

# Weekly game-line O/U boards (category_id, subcategory_id) → market key.
WEEKLY_OU_BOARDS: tuple[tuple[int, int, str], ...] = (
    (1000, 9524, "pass_yards"),
    (1000, 9525, "pass_tds"),
    (1000, 9517, "pass_attempts"),
    (1000, 9522, "pass_completions"),
    (1000, 15937, "pass_ints"),
    (1001, 9514, "rush_yards"),
    (1001, 9518, "rush_attempts"),
    (1342, 14114, "rec_yards"),
    (1342, 14115, "receptions"),
)

# Season-long Player Futures (category 1759).
SEASON_FUTURE_BOARDS: tuple[tuple[int, str], ...] = (
    (17147, "pass_yards"),
    (17148, "pass_tds"),
    (17223, "rush_yards"),
    (17224, "rush_tds"),
    (17314, "rec_yards"),
    (17315, "rec_tds"),
    (20168, "receptions"),
)

_AMERICAN_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})
_LABEL_LINE = re.compile(
    r"^(?P<side>Over|Under)\s+(?P<line>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_SEASON_PREFIX = re.compile(r"^NFL\s+\d{4}/\d{2}\s+-\s+", re.IGNORECASE)
_MARKET_SUFFIXES: tuple[str, ...] = (
    "Regular Season Passing Yards",
    "Regular Season Passing TDs",
    "Regular Season Passing Touchdowns",
    "Regular Season Rushing Yards",
    "Regular Season Rushing TDs",
    "Regular Season Rushing Touchdowns",
    "Regular Season Receiving Yards",
    "Regular Season Receiving TDs",
    "Regular Season Receiving Touchdowns",
    "Regular Season Receptions",
    "Passing Yards O/U",
    "Passing TDs O/U",
    "Passing Touchdowns O/U",
    "Pass Attempts O/U",
    "Completions O/U",
    "Interceptions O/U",
    "Rushing Yards O/U",
    "Rush Attempts O/U",
    "Receiving Yards O/U",
    "Receptions O/U",
    "Receiving TDs O/U",
    "Passing Yards",
    "Passing TDs",
    "Rushing Yards",
    "Rushing TDs",
    "Receiving Yards",
    "Receptions",
    "Receiving TDs",
)


def _parse_american(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().translate(_AMERICAN_MINUS)
    if not text or text in {"", "—", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # DK uses 7-digit fractional seconds sometimes.
    if text.endswith("Z") and "." in text:
        head, frac = text[:-1].split(".", 1)
        frac = (frac + "000000")[:6]
        text = f"{head}.{frac}+00:00"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _event_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(e.get("id")): e for e in (payload.get("events") or []) if e.get("id")}


def _team_abbr(participant: dict[str, Any] | None) -> str | None:
    if not participant:
        return None
    meta = participant.get("metadata") or {}
    short = meta.get("shortName") or participant.get("seoIdentifier")
    if short:
        return str(short).upper()[:3] if len(str(short)) <= 3 else str(short).upper()
    name = str(participant.get("name") or "")
    # "HOU Texans" / "BUF Bills"
    parts = name.split()
    if parts and len(parts[0]) <= 3:
        return parts[0].upper()
    return None


def _event_teams(event: dict[str, Any] | None) -> tuple[str | None, str | None, datetime | None]:
    if not event:
        return None, None, None
    home = away = None
    for part in event.get("participants") or []:
        role = str(part.get("venueRole") or "").lower()
        abbr = _team_abbr(part)
        if role == "home":
            home = abbr
        elif role == "away":
            away = abbr
    start = _parse_dt(event.get("startEventDate"))
    return home, away, start


def _player_name_from_market(market_name: str) -> str | None:
    text = _SEASON_PREFIX.sub("", market_name.strip()).strip()
    lowered = text.lower()
    for suffix in _MARKET_SUFFIXES:
        if lowered.endswith(suffix.lower()):
            player = text[: -len(suffix)].strip(" -")
            return player or None
    return None


def _selections_by_market(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sel in payload.get("selections") or []:
        mid = sel.get("marketId")
        if mid is None:
            continue
        out.setdefault(str(mid), []).append(sel)
    return out


def _line_from_selection(sel: dict[str, Any]) -> float | None:
    points = sel.get("points")
    if points is not None:
        try:
            return float(points)
        except (TypeError, ValueError):
            pass
    label = str(sel.get("label") or "")
    match = _LABEL_LINE.match(label)
    if match:
        return float(match.group("line"))
    return None


def _odds_from_selection(sel: dict[str, Any]) -> float | None:
    display = sel.get("displayOdds") or {}
    return _parse_american(display.get("american"))


def parse_weekly_board(
    payload: dict[str, Any],
    *,
    market: str,
    source: str = "draftkings",
    fetched_at: datetime,
    source_url: str | None = None,
) -> list[NormalizedQuote]:
    events = _event_map(payload)
    by_market = _selections_by_market(payload)
    quotes: list[NormalizedQuote] = []
    for raw_market in payload.get("markets") or []:
        mid = str(raw_market.get("id") or "")
        name = str(raw_market.get("name") or "")
        player = _player_name_from_market(name)
        if not player:
            # Fall back to participant on first selection.
            sels = by_market.get(mid) or []
            for sel in sels:
                parts = sel.get("participants") or []
                if parts:
                    player = str(parts[0].get("name") or "").strip() or None
                    break
        if not player:
            continue
        event = events.get(str(raw_market.get("eventId") or ""))
        home, away, event_start = _event_teams(event)
        # Prefer player venueRole for team.
        team = None
        sels = by_market.get(mid) or []
        for sel in sels:
            for part in sel.get("participants") or []:
                role = str(part.get("venueRole") or "")
                if "Home" in role and home:
                    team = home
                elif "Away" in role and away:
                    team = away
        opponent = None
        if team and home and away:
            opponent = away if team == home else home if team == away else None
        over_odds = under_odds = None
        line: float | None = None
        for sel in sels:
            side = str(sel.get("outcomeType") or sel.get("label") or "").lower()
            sel_line = _line_from_selection(sel)
            if sel_line is not None:
                line = sel_line
            odds = _odds_from_selection(sel)
            if side.startswith("over"):
                over_odds = odds
            elif side.startswith("under"):
                under_odds = odds
        if line is None:
            continue
        quotes.append(
            build_normalized_quote(
                source=source,
                sportsbook="draftkings",
                player_name_raw=player,
                market=market,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                fetched_at=fetched_at,
                team=team,
                opponent=opponent,
                event_id=None if event is None else str(event.get("id")),
                event_start=event_start,
                source_url=source_url,
                kind="book",
                raw={"market": raw_market, "selections": sels},
            )
        )
    return quotes


def parse_season_board(
    payload: dict[str, Any],
    *,
    market: str,
) -> list[dict[str, Any]]:
    """Return vegas_raw-style player market dict fragments for one board."""
    by_market = _selections_by_market(payload)
    events = _event_map(payload)
    players: list[dict[str, Any]] = []
    for raw_market in payload.get("markets") or []:
        mid = str(raw_market.get("id") or "")
        name = str(raw_market.get("name") or "")
        player = _player_name_from_market(name)
        if not player:
            continue
        event = events.get(str(raw_market.get("eventId") or ""))
        team = None
        if event:
            for part in event.get("participants") or []:
                # Season futures often encode team as a participant next to player.
                if str(part.get("type") or "") == "Team" and part.get("metadata"):
                    team = _team_abbr(part) or team
        over_odds = under_odds = None
        line: float | None = None
        for sel in by_market.get(mid) or []:
            side = str(sel.get("outcomeType") or "").lower()
            label = str(sel.get("label") or "")
            match = _LABEL_LINE.match(label)
            if match:
                line = float(match.group("line"))
                side = match.group("side").lower()
            else:
                sel_line = _line_from_selection(sel)
                if sel_line is not None:
                    line = sel_line
            odds = _odds_from_selection(sel)
            if side.startswith("over"):
                over_odds = odds
            elif side.startswith("under"):
                under_odds = odds
        if line is None:
            continue
        players.append(
            {
                "name": player,
                "team": team,
                "markets": {
                    market: {
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                    }
                },
            }
        )
    return players


def _board_url(category_id: int, subcategory_id: int) -> str:
    return (
        f"{DK_API_BASE}/leagues/{DK_LEAGUE_ID}/categories/{category_id}"
        f"/subcategories/{subcategory_id}"
    )


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
    for category_id, subcategory_id, market in WEEKLY_OU_BOARDS:
        url = _board_url(category_id, subcategory_id)
        urls.append(url)
        try:
            payload = fetch_json(
                url,
                headers={"Referer": f"{DK_SITE}/leagues/football/nfl"},
                prefer_curl_cffi=True,
            )
            quotes.extend(
                parse_weekly_board(
                    payload,
                    market=market,
                    fetched_at=clock,
                    source_url=url,
                )
            )
        except LiveFetchError as exc:
            errors.append(f"{market}: {exc}")
        except Exception as exc:  # noqa: BLE001 — provider isolation
            errors.append(f"{market}: {exc}")
    success = len(quotes) > 0
    return ProviderSnapshot(
        source="draftkings",
        season=season,
        week=week,
        fetched_at=clock,
        urls=tuple(urls),
        quotes=tuple(quotes),
        success=success,
        error=None if success else ("; ".join(errors) or "no_quotes"),
        metadata={
            "live": True,
            "provider": "draftkings",
            "quote_count": len(quotes),
            "board_errors": errors,
            "period": "game",
        },
    )


def fetch_season_raw(
    *,
    season: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch season-long player O/Us into a vegas_raw-compatible payload."""
    clock = now or datetime.now(timezone.utc)
    urls: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for subcategory_id, market in SEASON_FUTURE_BOARDS:
        url = _board_url(1759, subcategory_id)
        urls.append(url)
        try:
            payload = fetch_json(
                url,
                headers={"Referer": f"{DK_SITE}/leagues/football/nfl"},
                prefer_curl_cffi=True,
            )
            for row in parse_season_board(payload, market=market):
                key = str(row["name"]).strip().lower()
                bucket = merged.setdefault(
                    key,
                    {
                        "name": row["name"],
                        "team": row.get("team"),
                        "markets": {},
                    },
                )
                if row.get("team") and not bucket.get("team"):
                    bucket["team"] = row["team"]
                bucket["markets"].update(row.get("markets") or {})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{market}: {exc}")
    return {
        "source": "draftkings",
        "season": season,
        "fetched_at": clock.isoformat(),
        "urls": urls,
        "players": list(merged.values()),
        "notes": (
            "Live DraftKings season Player Futures O/U via sportscontent dkusoh v1. "
            + (f"Board errors: {errors}" if errors else "")
        ).strip(),
        "live": True,
        "errors": errors,
    }
