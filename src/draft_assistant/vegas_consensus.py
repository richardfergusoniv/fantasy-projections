"""Merge multi-book Vegas raw scrapes into a sealed consensus board.

Yardage / TD / receptions prefer the median of public sportsbook O/U lines.
Pass attempts, rush attempts, and targets are not posted as public season
O/Us (sportsbooks + prediction markets checked). Checklist volume ranks use
Vegas yards and receptions instead; numberFire attempt/target projections may
still appear in the consensus for reference.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from src.draft_assistant.market_adp import canonicalize_player_name, normalize_player_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DATA_DIR = REPO_ROOT / "draft_assistant" / "data"
VEGAS_RAW_DIR = DRAFT_DATA_DIR / "vegas_raw"

PLAYER_MARKET_ALIASES: dict[str, str] = {
    "pass_yards": "pass_yards",
    "passing_yards": "pass_yards",
    "season_passing_yards": "pass_yards",
    "season_total_passing_yards": "pass_yards",
    "rg_best_bet_total_passing_yards": "pass_yards",
    "pass_tds": "pass_tds",
    "passing_tds": "pass_tds",
    "pass_attempts": "pass_attempts",
    "passing_attempts": "pass_attempts",
    "rush_yards": "rush_yards",
    "rushing_yards": "rush_yards",
    "season_rushing_yards": "rush_yards",
    "season_total_rushing_yards": "rush_yards",
    "rg_best_bet_total_rushing_yards": "rush_yards",
    "rush_tds": "rush_tds",
    "rushing_tds": "rush_tds",
    "season_rushing_touchdowns": "rush_tds",
    "rush_attempts": "rush_attempts",
    "rushing_attempts": "rush_attempts",
    "carries": "rush_attempts",
    "rec_yards": "rec_yards",
    "receiving_yards": "rec_yards",
    "season_receiving_yards": "rec_yards",
    "season_total_receiving_yards": "rec_yards",
    "rg_best_bet_total_receiving_yards": "rec_yards",
    "sharp_proj_receiving_yards": "rec_yards",
    "rec_tds": "rec_tds",
    "receiving_tds": "rec_tds",
    "receptions": "receptions",
    "season_receptions": "receptions",
    "sharp_proj_receptions": "receptions",
    "targets": "targets",
    "fantasy_points": "fantasy_points",
}

TEAM_MARKET_ALIASES: dict[str, str] = {
    "points_scored": "points_scored",
    "points_scored_implied": "points_scored",
    "implied_season_points": "points_scored",
    "points_scored_an_median": "points_scored",
    "points_scored_pg": "points_scored_pg",
    "points_scored_per_game": "points_scored_pg",
    "points_scored_per_game_implied": "points_scored_pg",
    "implied_ppg": "points_scored_pg",
    "total_yards": "total_yards",
    "pass_yards": "pass_yards",
    "rush_yards": "rush_yards",
    "win_total": "win_total",
    "rg_win_total": "win_total",
    "sbr_win_total": "win_total",
}

TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}

ABBR_ALIASES = {
    "JAC": "JAX",
    "WSH": "WAS",
    "LAR": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}


PREDICTION_MARKET_BOOKS = frozenset(
    {
        "kalshi",
        "polymarket",
        "polymarket us",
        "polymarket.com",
    }
)
SKEWED_ODDS_GAP = 0.35  # implied-probability gap between over and under


def _american_implied_prob(odds: Any) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def _odds_skewed(over: Any, under: Any, *, gap: float = SKEWED_ODDS_GAP) -> bool:
    over_p = _american_implied_prob(over)
    under_p = _american_implied_prob(under)
    if over_p is None or under_p is None:
        return False
    return abs(over_p - under_p) > gap


def _one_sided_longshot(over: Any, under: Any, *, threshold: float = 100.0) -> bool:
    """Reject threshold props posted as plus-money on only one side.

    Plus-100 overlays (Alec Pierce DK 999.5 receiving yards at +125) are juice
    ladders, not main season totals.
    """

    def _is_longshot(odds: Any) -> bool:
        try:
            return float(odds) >= threshold
        except (TypeError, ValueError):
            return False

    if over is not None and under is None and _is_longshot(over):
        return True
    if under is not None and over is None and _is_longshot(under):
        return True
    return False


def _is_prediction_market_book(name: str) -> bool:
    key = str(name or "").strip().lower()
    if key in PREDICTION_MARKET_BOOKS:
        return True
    return "kalshi" in key or "polymarket" in key


def _conflicts_with_projection(line: float, projection: float) -> bool:
    """True when a book line is juiced upward vs the same-source projection.

    TD / count markets use a tighter absolute gap (Caesars-only Alec Pierce 7.5
    TDs vs RotoWire 6.0). Yard markets catch milder overlays like 999.5 vs 872
    that the prior 25%/150-yard gate missed.
    """
    scale = max(abs(projection), 1.0)
    delta = abs(line - projection)
    rel = delta / scale
    if scale <= 25:
        # Receptions / passing·rushing·receiving TDs.
        return line > projection and delta >= 1.0 and rel >= 0.15
    return (rel > 0.5 and delta > max(1.5, 0.05 * scale)) or (
        line > projection and rel > 0.12 and delta > 80
    )


def _book_entry_line(raw: Any) -> float | None:
    if not isinstance(raw, dict):
        return _scalar_line(raw)
    if (
        _odds_skewed(raw.get("over_odds"), raw.get("under_odds"))
        or _odds_skewed(raw.get("over_odds_american"), raw.get("under_odds_american"))
        or _one_sided_longshot(raw.get("over_odds"), raw.get("under_odds"))
        or _one_sided_longshot(
            raw.get("over_odds_american"), raw.get("under_odds_american")
        )
    ):
        return None
    for key in ("line", "value", "ou", "total"):
        if key in raw:
            value = _scalar_line(raw[key])
            if value is not None:
                return value
    return None


def _scalar_line(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _extract_quote(raw: Any) -> tuple[float, str] | None:
    """Return ``(value, kind)`` where kind is ``book`` or ``projection``.

    Prediction-market-only boards and heavily skewed O/U prices (Kalshi /
    Polymarket thresholds such as Kenny Gainwell 749.5 rush yards at +355/-567)
    are rejected so they cannot inflate consensus.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float, str)):
        value = _scalar_line(raw)
        return (value, "book") if value is not None else None
    if not isinstance(raw, dict):
        return None

    projection = None
    for key in ("rotowire_proj", "projection", "proj", "projected"):
        if key in raw:
            projection = _scalar_line(raw[key])
            if projection is not None:
                break

    books = raw.get("books")
    sided_lines: list[float] = []
    unsided_lines: list[float] = []
    if isinstance(books, dict) and books:
        for book_name, book_raw in books.items():
            if _is_prediction_market_book(str(book_name)):
                continue
            book_line = _book_entry_line(book_raw)
            if book_line is None:
                continue
            # Drop per-book juice vs the same-source projection before median
            # (Pierce: keep FanDuel 925.5, drop Caesars/DK 999.5 vs RW 872).
            if projection is not None and _conflicts_with_projection(
                book_line, projection
            ):
                continue
            # Prefer two-sided prices. Odds-less alt totals (DK/Caesars 3499.5
            # next to a fair FanDuel 1950.5 for Fernando Mendoza) are often
            # juice thresholds and can dominate a plain median.
            if isinstance(book_raw, dict) and (
                (
                    book_raw.get("over_odds") is not None
                    and book_raw.get("under_odds") is not None
                )
                or (
                    book_raw.get("over_odds_american") is not None
                    and book_raw.get("under_odds_american") is not None
                )
            ):
                sided_lines.append(book_line)
            else:
                unsided_lines.append(book_line)

    traditional_lines = sided_lines if sided_lines else unsided_lines

    line = None
    if traditional_lines:
        line = float(median(traditional_lines))
    else:
        # No traditional sportsbook quotes. Reject prediction-market-only boards
        # and top-level lines with heavily skewed over/under prices.
        only_prediction_markets = False
        if isinstance(books, dict) and books:
            only_prediction_markets = all(
                _is_prediction_market_book(str(name)) for name in books
            )
        top_line = None
        for key in ("line", "value", "ou", "total"):
            if key in raw:
                top_line = _scalar_line(raw[key])
                if top_line is not None:
                    break
        skewed = (
            _odds_skewed(raw.get("over_odds"), raw.get("under_odds"))
            or _odds_skewed(raw.get("over_odds_american"), raw.get("under_odds_american"))
            or _one_sided_longshot(raw.get("over_odds"), raw.get("under_odds"))
            or _one_sided_longshot(
                raw.get("over_odds_american"), raw.get("under_odds_american")
            )
        )
        if top_line is not None and not only_prediction_markets and not skewed:
            line = top_line
        elif top_line is not None and only_prediction_markets:
            line = None
        elif top_line is not None and skewed:
            line = None

    if line is not None and projection is not None:
        scale = max(abs(projection), 1.0)
        delta = abs(line - projection)
        rel = delta / scale
        if _conflicts_with_projection(line, projection):
            # Severe juice (Kupp 1499.5 vs 364): drop the whole source so a
            # low RotoWire proj cannot blend with NumberFire. Milder juice
            # (Pierce Caesars 7.5 TDs vs RW 6.0): trust the source projection.
            if rel > 0.5:
                return None
            return (projection, "projection")
    if line is not None:
        return (line, "book")
    if projection is not None:
        return (projection, "projection")
    return None


def _line_value(raw: Any) -> float | None:
    quote = _extract_quote(raw)
    return None if quote is None else quote[0]


def _robust_median(values: list[float]) -> float:
    """Median after dropping far outliers when 3+ quotes exist."""
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return float(median(values))
    center = float(median(values))
    tolerance = max(abs(center) * 0.35, 100.0)
    kept = [value for value in values if abs(value - center) <= tolerance]
    if not kept:
        kept = list(values)
    return float(median(kept))


def _team_abbr(team: Any = None, name: Any = None) -> str | None:
    if team:
        abbr = str(team).strip().upper()
        if not abbr:
            return None
        return ABBR_ALIASES.get(abbr, abbr)
    if name:
        return TEAM_NAME_TO_ABBR.get(str(name).strip().lower())
    return None


def _load_raw_files() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if not VEGAS_RAW_DIR.is_dir():
        return out
    for path in sorted(VEGAS_RAW_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        out.append((path.stem, payload))
    return out


def _add_line(
    store: dict[str, dict[str, list[tuple[float, str]]]],
    meta: dict[str, dict[str, Any]],
    *,
    name: str,
    team: str | None,
    position: str | None,
    market: str,
    value: float,
    source: str,
    kind: str = "book",
) -> None:
    canon = PLAYER_MARKET_ALIASES.get(market)
    if not canon:
        return
    key = canonicalize_player_name(name)
    store.setdefault(key, {}).setdefault(canon, []).append((float(value), kind))
    identity = meta.setdefault(
        key,
        {"name": name, "team": team, "position": position, "sources": set()},
    )
    # Prefer the longer/formal display name when nicknames merge (Kenny -> Kenneth).
    if len(str(name)) > len(str(identity.get("name") or "")):
        identity["name"] = name
    if team and not identity.get("team"):
        identity["team"] = team
    if position and not identity.get("position"):
        identity["position"] = position
    identity["sources"].add(source)


def _collect_player_lines(
    files: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, Any]]]:
    lines: dict[str, dict[str, list[tuple[float, str]]]] = {}
    meta: dict[str, dict[str, Any]] = {}

    nf_map = {
        "pass_attempts": "pass_attempts",
        "rush_attempts": "rush_attempts",
        "targets": "targets",
        "receptions": "receptions",
        "pass_yards": "pass_yards",
        "rush_yards": "rush_yards",
        "receiving_yards": "rec_yards",
        "pass_tds": "pass_tds",
        "rush_tds": "rush_tds",
        "receiving_tds": "rec_tds",
        "fantasy_points": "fantasy_points",
    }

    for source, payload in files:
        for row in payload.get("players") or []:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            team = _team_abbr(row.get("team"))
            position = str(row.get("position") or "").strip().upper() or None
            markets = dict(row.get("markets") or {})
            projections = row.get("projections") or {}
            nf = projections.get("numberfire") if isinstance(projections, dict) else None
            if isinstance(nf, dict):
                for nf_key, canon in nf_map.items():
                    value = _scalar_line(nf.get(nf_key))
                    if value is not None:
                        _add_line(
                            lines,
                            meta,
                            name=name,
                            team=team,
                            position=position,
                            market=canon,
                            value=value,
                            source=f"{source}:numberfire",
                            kind="projection",
                        )
            for market, raw in markets.items():
                quote = _extract_quote(raw)
                if quote is None:
                    continue
                value, kind = quote
                _add_line(
                    lines,
                    meta,
                    name=name,
                    team=team,
                    position=position,
                    market=str(market),
                    value=value,
                    source=source,
                    kind=kind,
                )
    return lines, meta


def _collect_team_lines(
    files: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, list[float]]]:
    lines: dict[str, dict[str, list[float]]] = {}
    for _source, payload in files:
        for row in payload.get("teams") or []:
            abbr = _team_abbr(row.get("abbr") or row.get("team"), row.get("name"))
            if not abbr:
                continue
            markets = dict(row.get("markets") or {})
            projections = row.get("projections") or {}
            if isinstance(projections, dict):
                for block in projections.values():
                    if not isinstance(block, dict):
                        continue
                    for key, raw in block.items():
                        if key in TEAM_MARKET_ALIASES or key in {
                            "total_yards",
                            "pass_yards",
                            "rush_yards",
                            "points_scored",
                            "points_scored_approx",
                        }:
                            markets.setdefault(key, raw)
            for market, raw in markets.items():
                market_key = str(market)
                canon = TEAM_MARKET_ALIASES.get(market_key)
                if market_key == "points_scored_approx":
                    canon = "points_scored"
                if market_key == "total_yards":
                    canon = "total_yards"
                if not canon:
                    continue
                value = _line_value(raw)
                if value is None:
                    continue
                if canon == "points_scored" and value < 50:
                    value *= 17.0
                lines.setdefault(abbr, {}).setdefault(canon, []).append(value)
    return lines


SCORING_MARKETS = frozenset(
    {
        "pass_yards",
        "rush_yards",
        "rec_yards",
        "pass_tds",
        "rush_tds",
        "rec_tds",
        "receptions",
    }
)


def _median_map(values: dict[str, list]) -> dict[str, float]:
    """Prefer sportsbook O/U quotes; fall back to model projections when needed."""
    consensus, _kinds = _median_map_with_kinds(values)
    return consensus


def _median_map_with_kinds(
    values: dict[str, list],
) -> tuple[dict[str, float], dict[str, str]]:
    """Return consensus values plus per-market kind (``book`` or ``projection``).

    Player quotes are ``(value, kind)`` tuples. Team quotes may still be bare floats.
    """
    out: dict[str, float] = {}
    kinds: dict[str, str] = {}
    for key, quotes in values.items():
        if not quotes:
            continue
        if isinstance(quotes[0], tuple):
            books = [value for value, kind in quotes if kind == "book"]
            projections = [value for value, kind in quotes if kind == "projection"]
            if books:
                out[key] = _robust_median(books)
                kinds[key] = "book"
            elif projections:
                out[key] = _robust_median(projections)
                kinds[key] = "projection"
        else:
            out[key] = _robust_median([float(value) for value in quotes])
            kinds[key] = "book"
    return out, kinds


def _prop_coverage(kinds: dict[str, str]) -> str:
    """Classify whether scoring markets are sportsbook-backed.

    ``projection`` means Vegas FP would be model-only (e.g. Cooper Kupp with the
    Caesars juice dropped and only numberFire left) and must not rank as Vegas.
    """
    scoring = {key: kind for key, kind in kinds.items() if key in SCORING_MARKETS}
    if not scoring:
        return "none"
    distinct = set(scoring.values())
    if distinct == {"book"}:
        return "books"
    if distinct == {"projection"}:
        return "projection"
    return "mixed"


def build_consensus(*, season: int = 2026) -> dict[str, Any]:
    files = _load_raw_files()
    player_lines, player_meta = _collect_player_lines(files)
    team_lines = _collect_team_lines(files)

    players_out: list[dict[str, Any]] = []
    for key, markets in sorted(player_lines.items()):
        identity = player_meta.get(key) or {}
        consensus, kinds = _median_map_with_kinds(markets)
        if not consensus:
            continue
        players_out.append(
            {
                "name": identity.get("name") or key,
                "name_norm": key,
                "team": identity.get("team"),
                "position": identity.get("position"),
                "markets": consensus,
                "market_kinds": kinds,
                "prop_coverage": _prop_coverage(kinds),
                "sources": sorted(identity.get("sources") or []),
            }
        )

    teams_out: list[dict[str, Any]] = []
    for abbr, markets in sorted(team_lines.items()):
        consensus = _median_map(markets)
        if "points_scored" not in consensus and "points_scored_pg" in consensus:
            consensus["points_scored"] = consensus["points_scored_pg"] * 17.0
        if "total_yards" not in consensus:
            pass_yards = consensus.get("pass_yards")
            rush_yards = consensus.get("rush_yards")
            if pass_yards is not None and rush_yards is not None:
                consensus["total_yards"] = pass_yards + rush_yards
        teams_out.append({"abbr": abbr, "markets": consensus})

    return {
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [name for name, _ in files],
        "method": {
            "yards_tds_receptions": (
                "median of DraftKings/FanDuel/RotoWire/Oddschecker/FTA/"
                "ESPN-Fox/Action/Sharp-RG-SBR lines; book lines that conflict "
                "with the same source's projection (>50% and >100 absolute) "
                "are dropped; prediction-market-only / heavily skewed O/U "
                "prices (Kalshi/Polymarket thresholds) are dropped; numberFire "
                "projections fill only when no sportsbook quote remains; "
                "nickname aliases (Kenny/Kenneth, Chig/Chigoziem, Cam/Cameron) "
                "are merged before the median; checklist Vegas FP requires "
                "book-backed scoring coverage (projection-only rows are not ranked)"
            ),
            "volume_attempts_targets": (
                "not used for checklist ranks; public boards lack attempt/target "
                "season O/Us (VI/BettingPros/Unabated/Kalshi/Polymarket checked). "
                "Checklist uses Vegas yards + receptions instead."
            ),
            "extra_sources": (
                "vegasinsider, bettingpros, unabated, prediction_markets "
                "(kalshi/polymarket) merged when present"
            ),
            "team_points": "median Vegas-implied season points (Sharp/AN/volume scrape)",
            "team_yards": "numberFire team aggregates and/or QB pass-yard proxies",
        },
        "players": players_out,
        "teams": teams_out,
        "counts": {
            "players": len(players_out),
            "teams": len(teams_out),
            "raw_files": len(files),
        },
    }


def export_consensus(
    season: int = 2026,
    *,
    out_path: Path | None = None,
) -> Path:
    payload = build_consensus(season=season)
    destination = out_path or DRAFT_DATA_DIR / f"vegas_consensus_{season}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return destination


def main() -> None:
    path = export_consensus()
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"Wrote {path} players={payload['counts']['players']} "
        f"teams={payload['counts']['teams']} files={payload['counts']['raw_files']}"
    )


if __name__ == "__main__":
    main()
