"""Merge multi-book Vegas raw scrapes into a sealed consensus board.

Yardage / TD / receptions prefer the median of public sportsbook O/U lines.
Pass attempts, rush attempts, and targets are almost never posted as season
O/Us; for those we use numberFire remaining-season projections captured in the
Action Network + numberFire scrape.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from src.draft_assistant.market_adp import normalize_player_name

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


def _line_value(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    if isinstance(raw, dict):
        for key in ("line", "value", "ou", "total", "projection"):
            if key in raw:
                parsed = _line_value(raw[key])
                if parsed is not None:
                    return parsed
        return None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


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
    store: dict[str, dict[str, list[float]]],
    meta: dict[str, dict[str, Any]],
    *,
    name: str,
    team: str | None,
    position: str | None,
    market: str,
    value: float,
    source: str,
) -> None:
    canon = PLAYER_MARKET_ALIASES.get(market)
    if not canon:
        return
    key = normalize_player_name(name)
    store.setdefault(key, {}).setdefault(canon, []).append(value)
    identity = meta.setdefault(
        key,
        {"name": name, "team": team, "position": position, "sources": set()},
    )
    if team and not identity.get("team"):
        identity["team"] = team
    if position and not identity.get("position"):
        identity["position"] = position
    identity["sources"].add(source)


def _collect_player_lines(
    files: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, Any]]]:
    lines: dict[str, dict[str, list[float]]] = {}
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
                    value = _line_value(nf.get(nf_key))
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
                        )
            for market, raw in markets.items():
                value = _line_value(raw)
                if value is None:
                    continue
                _add_line(
                    lines,
                    meta,
                    name=name,
                    team=team,
                    position=position,
                    market=str(market),
                    value=value,
                    source=source,
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


def _median_map(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(median(vals)) for key, vals in values.items() if vals}


def build_consensus(*, season: int = 2026) -> dict[str, Any]:
    files = _load_raw_files()
    player_lines, player_meta = _collect_player_lines(files)
    team_lines = _collect_team_lines(files)

    players_out: list[dict[str, Any]] = []
    for key, markets in sorted(player_lines.items()):
        identity = player_meta.get(key) or {}
        consensus = _median_map(markets)
        if not consensus:
            continue
        players_out.append(
            {
                "name": identity.get("name") or key,
                "name_norm": key,
                "team": identity.get("team"),
                "position": identity.get("position"),
                "markets": consensus,
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
                "ESPN-Fox/Action/Sharp-RG-SBR lines"
            ),
            "volume_attempts_targets": (
                "numberFire remaining-season projections "
                "(season O/U boards unavailable publicly)"
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
