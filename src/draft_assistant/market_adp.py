"""Platform ADP + FantasyPros ECR market average for checklist ordering.

Sources (equal weight when present):
- ESPN PPR ADP
- Fantasy Football Calculator PPR ADP
- MyFantasyLeague ADP
- FantasyPros ECR (from sealed comparison_{season}.json)

Missing sources are skipped per player; the average uses whatever is available.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}


def normalize_player_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace(".", "").replace("'", "").replace("-", " ")
    for suffix in (" jr", " sr", " iii", " ii", " iv"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return " ".join(text.split())


def mfl_name_to_display(name: str) -> str:
    """Convert ``Last, First`` to ``First Last``."""
    raw = str(name or "").strip()
    if "," not in raw:
        return raw
    last, first = raw.split(",", 1)
    return f"{first.strip()} {last.strip()}".strip()


def _http_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    req_headers = {"User-Agent": "fantasy-projections-checklist/1.0"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_espn_ppr_adp(*, limit: int = 500) -> dict[tuple[str, str], float]:
    filt = {
        "players": {
            "limit": limit,
            "sortDraftRanks": {"sortPriority": 1, "sortAsc": True, "value": "PPR"},
            "sortPercOwned": {"sortPriority": 2, "sortAsc": False},
            "filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]},
        }
    }
    url = (
        "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/"
        "segments/0/leaguedefaults/3?view=kona_player_info"
    )
    payload = _http_json(url, headers={"X-Fantasy-Filter": json.dumps(filt)})
    out: dict[tuple[str, str], float] = {}
    for row in payload.get("players") or []:
        player = row.get("player") or {}
        pos = ESPN_POS.get(player.get("defaultPositionId"))
        if not pos:
            continue
        ownership = player.get("ownership") or {}
        adp = ownership.get("averageDraftPosition")
        name = player.get("fullName")
        if adp is None or not name:
            continue
        out[(pos, normalize_player_name(name))] = float(adp)
    return out


def fetch_ffc_ppr_adp(*, teams: int = 12) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams={teams}"
    payload = _http_json(url)
    out: dict[tuple[str, str], float] = {}
    for row in payload.get("players") or []:
        pos = str(row.get("position") or "")
        name = str(row.get("name") or "")
        adp = row.get("adp")
        if pos not in {"QB", "RB", "WR", "TE"} or adp is None or not name:
            continue
        out[(pos, normalize_player_name(name))] = float(adp)
    meta = dict(payload.get("meta") or {})
    meta["url"] = url
    return out, meta


def fetch_mfl_adp() -> dict[tuple[str, str], float]:
    adp_payload = _http_json(
        "https://api.myfantasyleague.com/2026/export?TYPE=adp&L=&JSON=1"
    )
    players_payload = _http_json(
        "https://api.myfantasyleague.com/2026/export?TYPE=players&L=&JSON=1"
    )
    by_id = {
        str(row.get("id")): row
        for row in ((players_payload.get("players") or {}).get("player") or [])
    }
    out: dict[tuple[str, str], float] = {}
    for row in (adp_payload.get("adp") or {}).get("player") or []:
        meta = by_id.get(str(row.get("id")))
        if not meta:
            continue
        pos = str(meta.get("position") or "")
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue
        name = mfl_name_to_display(str(meta.get("name") or ""))
        try:
            adp = float(row.get("averagePick"))
        except (TypeError, ValueError):
            continue
        out[(pos, normalize_player_name(name))] = adp
    return out


def load_comparison_ecr(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[str, float] = {}
    for row in payload.get("players") or []:
        pid = row.get("player_id")
        ecr = row.get("ecr")
        if pid is None or ecr is None:
            continue
        try:
            out[str(pid)] = float(ecr)
        except (TypeError, ValueError):
            continue
    return out


def market_components_for_player(
    *,
    position: str,
    name: str,
    player_id: str,
    espn: dict[tuple[str, str], float],
    ffc: dict[tuple[str, str], float],
    mfl: dict[tuple[str, str], float],
    ecr_by_id: dict[str, float],
) -> dict[str, float | None]:
    key = (position, normalize_player_name(name))
    return {
        "adp_espn": espn.get(key),
        "adp_ffc": ffc.get(key),
        "adp_mfl": mfl.get(key),
        "ecr": ecr_by_id.get(str(player_id)),
    }


def average_market_value(components: dict[str, float | None]) -> float | None:
    values = [float(v) for v in components.values() if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def fetch_market_maps(
    *,
    comparison_path: Path,
    ffc_teams: int = 12,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, Any],
]:
    espn = fetch_espn_ppr_adp()
    ffc, ffc_meta = fetch_ffc_ppr_adp(teams=ffc_teams)
    mfl = fetch_mfl_adp()
    ecr = load_comparison_ecr(comparison_path)
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "espn": {"scoring": "ppr", "matched": len(espn)},
            "ffc": {**ffc_meta, "matched": len(ffc)},
            "mfl": {"matched": len(mfl)},
            "fantasypros_ecr": {
                "path": str(comparison_path.name),
                "matched": len(ecr),
            },
        },
        "formula": "mean(available among ESPN ADP, FFC ADP, MFL ADP, FantasyPros ECR)",
    }
    return espn, ffc, mfl, ecr, meta
