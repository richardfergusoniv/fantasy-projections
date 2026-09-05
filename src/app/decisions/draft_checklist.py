"""Load sealed draft checklist for the PWA (market ranks + context ranks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.app.persistence.models import PlayerIdentity
from src.projection.active_release import read_active_pointer
from src.projection.contracts import REPO_ROOT

DRAFT_DATA_DIR = Path(REPO_ROOT) / "draft_assistant" / "data"


def _checklist_candidates(season: int) -> list[Path]:
    paths: list[Path] = []
    try:
        pointer = read_active_pointer(season)
    except Exception:
        pointer = None
    if pointer:
        rel = (pointer.get("public_urls") or {}).get("draft_checklist")
        if rel:
            # Pointer paths look like ``data/releases/...`` under draft_assistant/.
            cleaned = rel.removeprefix("data/") if rel.startswith("data/") else rel
            paths.append(DRAFT_DATA_DIR / cleaned)
            paths.append(Path(REPO_ROOT) / "draft_assistant" / rel)
        public_base = pointer.get("public_base")
        if public_base:
            cleaned_base = (
                public_base.removeprefix("data/")
                if public_base.startswith("data/")
                else public_base
            )
            paths.append(DRAFT_DATA_DIR / cleaned_base / f"draft_checklist_{season}.json")
    paths.append(DRAFT_DATA_DIR / f"draft_checklist_{season}.json")
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def load_checklist_payload(season: int = 2026) -> dict[str, Any] | None:
    """Load the sealed checklist for ``season``, falling back to 2026 if needed.

    Historical Sleeper leagues often carry a prior ``season`` value even when the
    only published draft board is for the current fantasy year. Prefer an exact
    season match, then serve the 2026 artifact rather than an empty checklist.
    """
    seasons = [int(season)]
    if int(season) != 2026:
        seasons.append(2026)
    for candidate_season in seasons:
        for path in _checklist_candidates(candidate_season):
            if not path.is_file():
                continue
            try:
                with path.open(encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("players"), list):
                continue
            if candidate_season != int(season):
                meta = dict(payload.get("meta") or {})
                meta["season_fallback"] = {
                    "requested": int(season),
                    "served": candidate_season,
                    "reason": "requested_season_checklist_missing",
                }
                payload["meta"] = meta
            return payload
    return None



def _coerce_rank_map(raw: Any) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for key, value in dict(raw or {}).items():
        if value is None or value == "":
            ranks[str(key)] = None
            continue
        try:
            ranks[str(key)] = int(value)
        except (TypeError, ValueError):
            ranks[str(key)] = None
    return ranks


class DraftChecklistService:
    """Serve the committed checklist JSON for a league."""

    def __init__(self, db: Session | None = None):
        self.db = db

    def _sleeper_map(self) -> dict[str, str]:
        if self.db is None:
            return {}
        try:
            rows = (
                self.db.query(PlayerIdentity.gsis_id, PlayerIdentity.sleeper_id)
                .filter(PlayerIdentity.gsis_id.isnot(None))
                .all()
            )
        except Exception:
            return {}
        return {
            str(gsis): str(sleeper)
            for gsis, sleeper in rows
            if gsis and sleeper
        }

    def load(self, season: int = 2026, *, league_id: str | None = None) -> dict[str, Any]:
        payload = load_checklist_payload(season)
        if payload is None:
            return {
                "league_id": league_id,
                "season": season,
                "available": False,
                "entries": [],
                "teams": [],
                "criteria_by_position": {},
                "meta": {
                    "error": "draft_checklist_missing",
                    "hint": (
                        "Publish draft_checklist_{season}.json under "
                        "draft_assistant/data/ (and the active release "
                        "public_base) via checklist_prepare."
                    ).format(season=season),
                },
            }

        sleeper_by_gsis = self._sleeper_map()
        entries = []
        for row in payload.get("players") or []:
            player_id = str(row.get("player_id") or "")
            entries.append(
                {
                    "player_id": player_id,
                    "sleeper_id": sleeper_by_gsis.get(player_id),
                    "name": row.get("name") or player_id,
                    "position": row.get("position"),
                    "team": row.get("team"),
                    "adp": row.get("adp"),
                    "ecr": row.get("ecr"),
                    "prior_pts": row.get("prior_pts"),
                    "vegas_fp": row.get("vegas_fp"),
                    "rank_tier": row.get("rank_tier"),
                    "pos_market_rank": row.get("pos_market_rank"),
                    "unranked_break": bool(row.get("unranked_break")),
                    "ranks": _coerce_rank_map(row.get("ranks")),
                    "checks": dict(row.get("checks") or {}),
                    "sentiment": dict(row.get("sentiment") or {}) or None,
                }
            )

        meta = dict(payload.get("meta") or {})
        served_season = int(payload.get("season") or season)
        return {
            "league_id": league_id,
            "season": served_season,
            "available": True,
            "entries": entries,
            "teams": list(payload.get("teams") or []),
            "criteria_by_position": dict(payload.get("criteria_by_position") or {}),
            "criteria_labels": dict(
                meta.get("criteria_labels")
                or payload.get("criteria_labels")
                or {}
            ),
            "meta": meta,
            "data_as_of": meta.get("generated_at"),
            "projection_run_id": f"checklist-{served_season}",
        }
