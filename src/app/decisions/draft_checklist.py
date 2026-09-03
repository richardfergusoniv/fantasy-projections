"""Load sealed draft checklist for the PWA (market ranks + context checks)."""

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
    for path in _checklist_candidates(season):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return None


class DraftChecklistService:
    """Serve the committed checklist JSON for a league."""

    def __init__(self, db: Session | None = None):
        self.db = db

    def _sleeper_map(self) -> dict[str, str]:
        if self.db is None:
            return {}
        rows = (
            self.db.query(PlayerIdentity.gsis_id, PlayerIdentity.sleeper_id)
            .filter(PlayerIdentity.gsis_id.isnot(None))
            .all()
        )
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
                "meta": {"error": "draft_checklist_missing"},
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
                    "rank_tier": row.get("rank_tier"),
                    "pos_market_rank": row.get("pos_market_rank"),
                    "unranked_break": bool(row.get("unranked_break")),
                    "checks": dict(row.get("checks") or {}),
                }
            )

        meta = dict(payload.get("meta") or {})
        return {
            "league_id": league_id,
            "season": season,
            "available": True,
            "entries": entries,
            "teams": list(payload.get("teams") or []),
            "criteria_by_position": dict(payload.get("criteria_by_position") or {}),
            "criteria_labels": dict(meta.get("criteria_labels") or {}),
            "meta": meta,
            "data_as_of": meta.get("generated_at"),
            "projection_run_id": f"checklist-{season}",
        }
