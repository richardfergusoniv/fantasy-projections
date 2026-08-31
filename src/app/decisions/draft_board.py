"""Draft board loader from sealed release / draft assistant data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.projection.active_release import read_active_pointer
from src.projection.contracts import REPO_ROOT


class DraftBoardService:
    def load_board(self, season: int = 2026, *, limit: int = 100) -> dict:
        pointer = read_active_pointer(season)
        players_path = self._players_path(season, pointer)
        if players_path is None or not players_path.exists():
            return {
                "entries": [],
                "source": "unavailable",
                "data_as_of": datetime.now(UTC).isoformat(),
                "projection_run_id": "fixture",
            }
        payload = json.loads(players_path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        players = payload.get("players", [])
        ranked = sorted(
            players,
            key=lambda row: float(row.get("vorp") or row.get("fantasy_pts_season") or 0.0),
            reverse=True,
        )
        entries = []
        for index, row in enumerate(ranked[:limit], start=1):
            entries.append(
                {
                    "player_id": row.get("player_id"),
                    "name": row.get("display_name") or row.get("name"),
                    "position": row.get("position"),
                    "team": row.get("team"),
                    "rank": index,
                    "tier": row.get("overall_tier") or row.get("pos_tier"),
                    "vorp": row.get("vorp"),
                    "points_mean": row.get("fantasy_pts_season"),
                }
            )
        namespace = pointer.get("namespace") if pointer else "fixture"
        return {
            "entries": entries,
            "source": "draft_assistant_release",
            "namespace": namespace,
            "data_as_of": meta.get("generated_at", datetime.now(UTC).isoformat()),
            "projection_run_id": f"preseason-{namespace}",
        }

    def _players_path(self, season: int, pointer: dict | None) -> Path | None:
        if pointer and pointer.get("public_urls", {}).get("players"):
            rel = pointer["public_urls"]["players"]
            if rel.startswith("data/"):
                rel = rel.removeprefix("data/")
            candidate = Path(REPO_ROOT) / "draft_assistant" / "data" / rel
            if candidate.exists():
                return candidate
        fallback = Path(REPO_ROOT) / "draft_assistant" / "data" / f"players_{season}.json"
        return fallback if fallback.exists() else None
