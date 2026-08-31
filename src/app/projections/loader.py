"""Load player projections from sealed release bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from src.projection.active_release import read_active_pointer
from src.projection.contracts import REPO_ROOT


@dataclass(frozen=True)
class PlayerSummary:
    player_id: str
    name: str
    position: str
    team: str | None
    mean_points: float
    quantiles: dict[str, float]
    availability_probability: float = 1.0


class ReleaseBundleLoader:
    def __init__(self, season: int = 2026) -> None:
        self.season = season
        self._players: dict[str, PlayerSummary] | None = None
        self._meta: dict | None = None

    def players_path(self) -> Path | None:
        pointer = read_active_pointer(self.season)
        if pointer is None:
            return None
        public_urls = pointer.get("public_urls") or {}
        players_rel = public_urls.get("players")
        if not players_rel:
            namespace = pointer["namespace"]
            players_rel = f"data/releases/{namespace}/players_{self.season}.json"
        if players_rel.startswith("data/"):
            path = Path(REPO_ROOT) / "draft_assistant" / players_rel
        else:
            path = Path(REPO_ROOT) / "draft_assistant" / "data" / players_rel
        if path.exists():
            return path
        alt = Path(REPO_ROOT) / players_rel
        return alt if alt.exists() else None

    def load(self) -> dict[str, PlayerSummary]:
        if self._players is not None:
            return self._players
        path = self.players_path()
        if path is None:
            self._players = {}
            return self._players
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._meta = payload.get("meta", {})
        index: dict[str, PlayerSummary] = {}
        for row in payload.get("players", []):
            player_id = str(row.get("player_id", ""))
            if not player_id:
                continue
            games = float(row.get("projected_games") or 17.0)
            season_pts = float(row.get("fantasy_pts_season") or row.get("fantasy_pts", 0.0) * games)
            per_game = season_pts / games if games else float(row.get("fantasy_pts", 0.0))
            index[player_id] = PlayerSummary(
                player_id=player_id,
                name=str(row.get("display_name", player_id)),
                position=str(row.get("position", "RB")),
                team=row.get("team"),
                mean_points=per_game,
                quantiles={
                    "0.1": float(row.get("fantasy_pts_p10", per_game * 0.7)) / games if games else per_game * 0.7,
                    "0.5": float(row.get("fantasy_pts_p50", season_pts)) / games if games else per_game,
                    "0.9": float(row.get("fantasy_pts_p90", per_game * 1.3)) / games if games else per_game * 1.3,
                },
                availability_probability=min(1.0, games / 17.0),
            )
        self._players = index
        return self._players

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self.load()
        return self._meta or {}

    def get(self, player_id: str) -> PlayerSummary | None:
        return self.load().get(player_id)

    def available_pool(self, rostered_ids: set[str], positions: set[str] | None = None) -> list[PlayerSummary]:
        pool = []
        for player_id, summary in self.load().items():
            if player_id in rostered_ids:
                continue
            if positions and summary.position not in positions:
                continue
            pool.append(summary)
        pool.sort(key=lambda p: p.mean_points, reverse=True)
        return pool[:100]

    def as_of(self) -> str:
        generated = self.meta.get("generated_at")
        return generated or datetime.now(UTC).isoformat()


@lru_cache
def get_bundle_loader(season: int = 2026) -> ReleaseBundleLoader:
    return ReleaseBundleLoader(season=season)
