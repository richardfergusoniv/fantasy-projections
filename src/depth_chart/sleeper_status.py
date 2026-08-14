"""Ingest Sleeper player injury/status fields for depth-chart refresh."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from src.comparison.sleeper_compare import (
    SNAPSHOT_DIR,
    _fetch_json,
    _normalize_name,
    PLAYERS_URL,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "sleeper")
STATUS_PARQUET = os.path.join(DATA_DIR, "sleeper_player_status.parquet")

# Sleeper often uses older/alternate club codes.
SLEEPER_TEAM_MAP = {
    "LAR": "LA",
    "GNB": "GB",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "JAC": "JAX",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "PHX": "ARI",
}


def normalize_team(team: str | None) -> str | None:
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return None
    t = str(team).strip().upper()
    return SLEEPER_TEAM_MAP.get(t, t)


def _rows_from_players_payload(players: dict, snapshot: dict) -> pd.DataFrame:
    rows = []
    for sid, p in players.items():
        if not isinstance(p, dict):
            continue
        rows.append({
            "sleeper_id": sid,
            "gsis_id": p.get("gsis_id") or None,
            "display_name": p.get("full_name"),
            "name_key": _normalize_name(p.get("full_name")),
            "team": normalize_team(p.get("team")),
            "position": p.get("position"),
            "injury_status": p.get("injury_status"),
            "injury_body_part": p.get("injury_body_part"),
            "injury_notes": p.get("injury_notes"),
            "injury_start_date": p.get("injury_start_date"),
            "roster_status": p.get("status"),
            "practice_participation": p.get("practice_participation"),
            "depth_chart_order": p.get("depth_chart_order"),
            "depth_chart_position": p.get("depth_chart_position"),
            "fetched_at": snapshot["fetched_at"],
            "snapshot_sha256": snapshot["sha256"],
            "snapshot_path": snapshot["raw_path"],
        })
    return pd.DataFrame(rows)


def ingest_sleeper_player_status(
    snapshot_dir: str = SNAPSHOT_DIR,
    out_path: str = STATUS_PARQUET,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch (or reuse snapshot) Sleeper players and write a status parquet.

    When ``force`` is False and a parquet already exists from today UTC, reuse
    it. Always reuses the content-addressed JSON snapshot cache inside
    ``_fetch_json``.
    """
    if (
        not force
        and os.path.exists(out_path)
    ):
        existing = pd.read_parquet(out_path)
        if "fetched_at" in existing.columns and not existing.empty:
            fetched = pd.to_datetime(existing["fetched_at"].iloc[0], utc=True, errors="coerce")
            now = datetime.now(timezone.utc)
            if pd.notna(fetched) and fetched.date() == now.date():
                return existing

    players, snapshot = _fetch_json(
        PLAYERS_URL, "players_nfl", snapshot_dir=snapshot_dir)
    if not isinstance(players, dict):
        raise ValueError("Sleeper players response was not a player-id mapping")
    df = _rows_from_players_payload(players, snapshot)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_sleeper_player_status(
    path: str = STATUS_PARQUET,
    ingest_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the latest sleeper_player_status parquet, ingesting if needed."""
    if not os.path.exists(path):
        if not ingest_if_missing:
            return pd.DataFrame()
        return ingest_sleeper_player_status(out_path=path)
    return pd.read_parquet(path)


def load_sleeper_status_from_snapshot_json(path: str) -> pd.DataFrame:
    """Test/helper: build a status frame from a raw Sleeper players JSON file."""
    with open(path, encoding="utf-8") as fh:
        players = json.load(fh)
    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sha256": "test",
        "raw_path": os.path.abspath(path),
    }
    return _rows_from_players_payload(players, snapshot)
