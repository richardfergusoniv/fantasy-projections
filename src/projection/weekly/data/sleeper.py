"""Rate-conscious current NFL player snapshot from Sleeper's read-only API."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from datetime import date

import httpx
import polars as pl

from src.projection.weekly.config.paths import DATA_DIR, ensure_dirs

logger = logging.getLogger(__name__)

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_SOURCE_METADATA = {
    "source": "Sleeper read-only API",
    "url": SLEEPER_PLAYERS_URL,
    "access": "free for non-commercial use; no token; fetch at most daily",
    "license_note": "Commercial use requires permission from Sleeper",
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "sleeper_id": pl.Utf8,
            "player_name": pl.Utf8,
            "team": pl.Utf8,
            "position": pl.Utf8,
            "sleeper_active": pl.Boolean,
            "sleeper_status": pl.Utf8,
            "sleeper_injury_status": pl.Utf8,
            "sleeper_practice_participation": pl.Utf8,
            "sleeper_depth_rank": pl.Float64,
            "snapshot_date": pl.Utf8,
        }
    )


def parse_sleeper_players(payload: dict, *, snapshot_date: str | None = None) -> pl.DataFrame:
    """Normalize the API's player-id-keyed object into a compact table."""
    if not isinstance(payload, dict):
        return _empty()
    stamp = snapshot_date or date.today().isoformat()
    rows = []
    for sleeper_id, player in payload.items():
        if not isinstance(player, dict):
            continue
        pos = player.get("position")
        fantasy_positions = player.get("fantasy_positions") or []
        if pos not in {"QB", "RB", "WR", "TE"} and not any(
            p in {"QB", "RB", "WR", "TE"} for p in fantasy_positions
        ):
            continue
        rows.append(
            {
                "sleeper_id": str(sleeper_id),
                "player_name": player.get("full_name")
                or " ".join(
                    x
                    for x in (player.get("first_name"), player.get("last_name"))
                    if x
                ),
                "team": player.get("team"),
                "position": pos,
                "sleeper_active": player.get("active"),
                "sleeper_status": player.get("status"),
                "sleeper_injury_status": player.get("injury_status"),
                "sleeper_practice_participation": player.get("practice_participation"),
                "sleeper_depth_rank": player.get("depth_chart_position"),
                "snapshot_date": stamp,
            }
        )
    if not rows:
        return _empty()
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("sleeper_id").cast(pl.Utf8),
            pl.col("sleeper_depth_rank").cast(pl.Float64, strict=False),
        ]
    )


def fetch_sleeper_players(
    *, force: bool = False, timeout: float = 45.0
) -> pl.DataFrame:
    """Fetch at most once per UTC-local calendar day and cache as parquet."""
    ensure_dirs()
    stamp = date.today().isoformat()
    path = DATA_DIR / "raw" / f"sleeper_players_{stamp}.parquet"
    if path.exists() and not force:
        logger.info("Loading cached Sleeper player snapshot from %s", path)
        return pl.read_parquet(path)
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "fantasy-projections/0.1 (daily player snapshot)"},
        ) as client:
            response = client.get(SLEEPER_PLAYERS_URL)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning("Sleeper player snapshot failed: %s", exc)
        return pl.read_parquet(path) if path.exists() else _empty()
    df = parse_sleeper_players(payload, snapshot_date=stamp)
    if not df.is_empty():
        df.write_parquet(path)
        logger.info("Cached Sleeper player snapshot -> %s (%d rows)", path, df.height)
    return df
