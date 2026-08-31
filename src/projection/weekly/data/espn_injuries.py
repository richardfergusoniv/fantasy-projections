"""ESPN unofficial injury feed for 2025+ seasons."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from src.projection.weekly.config.paths import DATA_DIR, ensure_dirs
from src.projection.weekly.data.ids import coerce_id_columns

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
)

# Athlete id is often only present in playercard links, not athlete.id.
_ATHLETE_ID_PATTERNS = (
    re.compile(r"/id/(\d+)"),
    re.compile(r"[~&]a:(\d+)"),
)


def _athlete_espn_id(athlete: dict, inj: dict | None = None) -> str:
    """Extract ESPN athlete id (not the injury-report id)."""
    raw = athlete.get("id")
    if raw is not None and str(raw).strip() and not str(raw).strip().startswith("-"):
        return str(raw).strip()
    for link in athlete.get("links") or []:
        href = str(link.get("href") or "")
        for pat in _ATHLETE_ID_PATTERNS:
            m = pat.search(href)
            if m:
                return m.group(1)
    # Last resort: positive inj.id only (negative ids are injury-row keys, not athletes)
    if inj:
        inj_id = str(inj.get("id") or "").strip()
        if inj_id.isdigit():
            return inj_id
    return ""


def _cache_path() -> Path:
    ensure_dirs()
    return DATA_DIR / "raw" / f"espn_injuries_{date.today().isoformat()}.parquet"


def fetch_espn_injuries(*, force: bool = False, timeout: float = 30.0) -> pl.DataFrame:
    """Fetch current ESPN NFL injury report and cache daily."""
    path = _cache_path()
    if path.exists() and not force:
        logger.info("Loading cached ESPN injuries from %s", path)
        return pl.read_parquet(path)

    logger.info("Fetching ESPN injuries from %s", ESPN_INJURIES_URL)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(ESPN_INJURIES_URL)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning("ESPN injuries fetch failed: %s", exc)
        if path.exists():
            return pl.read_parquet(path)
        return pl.DataFrame(
            schema={
                "espn_id": pl.Utf8,
                "player_name": pl.Utf8,
                "team": pl.Utf8,
                "position": pl.Utf8,
                "status": pl.Utf8,
                "injury": pl.Utf8,
                "date": pl.Utf8,
            }
        )

    rows: list[dict] = []
    for team_block in payload.get("injuries", payload.get("items", [])):
        team_abbr = (
            team_block.get("team", {}).get("abbreviation")
            if isinstance(team_block.get("team"), dict)
            else team_block.get("abbreviation")
        )
        injuries = team_block.get("injuries") or team_block.get("items") or []
        if not injuries and "athlete" in team_block:
            injuries = [team_block]
        for inj in injuries:
            athlete = inj.get("athlete") or {}
            rows.append(
                {
                    "espn_id": _athlete_espn_id(athlete, inj),
                    "player_name": athlete.get("displayName") or inj.get("displayName"),
                    "team": team_abbr,
                    "position": (athlete.get("position") or {}).get("abbreviation")
                    if isinstance(athlete.get("position"), dict)
                    else athlete.get("position"),
                    "status": inj.get("status") or (inj.get("type") or {}).get("description"),
                    "injury": inj.get("longComment")
                    or inj.get("shortComment")
                    or (inj.get("details") or {}).get("type"),
                    "date": inj.get("date") or date.today().isoformat(),
                }
            )

    df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "espn_id": pl.Utf8,
            "player_name": pl.Utf8,
            "team": pl.Utf8,
            "position": pl.Utf8,
            "status": pl.Utf8,
            "injury": pl.Utf8,
            "date": pl.Utf8,
        }
    )
    df = coerce_id_columns(df, ("espn_id",))
    df = df.filter(pl.col("espn_id").is_not_null() & (pl.col("espn_id") != ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Cached ESPN injuries -> %s (%d rows)", path, df.height)
    return df


def injury_status_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add binary / ordinal injury flags from status text."""
    if df.is_empty() or "status" not in df.columns:
        return df
    status = pl.col("status").cast(pl.Utf8).str.to_lowercase().fill_null("")
    inactive = (
        status.str.contains("out")
        | status.str.contains("injured reserve")
        | (status == "ir")
        | status.str.contains("suspend")
    )
    return df.with_columns(
        [
            inactive.alias("is_out"),
            status.str.contains("doubt").alias("is_doubtful"),
            status.str.contains("question").alias("is_questionable"),
            pl.when(inactive)
            .then(pl.lit(0.0))
            .when(status.str.contains("doubt"))
            .then(pl.lit(0.25))
            .when(status.str.contains("question"))
            .then(pl.lit(0.75))
            .otherwise(pl.lit(1.0))
            .alias("play_prob"),
        ]
    )
