"""CollegeFootballData API loader for rookie college production."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
import polars as pl

from src.projection.weekly.config.paths import DATA_DIR, ensure_dirs

logger = logging.getLogger(__name__)

CFBD_BASE = "https://api.collegefootballdata.com"


class CFBDClient:
    """Thin CFBD API client with local caching and call budgeting."""

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        self.api_key = api_key or os.getenv("CFBD_API_KEY", "")
        self.timeout = timeout
        self._calls = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "CFBD_API_KEY is not set. Add it to .env "
                "(https://collegefootballdata.com/key)."
            )
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def get(self, path: str, params: dict | None = None) -> list | dict:
        url = f"{CFBD_BASE}{path}"
        with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
            resp = client.get(url, params=params or {})
            self._calls += 1
            if resp.status_code == 401:
                raise RuntimeError("CFBD API key rejected (401). Check CFBD_API_KEY.")
            if resp.status_code == 429:
                raise RuntimeError(
                    "CFBD rate/quota exceeded. Free tier is 1,000 calls/month."
                )
            resp.raise_for_status()
            return resp.json()

    @property
    def call_count(self) -> int:
        return self._calls


def _cache_path(name: str) -> Path:
    ensure_dirs()
    return DATA_DIR / "raw" / f"cfbd_{name}.parquet"


def load_player_season_stats(
    seasons: list[int],
    *,
    category: str = "receiving",
    force: bool = False,
    client: CFBDClient | None = None,
) -> pl.DataFrame:
    """Load season-level player stats for one category (receiving/rushing/passing)."""
    client = client or CFBDClient()
    path = _cache_path(f"player_season_{category}_{min(seasons)}_{max(seasons)}")
    if path.exists() and not force:
        return pl.read_parquet(path)

    if not client.available:
        logger.warning("No CFBD_API_KEY; returning empty college stats for %s", category)
        return pl.DataFrame()

    rows: list[dict] = []
    for season in seasons:
        try:
            payload = client.get(
                "/stats/player/season",
                params={"year": season, "category": category, "seasonType": "regular"},
            )
        except Exception as exc:
            logger.warning("CFBD %s %s failed: %s", category, season, exc)
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            rows.append(
                {
                    "season": season,
                    "category": category,
                    "player_id": str(item.get("playerId") or item.get("id") or ""),
                    "player": item.get("player") or item.get("name"),
                    "team": item.get("team"),
                    "conference": item.get("conference"),
                    "stat_type": item.get("statType") or item.get("stat"),
                    "stat": item.get("stat"),
                }
            )

    if not rows:
        return pl.DataFrame()

    long = pl.DataFrame(rows)
    # Pivot common stats when present
    wide = (
        long.filter(pl.col("stat_type").is_not_null())
        .with_columns(pl.col("stat").cast(pl.Float64, strict=False))
        .pivot(
            values="stat",
            index=["season", "player_id", "player", "team", "conference", "category"],
            on="stat_type",
            aggregate_function="first",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.write_parquet(path)
    logger.info("Cached CFBD %s stats -> %s (%d rows, %d API calls)", category, path, wide.height, client.call_count)
    return wide


def load_player_ppa(
    seasons: list[int],
    *,
    force: bool = False,
    client: CFBDClient | None = None,
) -> pl.DataFrame:
    """Load season-level player PPA if available on free tier."""
    client = client or CFBDClient()
    path = _cache_path(f"player_ppa_{min(seasons)}_{max(seasons)}")
    if path.exists() and not force:
        return pl.read_parquet(path)

    if not client.available:
        logger.warning("No CFBD_API_KEY; returning empty PPA table")
        return pl.DataFrame()

    rows: list[dict] = []
    for season in seasons:
        try:
            payload = client.get(
                "/ppa/players/season",
                params={"year": season, "excludeGarbageTime": True},
            )
        except Exception as exc:
            logger.warning("CFBD PPA %s failed (may be paid-tier): %s", season, exc)
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            average = item.get("averagePPA") or {}
            rows.append(
                {
                    "season": season,
                    "player_id": str(item.get("id") or item.get("playerId") or ""),
                    "player": item.get("name") or item.get("player"),
                    "position": item.get("position"),
                    "team": item.get("team"),
                    "conference": item.get("conference"),
                    "ppa_all": average.get("all"),
                    "ppa_pass": average.get("pass"),
                    "ppa_rush": average.get("rush"),
                    "ppa_first_down": average.get("firstDown"),
                    "ppa_second_down": average.get("secondDown"),
                    "ppa_third_down": average.get("thirdDown"),
                }
            )

    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    if not df.is_empty():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    return df


def load_recruiting(
    seasons: list[int],
    *,
    force: bool = False,
    client: CFBDClient | None = None,
) -> pl.DataFrame:
    """Load recruiting rankings (stars/rating) by recruit class year — 1 call per year."""
    client = client or CFBDClient()
    path = _cache_path(f"recruiting_{min(seasons)}_{max(seasons)}")
    if path.exists() and not force:
        return pl.read_parquet(path)

    if not client.available:
        logger.warning("No CFBD_API_KEY; returning empty recruiting table")
        return pl.DataFrame()

    rows: list[dict] = []
    for season in seasons:
        try:
            payload = client.get("/recruiting/players", params={"year": season})
        except Exception as exc:
            logger.warning("CFBD recruiting %s failed: %s", season, exc)
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            rows.append(
                {
                    "recruit_year": season,
                    "player": item.get("name") or item.get("athleteId"),
                    "position": item.get("position"),
                    "committed_to": item.get("committedTo") or item.get("school"),
                    "stars": item.get("stars"),
                    "rating": item.get("rating"),
                    "ranking": item.get("ranking"),
                }
            )

    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    if not df.is_empty():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        logger.info("Cached CFBD recruiting -> %s (%d rows)", path, df.height)
    return df


def load_team_context(
    seasons: list[int],
    *,
    force: bool = False,
    client: CFBDClient | None = None,
) -> pl.DataFrame:
    """Load compact team game-count and competition context (two calls/year).

    SRS is used because it has stable historical coverage and a small response.
    Failures are isolated by season/endpoint so rookie production remains usable
    if a rating endpoint is unavailable on the caller's access tier.
    """
    if not seasons:
        return pl.DataFrame()
    client = client or CFBDClient()
    path = _cache_path(f"team_context_{min(seasons)}_{max(seasons)}")
    if path.exists() and not force:
        return pl.read_parquet(path)
    if not client.available:
        logger.warning("No CFBD_API_KEY; returning empty team context")
        return pl.DataFrame()

    counts: dict[tuple[int, str], int] = {}
    ratings: dict[tuple[int, str], float | None] = {}
    for season in seasons:
        try:
            games = client.get(
                "/games", params={"year": season, "seasonType": "regular"}
            )
            if isinstance(games, list):
                for game in games:
                    for key in ("homeTeam", "awayTeam"):
                        team = game.get(key)
                        if team:
                            counts[(season, str(team))] = counts.get((season, str(team)), 0) + 1
        except Exception as exc:
            logger.warning("CFBD team games %s failed: %s", season, exc)
        try:
            srs = client.get("/ratings/srs", params={"year": season})
            if isinstance(srs, list):
                for item in srs:
                    team = item.get("team")
                    if team:
                        ratings[(season, str(team))] = item.get("rating")
        except Exception as exc:
            logger.warning("CFBD SRS %s failed: %s", season, exc)

    keys = sorted(set(counts) | set(ratings))
    rows = [
        {
            "college_season": season,
            "team": team,
            "college_team_games": counts.get((season, team)),
            "college_team_srs": ratings.get((season, team)),
        }
        for season, team in keys
    ]
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    if not df.is_empty():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    return df


def load_college_features_for_drafted(
    draft_picks: pl.DataFrame,
    *,
    seasons: list[int] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Build a compact college feature table keyed for join to NFL draft picks.

    Uses season-level receiving/rushing stats, season PPA, and recruiting stars.
    Designed to minimize CFBD API calls (batch by season, not by player).
    """
    if seasons is None:
        if "season" in draft_picks.columns:
            seasons = sorted(draft_picks["season"].unique().to_list())
        else:
            seasons = list(range(2015, 2026))

    # College production seasons: draft year and a few prior years
    college_seasons = sorted(set(seasons) | {s - 1 for s in seasons} | {s - 2 for s in seasons})
    # Recruiting classes often 2–4 years before draft
    recruit_seasons = sorted(
        set(range(min(seasons) - 4, max(seasons))) | set(college_seasons)
    )

    client = CFBDClient()
    receiving = load_player_season_stats(college_seasons, category="receiving", force=force, client=client)
    rushing = load_player_season_stats(college_seasons, category="rushing", force=force, client=client)
    ppa = load_player_ppa(college_seasons, force=force, client=client)
    recruiting = load_recruiting(recruit_seasons, force=force, client=client)
    team_context = load_team_context(college_seasons, force=force, client=client)

    frames = [f for f in (receiving, rushing) if not f.is_empty()]
    if not frames:
        return pl.DataFrame(
            schema={
                "college_player_id": pl.Utf8,
                "college_player": pl.Utf8,
                "college_season": pl.Int64,
                "rec_yards": pl.Float64,
                "rec_tds": pl.Float64,
                "receptions": pl.Float64,
                "rush_yards": pl.Float64,
                "rush_tds": pl.Float64,
                "ppa_all": pl.Float64,
                "college_recruiting_stars": pl.Float64,
                "college_recruiting_rating": pl.Float64,
            }
        )

    def _norm(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
        rename_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in {"yards", "yds"}:
                rename_map[col] = f"{prefix}_yards"
            elif lower in {"td", "tds", "touchdowns"}:
                rename_map[col] = f"{prefix}_tds"
            elif lower in {"receptions", "rec"}:
                rename_map[col] = "receptions"
            elif lower in {"attempts", "att", "carries"}:
                rename_map[col] = f"{prefix}_attempts"
        out = df.rename(rename_map) if rename_map else df
        if "player_id" in out.columns:
            out = out.rename(
                {"player_id": "college_player_id", "player": "college_player", "season": "college_season"}
            )
        return out

    receiving_n = _norm(receiving, "rec") if not receiving.is_empty() else None
    rushing_n = _norm(rushing, "rush") if not rushing.is_empty() else None

    base = receiving_n if receiving_n is not None else rushing_n
    assert base is not None
    if receiving_n is not None and rushing_n is not None:
        join_keys = [
            c
            for c in ("college_season", "college_player_id", "team")
            if c in receiving_n.columns and c in rushing_n.columns
        ]
        college = receiving_n.join(rushing_n, on=join_keys, how="full", coalesce=True)
    else:
        college = base

    if not ppa.is_empty() and "player_id" in ppa.columns:
        ppa_n = ppa.rename({"player_id": "college_player_id", "season": "college_season"})
        keys = [
            c
            for c in ("college_season", "college_player_id")
            if c in college.columns and c in ppa_n.columns
        ]
        ppa_cols = keys + [c for c in ppa_n.columns if c.startswith("ppa_")]
        college = college.join(ppa_n.select(ppa_cols), on=keys, how="left")

    if not recruiting.is_empty() and "player" in recruiting.columns:
        rec = recruiting.with_columns(
            pl.col("player").str.to_lowercase().str.strip_chars().alias("_join_name")
        )
        # Keep highest stars per player name
        rec = (
            rec.sort(["stars", "rating"], descending=True)
            .unique(subset=["_join_name"], keep="first")
            .select(
                [
                    "_join_name",
                    pl.col("stars").cast(pl.Float64).alias("college_recruiting_stars"),
                    pl.col("rating").cast(pl.Float64).alias("college_recruiting_rating"),
                ]
            )
        )
        if "college_player" in college.columns:
            college = college.with_columns(
                pl.col("college_player").str.to_lowercase().str.strip_chars().alias("_join_name")
            ).join(rec, on="_join_name", how="left").drop("_join_name")
        else:
            college = college.with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("college_recruiting_stars"),
                    pl.lit(None).cast(pl.Float64).alias("college_recruiting_rating"),
                ]
            )
    else:
        college = college.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("college_recruiting_stars"),
                pl.lit(None).cast(pl.Float64).alias("college_recruiting_rating"),
            ]
        )

    # Build a stable one-row-per-prospect table with production shares,
    # per-team-game rates, final-year/career summaries, breakout hooks, and
    # competition context. Imported lazily to keep the data layer lightweight.
    from src.projection.weekly.features.rookie_college import build_cfbd_prospect_features

    enriched = build_cfbd_prospect_features(college, team_context=team_context)
    logger.info(
        "College prospect features ready: %d rows (%d CFBD calls)",
        enriched.height,
        client.call_count,
    )
    return enriched
