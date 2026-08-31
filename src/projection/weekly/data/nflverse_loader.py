"""nflverse / nflreadpy loaders with local parquet caching."""
# Ported from fantasy-projections-2 (team-first weekly v2). See docs/WEEKLY_V2_PORT_PROVENANCE.md.

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from src.projection.weekly.config.paths import DATA_DIR, TRAIN_START_SEASON, VALIDATE_SEASON, ensure_dirs
from src.projection.weekly.data.ids import coerce_id_columns, coalesce_player_id, normalize_position

logger = logging.getLogger(__name__)


def _to_polars(obj: Any) -> pl.DataFrame:
    if isinstance(obj, pl.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return pl.from_pandas(obj.to_pandas())
    if hasattr(obj, "to_dict"):
        try:
            return pl.from_pandas(obj)
        except Exception:
            pass
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return pl.from_pandas(obj)
    raise TypeError(f"Cannot convert {type(obj)} to polars DataFrame")


def _season_list(
    seasons: int | Sequence[int] | None,
    *,
    start: int | None = None,
    end: int | None = None,
) -> list[int]:
    if seasons is None:
        start = start if start is not None else TRAIN_START_SEASON
        end = end if end is not None else VALIDATE_SEASON
        return list(range(start, end + 1))
    if isinstance(seasons, int):
        return [seasons]
    return list(seasons)


def cache_path(name: str, seasons: Sequence[int] | None = None) -> Path:
    ensure_dirs()
    if seasons is None:
        return DATA_DIR / "raw" / f"{name}.parquet"
    s0, s1 = min(seasons), max(seasons)
    return DATA_DIR / "raw" / f"{name}_{s0}_{s1}.parquet"


def load_or_fetch(
    name: str,
    fetch_fn: Callable[[], pl.DataFrame],
    *,
    seasons: Sequence[int] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    path = cache_path(name, seasons)
    if path.exists() and not force:
        logger.info("Loading cached %s from %s", name, path)
        return pl.read_parquet(path)
    logger.info("Fetching %s ...", name)
    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Cached %s -> %s (%d rows)", name, path, df.height)
    return df


def _import_nflreadpy():
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise ImportError(
            "nflreadpy is required. Install with: pip install -e '.[dev]'"
        ) from exc
    return nfl


def load_player_stats(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_player_stats(seasons=seasons_list)
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        if "player_id" in df.columns:
            df = df.with_columns(pl.col("player_id").cast(pl.Utf8).alias("gsis_id"))
        df = coalesce_player_id(df)
        df = normalize_position(df)
        # Keep regular season only when season_type / game_type present
        for col, vals in (("season_type", ["REG"]), ("game_type", ["REG"])):
            if col in df.columns:
                df = df.filter(pl.col(col).is_in(vals))
                break
        return df

    return load_or_fetch("player_stats", fetch, seasons=seasons_list, force=force)


def load_schedules(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_schedules(seasons=seasons_list)
        df = _to_polars(raw)
        if "game_type" in df.columns:
            df = df.filter(pl.col("game_type") == "REG")
        return df

    return load_or_fetch("schedules", fetch, seasons=seasons_list, force=force)


def load_snap_counts(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons, start=max(TRAIN_START_SEASON, 2012))
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_snap_counts(seasons=seasons_list)
        df = _to_polars(raw)
        df = coerce_id_columns(df, ("pfr_player_id", "pfr_id", "gsis_id"))
        if "pfr_player_id" in df.columns and "pfr_id" not in df.columns:
            df = df.with_columns(pl.col("pfr_player_id").alias("pfr_id"))
        return df

    return load_or_fetch("snap_counts", fetch, seasons=seasons_list, force=force)


def load_rosters(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_rosters(seasons=seasons_list)
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        df = coalesce_player_id(df) if "gsis_id" in df.columns or "player_id" in df.columns else df
        return normalize_position(df)

    return load_or_fetch("rosters", fetch, seasons=seasons_list, force=force)


def load_players(*, force: bool = False) -> pl.DataFrame:
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_players()
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        return normalize_position(df)

    return load_or_fetch("players", fetch, force=force)


def load_ff_playerids(*, force: bool = False) -> pl.DataFrame:
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        # Prefer load_ff_playerids; fall back to load_players
        if hasattr(nfl, "load_ff_playerids"):
            raw = nfl.load_ff_playerids()
        else:
            raw = nfl.load_players()
        df = _to_polars(raw)
        return coerce_id_columns(df)

    return load_or_fetch("ff_playerids", fetch, force=force)


def load_draft_picks(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons, start=TRAIN_START_SEASON - 5)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_draft_picks(seasons=seasons_list)
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        if "gsis_id" in df.columns:
            df = coalesce_player_id(df)
        return normalize_position(df, "position")

    return load_or_fetch("draft_picks", fetch, seasons=seasons_list, force=force)


def load_combine(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons, start=TRAIN_START_SEASON - 5)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_combine(seasons=seasons_list)
        df = _to_polars(raw)
        return coerce_id_columns(df)

    return load_or_fetch("combine", fetch, seasons=seasons_list, force=force)


def load_team_stats(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if hasattr(nfl, "load_team_stats"):
            raw = nfl.load_team_stats(seasons=seasons_list)
        else:
            # Derive team totals from player stats if team_stats loader missing
            ps = load_player_stats(seasons_list, force=force)
            raw = (
                ps.group_by(["season", "week", "recent_team"])
                .agg(
                    [
                        pl.col("attempts").sum().alias("attempts"),
                        pl.col("carries").sum().alias("carries"),
                        pl.col("passing_yards").sum().alias("passing_yards"),
                        pl.col("rushing_yards").sum().alias("rushing_yards"),
                        pl.col("passing_tds").sum().alias("passing_tds"),
                        pl.col("rushing_tds").sum().alias("rushing_tds"),
                        pl.col("targets").sum().alias("targets"),
                    ]
                )
                .rename({"recent_team": "team"})
            )
            return _to_polars(raw) if not isinstance(raw, pl.DataFrame) else raw
        return _to_polars(raw)

    return load_or_fetch("team_stats", fetch, seasons=seasons_list, force=force)


def load_injuries_nflverse(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Weekly injury reports from nflverse (prefer over ESPN for completed seasons)."""
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_injuries"):
            return pl.DataFrame()
        raw = nfl.load_injuries(seasons=seasons_list)
        df = _to_polars(raw)
        return coerce_id_columns(df)

    try:
        return load_or_fetch("injuries_nflverse", fetch, seasons=seasons_list, force=force)
    except Exception as exc:
        logger.warning("Could not load nflverse injuries: %s", exc)
        return pl.DataFrame()


def load_depth_charts(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Load depth charts. Pre-2025: weekly. 2025+: timestamped snapshots (dt)."""
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        raw = nfl.load_depth_charts(seasons=seasons_list)
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        if "club_code" in df.columns and "team" not in df.columns:
            df = df.with_columns(pl.col("club_code").alias("team"))
        return df

    return load_or_fetch("depth_charts", fetch, seasons=seasons_list, force=force)


def load_ff_opportunity(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Weekly expected fantasy points / yards (ffopportunity, CC-BY-SA)."""
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_ff_opportunity"):
            logger.warning("nflreadpy has no load_ff_opportunity")
            return pl.DataFrame()
        try:
            raw = nfl.load_ff_opportunity(seasons=seasons_list, stat_type="weekly")
        except TypeError:
            raw = nfl.load_ff_opportunity(seasons=seasons_list)
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        if "player_id" in df.columns and "gsis_id" not in df.columns:
            df = df.with_columns(pl.col("player_id").cast(pl.Utf8).alias("gsis_id"))
        return coalesce_player_id(df) if "gsis_id" in df.columns or "player_id" in df.columns else df

    return load_or_fetch("ff_opportunity", fetch, seasons=seasons_list, force=force)


def load_contracts(*, force: bool = False) -> pl.DataFrame:
    """Historical + active NFL contracts from OverTheCap via nflverse (rotc).

    Redistributed by nflverse — do not scrape overthecap.com directly.
    """
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_contracts"):
            logger.warning("nflreadpy has no load_contracts")
            return pl.DataFrame()
        raw = nfl.load_contracts()
        df = _to_polars(raw)
        df = coerce_id_columns(df)
        if "player_id" in df.columns and "gsis_id" not in df.columns:
            df = df.with_columns(pl.col("player_id").cast(pl.Utf8).alias("gsis_id"))
        if "gsis_id" in df.columns or "player_id" in df.columns:
            df = coalesce_player_id(df)
        return df

    return load_or_fetch("contracts", fetch, force=force)


def load_participation(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Play-level personnel/participation (NGS through 2022, FTN thereafter).

    FTN-supplied seasons are CC-BY-SA 4.0 and require attribution to
    ``FTN Data via nflverse``. Earlier seasons should be attributed to
    ``NFL Next Gen Stats via nflverse``.
    """
    seasons_list = [s for s in _season_list(seasons) if s >= 2016]
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_participation"):
            logger.warning("nflreadpy has no load_participation")
            return pl.DataFrame()
        try:
            raw = nfl.load_participation(seasons=seasons_list, include_pbp=False)
        except TypeError:
            raw = nfl.load_participation(seasons=seasons_list)
        return _to_polars(raw)

    return load_or_fetch("participation", fetch, seasons=seasons_list, force=force)


def load_nextgen_stats(
    seasons: int | Sequence[int] | None = None,
    *,
    stat_type: str,
    force: bool = False,
) -> pl.DataFrame:
    """Weekly NFL Next Gen Stats for passing, receiving, or rushing."""
    if stat_type not in {"passing", "receiving", "rushing"}:
        raise ValueError("stat_type must be passing, receiving, or rushing")
    seasons_list = [s for s in _season_list(seasons) if s >= 2016]
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_nextgen_stats"):
            logger.warning("nflreadpy has no load_nextgen_stats")
            return pl.DataFrame()
        raw = nfl.load_nextgen_stats(seasons=seasons_list, stat_type=stat_type)
        df = coerce_id_columns(_to_polars(raw), ("player_gsis_id", "gsis_id"))
        if "player_gsis_id" in df.columns and "gsis_id" not in df.columns:
            df = df.rename({"player_gsis_id": "gsis_id"})
        return df

    return load_or_fetch(
        f"nextgen_{stat_type}", fetch, seasons=seasons_list, force=force
    )


def load_rosters_weekly(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Week-level roster/status history, available from 2002 onward."""
    seasons_list = _season_list(seasons)
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_rosters_weekly"):
            logger.warning("nflreadpy has no load_rosters_weekly")
            return pl.DataFrame()
        return coerce_id_columns(
            _to_polars(nfl.load_rosters_weekly(seasons=seasons_list))
        )

    return load_or_fetch("rosters_weekly", fetch, seasons=seasons_list, force=force)


def load_ftn_charting(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """FTN play charting (2022+), licensed CC-BY-SA 4.0.

    Downstream artifacts using these fields must attribute
    ``FTN Data via nflverse``.
    """
    seasons_list = [s for s in _season_list(seasons) if s >= 2022]
    if not seasons_list:
        return pl.DataFrame()
    nfl = _import_nflreadpy()

    def fetch() -> pl.DataFrame:
        if not hasattr(nfl, "load_ftn_charting"):
            logger.warning("nflreadpy has no load_ftn_charting")
            return pl.DataFrame()
        return _to_polars(nfl.load_ftn_charting(seasons=seasons_list))

    return load_or_fetch("ftn_charting", fetch, seasons=seasons_list, force=force)


def ingest_all(
    seasons: int | Sequence[int] | None = None,
    *,
    force: bool = False,
) -> dict[str, pl.DataFrame]:
    """Download and cache all core tables."""
    seasons_list = _season_list(seasons)
    ensure_dirs()
    def optional(name: str, loader: Callable[[], pl.DataFrame]) -> pl.DataFrame:
        try:
            return loader()
        except Exception as exc:
            logger.warning("Optional %s ingestion skipped: %s", name, exc)
            return pl.DataFrame()

    tables = {
        "player_stats": load_player_stats(seasons_list, force=force),
        "schedules": load_schedules(seasons_list, force=force),
        "snap_counts": load_snap_counts(seasons_list, force=force),
        "rosters": load_rosters(seasons_list, force=force),
        "players": load_players(force=force),
        "ff_playerids": load_ff_playerids(force=force),
        "draft_picks": load_draft_picks(seasons_list, force=force),
        "combine": load_combine(seasons_list, force=force),
        "team_stats": load_team_stats(seasons_list, force=force),
        "injuries_nflverse": load_injuries_nflverse(seasons_list, force=force),
        "depth_charts": load_depth_charts(seasons_list, force=force),
        "ff_opportunity": load_ff_opportunity(seasons_list, force=force),
        "contracts": load_contracts(force=force),
        "participation": optional(
            "participation", lambda: load_participation(seasons_list, force=force)
        ),
        "nextgen_passing": optional(
            "nextgen passing",
            lambda: load_nextgen_stats(seasons_list, stat_type="passing", force=force),
        ),
        "nextgen_receiving": optional(
            "nextgen receiving",
            lambda: load_nextgen_stats(seasons_list, stat_type="receiving", force=force),
        ),
        "nextgen_rushing": optional(
            "nextgen rushing",
            lambda: load_nextgen_stats(seasons_list, stat_type="rushing", force=force),
        ),
        "rosters_weekly": optional(
            "weekly rosters", lambda: load_rosters_weekly(seasons_list, force=force)
        ),
        "ftn_charting": optional(
            "FTN charting", lambda: load_ftn_charting(seasons_list, force=force)
        ),
    }
    return tables
