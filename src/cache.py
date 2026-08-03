"""Generic on-disk cache for per-season nflverse pulls.

Every source is cached as data/raw/{source}/{season}.parquet so re-running
ingestion doesn't re-download. Pass force=True to bypass the cache for a
specific pull (e.g. refreshing the current in-progress season).
"""
import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def cached_season(source: str, season: int, fetch_fn, force: bool = False) -> pd.DataFrame:
    """fetch_fn: callable taking no args, returning a DataFrame for this season."""
    season_dir = os.path.join(RAW_DIR, source)
    os.makedirs(season_dir, exist_ok=True)
    path = os.path.join(season_dir, f"{season}.parquet")

    if os.path.exists(path) and not force:
        return pd.read_parquet(path)

    df = fetch_fn()
    df.to_parquet(path, index=False)
    return df


class SeasonFetchError(Exception):
    def __init__(self, season, cause):
        self.season = season
        self.cause = cause
        super().__init__(f"season {season}: {cause}")


def cached_multi_season(source: str, seasons, fetch_one_fn, force: bool = False, skip_missing: bool = False):
    """fetch_one_fn: callable(season) -> DataFrame for that single season.

    If skip_missing is True, a season whose fetch raises is dropped rather
    than aborting the whole pull, and the list of failed seasons is returned
    alongside the concatenated DataFrame as (df, failed_seasons) - callers
    MUST surface failed_seasons rather than silently ignoring it, per project
    rule: never silently drop a gap.
    """
    frames = []
    failed = []
    for season in seasons:
        try:
            frames.append(cached_season(source, season, lambda s=season: fetch_one_fn(s), force=force))
        except Exception as e:
            if not skip_missing:
                raise SeasonFetchError(season, e) from e
            failed.append((season, str(e)))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if skip_missing:
        return df, failed
    return df
