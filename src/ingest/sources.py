"""Per-source ingestion functions. Each wraps a raw fetch in the on-disk cache.

Every get_* function returns (DataFrame, failed_seasons) where failed_seasons
is a list of (season, error_str) for seasons that could not be fetched.
Callers MUST surface failed_seasons rather than silently dropping them - some
gaps are expected (a source's known start year), others are real upstream
publishing gaps (e.g. player_stats not yet published for a season that pbp
already has), and both need to be visible rather than silently absorbed.
"""
import pandas as pd
import nfl_data_py as nfl

from src.cache import cached_multi_season, cached_season

PARTICIPATION_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp_participation/pbp_participation_{season}.parquet"
)

# Participation data exists from 2016 on (verified in Phase 0 follow-up: 2015 -> 404).
PARTICIPATION_MIN_SEASON = 2016
# FTN charting exists from 2022 on (verified in Phase 0).
FTN_MIN_SEASON = 2022
# PFR advanced stats (weekly + seasonal) exist from 2018 on (nfl_data_py raises otherwise).
PFR_MIN_SEASON = 2018


def get_participation(seasons, force=False):
    """Play-level personnel data. Not wrapped by nfl_data_py 0.3.3 -> direct nflverse-data pull."""
    below_min = [s for s in seasons if s < PARTICIPATION_MIN_SEASON]
    seasons = [s for s in seasons if s >= PARTICIPATION_MIN_SEASON]

    def fetch_one(season):
        df = pd.read_parquet(PARTICIPATION_URL.format(season=season))
        df["season"] = season
        return df

    df, failed = cached_multi_season("participation", seasons, fetch_one, force=force, skip_missing=True)
    failed = [(s, "before PARTICIPATION_MIN_SEASON=2016") for s in below_min] + failed
    return df, failed


def get_pbp(seasons, force=False):
    def fetch_one(season):
        return nfl.import_pbp_data([season], downcast=True, cache=False, include_participation=False)

    return cached_multi_season("pbp", seasons, fetch_one, force=force, skip_missing=True)


def get_weekly_data(seasons, force=False):
    """Official nflverse player_stats file, per season. If a season's file
    hasn't been published upstream yet (e.g. 2025 as of this writing), fall
    back to aggregating attempts/yards/TDs/receptions/targets directly from
    pbp for that season rather than dropping it - decided explicitly with
    the user rather than silently choosing either behavior. Rows from the
    fallback are tagged stat_source='pbp_fallback' (official rows get
    stat_source='player_stats') so downstream code can tell them apart.
    """
    from src.ingest.pbp_stats_fallback import aggregate_weekly_stats_from_pbp

    def fetch_one(season):
        df = nfl.import_weekly_data([season], downcast=True)
        df["stat_source"] = "player_stats"
        return df

    df, failed = cached_multi_season("weekly", seasons, fetch_one, force=force, skip_missing=True)

    if failed:
        fallback_seasons = [s for s, _ in failed]
        pbp_df, pbp_failed = get_pbp(fallback_seasons, force=force)
        fallback_df = aggregate_weekly_stats_from_pbp(pbp_df)
        df = pd.concat([df, fallback_df], ignore_index=True)
        # seasons that failed both the official file AND the pbp fallback are real gaps
        failed = pbp_failed

    return df, failed


def get_snap_counts(seasons, force=False):
    def fetch_one(season):
        return nfl.import_snap_counts([season])

    return cached_multi_season("snap_counts", seasons, fetch_one, force=force, skip_missing=True)


def get_depth_charts(seasons, force=False):
    def fetch_one(season):
        return nfl.import_depth_charts([season])

    return cached_multi_season("depth_charts", seasons, fetch_one, force=force, skip_missing=True)


def get_seasonal_rosters(seasons, force=False):
    def fetch_one(season):
        return nfl.import_seasonal_rosters([season])

    return cached_multi_season("seasonal_rosters", seasons, fetch_one, force=force, skip_missing=True)


def get_weekly_rosters(seasons, force=False):
    def fetch_one(season):
        return nfl.import_weekly_rosters([season])

    return cached_multi_season("weekly_rosters", seasons, fetch_one, force=force, skip_missing=True)


def get_schedules(seasons, force=False):
    def fetch_one(season):
        return nfl.import_schedules([season])

    return cached_multi_season("schedules", seasons, fetch_one, force=force, skip_missing=True)


def get_ngs(seasons, stat_type, force=False):
    """stat_type: passing | rushing | receiving"""
    assert stat_type in ("passing", "rushing", "receiving")

    def fetch_one(season):
        return nfl.import_ngs_data(stat_type, [season])

    return cached_multi_season(f"ngs_{stat_type}", seasons, fetch_one, force=force, skip_missing=True)


def get_ftn(seasons, force=False):
    below_min = [s for s in seasons if s < FTN_MIN_SEASON]
    seasons = [s for s in seasons if s >= FTN_MIN_SEASON]

    def fetch_one(season):
        return nfl.import_ftn_data([season])

    df, failed = cached_multi_season("ftn", seasons, fetch_one, force=force, skip_missing=True)
    failed = [(s, "before FTN_MIN_SEASON=2022") for s in below_min] + failed
    return df, failed


def get_weekly_pfr(seasons, s_type, force=False):
    """s_type: pass | rec | rush | def"""
    assert s_type in ("pass", "rec", "rush", "def")
    below_min = [s for s in seasons if s < PFR_MIN_SEASON]
    seasons = [s for s in seasons if s >= PFR_MIN_SEASON]

    def fetch_one(season):
        return nfl.import_weekly_pfr(s_type, [season])

    df, failed = cached_multi_season(f"weekly_pfr_{s_type}", seasons, fetch_one, force=force, skip_missing=True)
    failed = [(s, "before PFR_MIN_SEASON=2018") for s in below_min] + failed
    return df, failed


def get_seasonal_pfr(seasons, s_type, force=False):
    assert s_type in ("pass", "rec", "rush", "def")
    below_min = [s for s in seasons if s < PFR_MIN_SEASON]
    seasons = [s for s in seasons if s >= PFR_MIN_SEASON]

    def fetch_one(season):
        return nfl.import_seasonal_pfr(s_type, [season])

    df, failed = cached_multi_season(f"seasonal_pfr_{s_type}", seasons, fetch_one, force=force, skip_missing=True)
    failed = [(s, "before PFR_MIN_SEASON=2018") for s in below_min] + failed
    return df, failed


def get_ids(force=False):
    """Master player-id crosswalk. Not season-indexed -> cache under a single pseudo-season key."""
    return cached_season("ids", 0, lambda: nfl.import_ids(), force=force), []


def get_players(force=False):
    return cached_season("players", 0, lambda: nfl.import_players(), force=force), []
