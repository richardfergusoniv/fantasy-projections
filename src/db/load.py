"""Load all Phase 1 sources into a local SQLite database.

Season window: 2016-2025. 2016 is the earliest season with play-level
participation data (verified in Phase 0 follow-up check; 2015 returns 404
from the nflverse-data participation release). Everything else (pbp, weekly,
snap counts, rosters, NGS) goes back further, but since the OL/coordinator
models both key off participation, there is no value in ingesting pbp-only
years before 2016 for this project.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingest import sources as src
from src.paths import DB_PATH

SEASONS = list(range(2016, 2027))

# (table_name, fetch_fn, index_columns)
TABLE_SPECS = [
    ("participation", lambda: src.get_participation(SEASONS), ["season", "nflverse_game_id"]),
    ("pbp", lambda: src.get_pbp(SEASONS), ["season", "week", "game_id"]),
    ("weekly", lambda: src.get_weekly_data(SEASONS), ["season", "week", "player_id"]),
    ("snap_counts", lambda: src.get_snap_counts(SEASONS), ["season", "week", "pfr_player_id"]),
    ("depth_charts", lambda: src.get_depth_charts(SEASONS), ["season", "week", "gsis_id"]),
    ("seasonal_rosters", lambda: src.get_seasonal_rosters(SEASONS), ["season", "player_id"]),
    ("weekly_rosters", lambda: src.get_weekly_rosters(SEASONS), ["season", "week", "player_id"]),
    ("schedules", lambda: src.get_schedules(SEASONS), ["season", "week", "game_id"]),
    ("ngs_passing", lambda: src.get_ngs(SEASONS, "passing"), ["season", "week", "player_gsis_id"]),
    ("ngs_rushing", lambda: src.get_ngs(SEASONS, "rushing"), ["season", "week", "player_gsis_id"]),
    ("ngs_receiving", lambda: src.get_ngs(SEASONS, "receiving"), ["season", "week", "player_gsis_id"]),
    ("ftn", lambda: src.get_ftn(SEASONS), ["season", "week", "nflverse_game_id"]),
    ("weekly_pfr_pass", lambda: src.get_weekly_pfr(SEASONS, "pass"), ["season", "week", "pfr_player_id"]),
    ("weekly_pfr_rec", lambda: src.get_weekly_pfr(SEASONS, "rec"), ["season", "week", "pfr_player_id"]),
    ("weekly_pfr_rush", lambda: src.get_weekly_pfr(SEASONS, "rush"), ["season", "week", "pfr_player_id"]),
    ("seasonal_pfr_pass", lambda: src.get_seasonal_pfr(SEASONS, "pass"), ["season", "pfr_id"]),
    ("seasonal_pfr_rec", lambda: src.get_seasonal_pfr(SEASONS, "rec"), ["season", "pfr_id"]),
    ("seasonal_pfr_rush", lambda: src.get_seasonal_pfr(SEASONS, "rush"), ["season", "pfr_id"]),
    ("draft_picks", lambda: src.get_draft_picks(SEASONS), ["season", "gsis_id"]),
    ("injuries", lambda: src.get_injuries(SEASONS), ["season", "week", "gsis_id"]),
    ("combine_data", lambda: src.get_combine_data(SEASONS), ["season", "pfr_id"]),
    ("ids", lambda: src.get_ids(), ["gsis_id", "pfr_id", "espn_id"]),
    ("players", lambda: src.get_players(), ["gsis_id"]),
]


def list_columns_to_str(df):
    """SQLite can't store list/array columns; nflverse ships some as numpy arrays
    (e.g. weekly_rosters game-by-game fields) - stringify anything non-scalar."""
    import numpy as np

    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], (list, np.ndarray)):
                df[col] = df[col].apply(lambda v: ";".join(map(str, v)) if isinstance(v, (list, np.ndarray)) else v)
    return df


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    report = []
    for table_name, fetch_fn, index_cols in TABLE_SPECS:
        print(f"Loading {table_name} ...")
        try:
            df, failed_seasons = fetch_fn()
            df = list_columns_to_str(df)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            for col in index_cols:
                if col in df.columns:
                    idx_name = f"idx_{table_name}_{col}"
                    conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col}")')
            conn.commit()
            print(f"  {table_name}: {len(df)} rows, {len(df.columns)} cols, failed_seasons={failed_seasons}")
            report.append((table_name, len(df), len(df.columns), failed_seasons, None))
        except Exception as e:
            print(f"  {table_name}: ERROR {repr(e)}")
            report.append((table_name, None, None, None, str(e)))

    conn.close()

    print("\n=== Load summary ===")
    for table_name, rows, cols, failed_seasons, err in report:
        if err:
            print(f"  {table_name}: FAILED - {err}")
        elif failed_seasons:
            print(f"  {table_name}: {rows} rows x {cols} cols -- GAPS: {failed_seasons}")
        else:
            print(f"  {table_name}: {rows} rows x {cols} cols -- full coverage")

    return report


if __name__ == "__main__":
    main()
