"""Player ID crosswalk helpers.

Two nflverse tables map player identity across sources, and they are NOT
interchangeable:

- `ids` (from import_ids()) is scoped to fantasy-relevant players - it's
  built to match fantasy platforms (ESPN/Yahoo/Sleeper/PFF/etc). It only
  covers ~59% of seasonal_rosters and drops most O-line/D-line/LB/DB
  players entirely (verified: joining seasonal_rosters to `ids` loses 2,862
  distinct OL players). Since the OL attribution model's whole point is
  O-line players, `ids` is the wrong table for it.
- `players` (from import_players()) is nflverse's general player master and
  covers essentially everyone: joining depth_charts/snap_counts/
  seasonal_rosters to `players` on gsis_id/pfr_id matches 99.6%-99.98% of
  rows, including linemen.

Use `players` as the primary crosswalk hub (gsis_id <-> pfr_id <-> espn_id
etc.) for anything touching the OL model or general player identity. Only
fall back to `ids` for fantasy-platform-specific ids (sleeper_id, yahoo_id)
needed at the Phase 5 output stage, and expect it to only cover
skill-position/fantasy-relevant players.
"""
import sqlite3

import pandas as pd

from src.db.load import DB_PATH


def get_conn():
    return sqlite3.connect(DB_PATH)


def crosswalk_coverage_report():
    """For each table with a player id column, report what fraction of
    distinct ids successfully join to each crosswalk table (`players` = the
    general nflverse master, `ids` = the fantasy-platform-scoped table)."""
    conn = get_conn()
    players = pd.read_sql("SELECT * FROM players", conn)
    ids = pd.read_sql("SELECT * FROM ids", conn)

    checks = [
        ("weekly", "player_id", "gsis_id"),
        ("snap_counts", "pfr_player_id", "pfr_id"),
        ("weekly_pfr_pass", "pfr_player_id", "pfr_id"),
        ("weekly_pfr_rec", "pfr_player_id", "pfr_id"),
        ("weekly_pfr_rush", "pfr_player_id", "pfr_id"),
        ("ngs_passing", "player_gsis_id", "gsis_id"),
        ("ngs_rushing", "player_gsis_id", "gsis_id"),
        ("ngs_receiving", "player_gsis_id", "gsis_id"),
        ("seasonal_rosters", "player_id", "gsis_id"),
        ("depth_charts", "gsis_id", "gsis_id"),
    ]

    report = []
    for table, table_id_col, crosswalk_id_col in checks:
        try:
            distinct_ids = pd.read_sql(f'SELECT DISTINCT "{table_id_col}" FROM "{table}"', conn)
            distinct_ids = distinct_ids[table_id_col].dropna()
            distinct_ids = distinct_ids[distinct_ids != ""]
            total = len(distinct_ids)

            row = {"table": table, "id_column": table_id_col, "distinct_ids": total}
            for crosswalk_name, crosswalk_df in [("players", players), ("ids", ids)]:
                crosswalk_set = set(crosswalk_df[crosswalk_id_col].dropna())
                matched = distinct_ids.isin(crosswalk_set).sum()
                row[f"match_rate_vs_{crosswalk_name}"] = round(matched / total, 4) if total else None
            report.append(row)
        except Exception as e:
            report.append({"table": table, "id_column": table_id_col, "error": str(e)})

    conn.close()
    return pd.DataFrame(report)


if __name__ == "__main__":
    df = crosswalk_coverage_report()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
