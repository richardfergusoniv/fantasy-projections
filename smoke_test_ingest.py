import sys
sys.path.insert(0, ".")
from src.ingest import sources as src

SEASON = 2024

checks = [
    ("participation", lambda: src.get_participation([SEASON])),
    ("pbp", lambda: src.get_pbp([SEASON])),
    ("weekly", lambda: src.get_weekly_data([SEASON])),
    ("snap_counts", lambda: src.get_snap_counts([SEASON])),
    ("depth_charts", lambda: src.get_depth_charts([SEASON])),
    ("seasonal_rosters", lambda: src.get_seasonal_rosters([SEASON])),
    ("weekly_rosters", lambda: src.get_weekly_rosters([SEASON])),
    ("schedules", lambda: src.get_schedules([SEASON])),
    ("ngs_passing", lambda: src.get_ngs([SEASON], "passing")),
    ("ngs_rushing", lambda: src.get_ngs([SEASON], "rushing")),
    ("ngs_receiving", lambda: src.get_ngs([SEASON], "receiving")),
    ("ftn", lambda: src.get_ftn([SEASON])),
    ("weekly_pfr_pass", lambda: src.get_weekly_pfr([SEASON], "pass")),
    ("weekly_pfr_rec", lambda: src.get_weekly_pfr([SEASON], "rec")),
    ("weekly_pfr_rush", lambda: src.get_weekly_pfr([SEASON], "rush")),
    ("seasonal_pfr_pass", lambda: src.get_seasonal_pfr([SEASON], "pass")),
    ("ids", lambda: src.get_ids()),
    ("players", lambda: src.get_players()),
]

for name, fn in checks:
    try:
        df = fn()
        print(f"{name}: OK rows={len(df)} cols={len(df.columns)}")
    except Exception as e:
        print(f"{name}: ERROR {repr(e)}")
