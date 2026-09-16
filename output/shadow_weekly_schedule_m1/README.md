# Shadow weekly schedule allocation (Milestone 1)

Research/shadow only. This directory is **not** a sealed League Value board, not a Vegas replacement, and not a PWA input.

- Schema: `weekly_latent_m1_v1`
- Conservation passes: `True`
- Failing checks: none
- Dry run: `False`

Large `player_weeks.csv` / `team_weeks.csv` / `shares.csv` are gitignored. `summary.json`, `conservation.json`, and `sample_player_weeks.csv` are the committed evidence files.

Local run (Windows DB):

```bat
set FANTASY_PROJECTIONS_DATA_DIR=D:\fantasy-projections-data
set FANTASY_PROJECTIONS_DB_PATH=D:\fantasy-projections-data\projections.db
uv run python scripts/run_weekly_schedule_m1.py --season 2026
```
