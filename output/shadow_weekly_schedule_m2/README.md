# Shadow weekly schedule allocation (Milestone 2)

Research/shadow only. This directory is **not** a sealed League Value board, not a Vegas replacement, and not a PWA input. Gate verdict: not promoting.

- Schema: `weekly_latent_m2_v1`
- Conservation / identity passes: `True`
- Failing checks: none
- Dry run: `False`
- Still shadow: `True`

Large `player_weeks.csv` / `team_weeks.csv` / `shares.csv` are gitignored. `summary.json`, `conservation.json`, and `sample_player_weeks.csv` are the committed evidence files. Milestone 2 also commits `backtest_synthetic.json` and `backtest_historical.json`.

Local run (Windows DB):

```bat
set FANTASY_PROJECTIONS_DATA_DIR=D:\fantasy-projections-data
set FANTASY_PROJECTIONS_DB_PATH=D:\fantasy-projections-data\projections.db
uv run python scripts/run_weekly_schedule_m2.py --season 2026
```
