# Shadow weekly schedule allocation (Milestone 3)

Research/shadow only. This directory is **not** a sealed League Value board, not a Vegas replacement, and not a PWA input. Gate verdict: not promoting.

- Schema: `weekly_latent_m3_v1`
- Conservation / identity passes: `True`
- Failing checks: none
- Dry run: `False`
- Still shadow: `True`

Large `player_weeks.csv` / `team_weeks.csv` / `shares.csv` / `shadow_board_role2.csv` are gitignored. `summary.json`, `conservation.json`, and `sample_player_weeks.csv` are the committed evidence files. Milestone 3 also commits `backtest_synthetic.json`, `backtest_historical.json`, `vegas_props_compare.json` (2026 live-data blocker), `vegas_props_compare_m3_dry_run.json` (n_matched>0 harness), `market_sanity.json`, and `live_shadow_weeks.json` (starts at 0).

Role 2 measure: `uv run python scripts/compare_shadow_vegas_props.py --m3-dry-run` and [`docs/ops/ROLE2_WEEKLY_MEASURE_RUNBOOK.md`](../../docs/ops/ROLE2_WEEKLY_MEASURE_RUNBOOK.md).

Local run (Windows DB):

```bat
set FANTASY_PROJECTIONS_DATA_DIR=D:\fantasy-projections-data
set FANTASY_PROJECTIONS_DB_PATH=D:\fantasy-projections-data\projections.db
uv run python scripts/run_weekly_schedule_m3.py --season 2026
```
