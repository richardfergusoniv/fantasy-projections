# Phase 1 — Data Layer

## What's built

- `src/cache.py` — on-disk per-season parquet cache under `data/raw/{source}/{season}.parquet`.
  Re-running ingestion only fetches what isn't already cached (or use `force=True`
  to bypass). `cached_multi_season(..., skip_missing=True)` fetches season-by-season
  and returns `(DataFrame, failed_seasons)` instead of aborting on one bad season —
  every failure is reported, never silently dropped.
- `src/ingest/sources.py` — one `get_*` function per nflverse source (19 total).
  Everything goes through `nfl_data_py` except `participation`, which isn't wrapped
  by the installed `nfl_data_py` version (0.3.3) and is fetched directly from the
  nflverse-data GitHub release parquet files (see [PHASE0_REPORT.md](PHASE0_REPORT.md)).
- `src/db/load.py` — loads all 19 sources into `data/projections.db` (SQLite),
  season window 2016-2025, with indexes on the relevant join columns per table.
- `src/crosswalk.py` — player-ID join coverage checker across `players` (nflverse
  general master) vs `ids` (fantasy-platform-scoped master).

## Season window

2016-2025. 2016 was chosen as the floor because that's the earliest season with
play-level participation data (2015 → 404 from nflverse-data). PBP/weekly/rosters
go back further but there's no value pulling pbp-only years for this project since
the OL and coordinator-tendency models both key off participation.

## Data gaps (reported, not silently filled)

| Source | Gap | Cause |
|---|---|---|
| `weekly` (player_stats) | **2025 missing entirely** | Real upstream gap — nflverse has not yet published `player_stats_2025.parquet`, even though `pbp`, `participation`, and `ftn` all have full 2025 data (confirmed 22 weeks incl. playoffs). This needs your attention: weekly attempts/yards/TDs/receptions/targets are the actual target variables for the projection models. **Workaround available but not yet built:** these stats can be aggregated directly from `pbp` for 2025 instead of relying on the pre-aggregated `player_stats` release. Flagging for your decision before Phase 4 rather than building the workaround silently. |
| `ftn` | 2016-2021 missing | Expected — FTN charting starts 2022 (confirmed in Phase 0). |
| `weekly_pfr_*` / `seasonal_pfr_*` | 2016-2017 missing | Expected — PFR advanced stats start 2018 (`nfl_data_py` raises below that). |

All other sources (participation, pbp, snap_counts, depth_charts, seasonal_rosters,
weekly_rosters, schedules, ngs_passing/rushing/receiving, ids, players) have full
2016-2025 coverage with zero fetch failures.

## Player ID crosswalk finding (load-bearing for Phase 2)

`import_ids()` (table `ids`) is **not** a general player crosswalk — it's scoped to
fantasy-relevant players (built to match ESPN/Yahoo/Sleeper/PFF). Joining
`seasonal_rosters` to it on `gsis_id` loses **2,862 distinct O-line players** (and
similar counts for DL/LB/DB) because those positions aren't fantasy-relevant. Since
Phase 2 is specifically about O-line players, `ids` would have silently produced an
OL model missing most linemen if used as the crosswalk.

`import_players()` (table `players`) is nflverse's general player master and covers
everyone. Re-checking joins against `players` instead:

| Table | join column | match rate vs `ids` | match rate vs `players` |
|---|---|---|---|
| seasonal_rosters | player_id → gsis_id | 59.4% | **84.4%** |
| depth_charts | gsis_id | 77.9% | **100.0%** |
| snap_counts | pfr_player_id → pfr_id | 80.5% | **99.6%** |
| weekly_pfr_rec | pfr_player_id → pfr_id | 95.2% | **99.6%** |
| weekly_pfr_rush | pfr_player_id → pfr_id | 99.1% | **99.7%** |
| weekly | player_id → gsis_id | 96.7% | **100.0%** |
| ngs (all 3) | player_gsis_id | 100.0% | 100.0% |
| weekly_pfr_pass | pfr_player_id → pfr_id | 100.0% | 100.0% |

**Decision made:** `players` is now the primary crosswalk hub for gsis_id ↔ pfr_id
↔ espn_id joins used anywhere in the OL model or general player identity work.
`ids` is kept only for fantasy-platform-specific IDs (sleeper_id, yahoo_id) needed
at the Phase 5 output stage, where its fantasy-only scope is actually correct.

`seasonal_rosters`'s 84.4% match rate (vs `players`) is the one remaining
imperfect join — not yet root-caused. It doesn't block Phase 2 (which joins through
`depth_charts`/`participation`, both ~100%), so it's noted here rather than blocking
the gate, but should be root-caused before Phase 4 depends on seasonal_rosters
directly for share calculations.

## Open item for your decision

Do you want the Phase 4 target-variable pull to fall back to aggregating weekly
stats from `pbp` for any season where `player_stats` isn't published yet (i.e.
2025 today, and potentially future in-season gaps if this tool is ever re-run
mid-season)? This wasn't built without checking with you first, per your validate-before-building rule.
