# Depth-chart injury refresh

## Purpose

Keep `starters_YYYY.csv` as the researched base chart. Detect live injuries
(Sleeper now; nflverse injuries/weekly_rosters when available in-season),
propose status overrides and membership changes, and optionally write a
derived `live_depth_YYYY.csv` plus dated rows in `status_overrides_YYYY.csv`.

## Commands

```bash
# Dry run: ingest Sleeper status + write proposals only
python -m src.depth_chart.refresh --season 2026

# Apply auto-safe events (IR → zero + promote; PUP → games cap)
python -m src.depth_chart.refresh --season 2026 --apply

# After apply, rebuild projections and Fantasy Tools JSON
python -m src.projection.predict --season 2026
python -m src.projection.fantasy_points --season 2026
python -m src.draft_assistant.prepare --season 2026
python -m src.team_stats.prepare --season 2026
```

Optional flags:

- `--force-ingest` — re-fetch Sleeper players even if today’s parquet exists
- `--as-of YYYY-MM-DD` — stamp / filter dated overrides; predict also accepts `--as-of`
- `--proposals path.csv` — read/write proposal file (set `confirmed=true` on rows to apply non-auto-safe events)

## Policy

| Signal | Override | Chart |
|--------|----------|-------|
| IR / Injured Reserve | `mode=zero` | Remove + promote next (auto-safe) |
| PUP | `mode=cap` (8 games) | Stay on chart; no promote (auto-safe) |
| Sus / Suspension | `mode=zero` (until a human sets a games cap) | Stay on chart (auto-safe) |
| Out / Doubtful | none | Proposal flag only |

Draft boards assume **17 games** for everyone else. Soft Gate A availability
stays in `projected_games_raw` for audit only. Do not use status overrides to
ack curated↔nflverse membership disagreements — those are chart edits.

## Files

- `src/depth_chart/starters_2026.csv` — curated base (never auto-edited)
- `src/depth_chart/live_depth_2026.csv` — derived chart after `--apply`
- `src/depth_chart/status_overrides_2026.csv` — dated games overrides
- `output/depth_refresh_proposals_2026.csv` — review queue
- `data/sleeper/sleeper_player_status.parquet` — ingested Sleeper status
