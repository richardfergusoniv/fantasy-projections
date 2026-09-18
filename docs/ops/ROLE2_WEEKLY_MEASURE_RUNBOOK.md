# Role 2 weekly measure runbook (shadow only)

Score the M3 weekly shadow board against **timestamped** Vegas weekly props.
This is Role 2 (evaluation comparator). It does **not** blend (Role 3),
promote, flip `APP_PROJECTION_SOURCE`, or change the PWA.

Locked roles:
[`docs/decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md`](../decisions/WEEKLY_MODEL_PROMOTION_AND_PROPS_ROLES_2026-09-15.md).

Live-week count toward 6–8:
[`docs/research/WEEKLY_SHADOW_LIVE_WEEK_TRACKER_2026-09-16.md`](../research/WEEKLY_SHADOW_LIVE_WEEK_TRACKER_2026-09-16.md).

## 1. Harness (no live books)

These prove matching and fail-closed leakage. They do **not** credit a live week.

**Do not write harness output into `output/shadow_weekly_schedule_m3/`.** That
directory holds committed evidence, including the 2026 live-data blocker in
`vegas_props_compare.json`. A dry-run against that path rewrites those
artifacts (flipping the blocker to a synthetic `n_matched>0` pass). If you do
it by mistake, discard the diff.

```bash
# Weekly-eval synthetic four-row fixture (n_matched=4, with actuals).
# Writes output/shadow_vegas_props_compare/summary.json (the --dry-run fixture).
uv run python scripts/compare_shadow_vegas_props.py --dry-run

# Real M3 dry-run board (p-qb / p-rb / q-qb) vs timestamped M3 fixture
# as_of 2026-09-09T18:00Z <= kickoff 2026-09-10T20:20Z.
# Writes m3_dry_run_summary.json under --output; does not clobber summary.json.
# Scratch board lands under --output/_m3_dry_run_board.
uv run python scripts/compare_shadow_vegas_props.py --m3-dry-run \
  --output output/shadow_vegas_props_compare

# Same path from the M3 CLI — write *outside* the tracked evidence dir.
uv run python scripts/run_weekly_schedule_m3.py --dry-run --skip-backtest \
  --output /tmp/shadow_weekly_schedule_m3_harness
```

Expect `n_matched > 0`, `role=evaluation_comparator`, `gate_verdict=not_promoting`.
Post-kickoff rows, missing `as_of` / `kickoff_at`, same-week outcome columns on
the board, and non-empty `blend_weights` fail closed.

## 2. Produce the M3 Role 2 board (2026 slate)

```bash
uv run python scripts/run_weekly_schedule_m3.py --season 2026
```

Writes `output/shadow_weekly_schedule_m3/shadow_board_role2.csv` (gitignored;
long `player_id,season,week,market,model_mean`). Player ids are sealed-board
gsis ids (`00-0034857`), not dry-run names.

Without live snapshots this compare is **blocked** (`harness=missing_live_snapshots`).
It will **not** silently join the dry-run fixture (`p-qb`) to the sealed board.

## 3. What Richard must supply for a live week

No timestamped book dump is committed. Role 1 `weekly_props` snapshots live in
the DB / artifact store and are not a Role 2 as-of CSV.

Supply a CSV (or set `WEEKLY_EVAL_PROPS_PATH`) at:

`output/shadow_vegas_props_compare/live_snapshots.csv`

| Column | Required | Notes |
|---|---|---|
| `player_id` | yes | **Same id as the M3 board** (gsis). Name-only rows will not match. |
| `season`, `week` | yes | Join key |
| `market` | yes | Role 2 names: `pass_yards`, `rec_yards`, `receptions`, `rush_yards`, … |
| `as_of` | yes | ISO-8601 UTC snapshot time |
| `kickoff_at` | yes | Game kickoff. `as_of > kickoff_at` fails closed |
| `line` or `implied_mean` | one of | Over/under and/or market location |
| `implied_p_over` | no | De-vig P(over) when available |

Optional after the week: outcomes CSV with `player_id,season,week,market,actual`.

Then:

```bash
uv run python scripts/compare_shadow_vegas_props.py \
  --board output/shadow_weekly_schedule_m3/shadow_board_role2.csv \
  --props output/shadow_vegas_props_compare/live_snapshots.csv \
  --outcomes path/to/outcomes.csv
```

A week credits toward 6–8 only when that live run has `n_matched > 0` and
`as_of <= kickoff_at`. See the tracker note for the exact rule.

## 4. Live blockers (current)

- No in-repo timestamped 2026 book snapshots (`as_of` + `kickoff_at`).
- Role 1 provider dumps are not exported as a Role 2 frame.
- Sealed-board player ids (`00-…`) do not match dry-run / book name keys
  unless identity is resolved **before** the CSV is handed to Role 2.
- Kickoff timestamps must come from the schedule, not a later closing line.
- Outcomes are absent until the week is final.

## 5. Still forbidden

Role 3 blend, `APP_PROJECTION_SOURCE` / PWA / sealed-pointer changes, treating
market agreement as a training target, and counting fixture weeks toward 6–8.
