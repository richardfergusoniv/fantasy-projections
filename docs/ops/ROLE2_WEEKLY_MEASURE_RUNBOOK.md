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

## 3. Export Role 1 → Role 2 live snapshots

No timestamped book dump is committed. Role 1 `weekly_props` snapshots live in
Postgres `source_snapshot` with bodies at `s3://fantasy-app/artifacts/...`.
Export them into the Role 2 CSV (gsis ids, schedule `kickoff_at`, snapshot
`as_of`) with:

```bash
# Requires DATABASE_URL + ARTIFACT_BACKEND=s3 + S3_* (see .env.production.example).
# Default --mode single keeps one row per book (Role 2 persistence; no MAE book-shop).
uv run python scripts/export_role2_live_props.py --season 2026 --week 2

# Role-1-shaped grain (robust_median across books) when you want one row per market:
uv run python scripts/export_role2_live_props.py --season 2026 --week 2 --mode consensus

# Offline / CI (fixture providers; does not credit a live week):
uv run python scripts/export_role2_live_props.py --from-fixtures --season 2026 --week 1
```

Writes (or set `WEEKLY_EVAL_PROPS_PATH`) to:

`output/shadow_vegas_props_compare/live_snapshots.csv`

| Column | Required | Notes |
|---|---|---|
| `player_id` | yes | **Same id as the M3 board** (gsis). Name-only rows will not match. |
| `season`, `week` | yes | Join key |
| `market` | yes | Role 2 names: `pass_yards`, `rec_yards`, `receptions`, `rush_yards`, … |
| `as_of` | yes | ISO-8601 UTC snapshot time |
| `kickoff_at` | yes | Schedule kickoff only (Eastern wall-clock → UTC). `as_of > kickoff_at` fails closed |
| `line` or `implied_mean` | one of | Over/under and/or market location |
| `implied_p_over` | no | De-vig P(over) when available |

`kickoff_at` is **schedule-only**. The exporter resolves kickoff from the NFL
schedule CSV by team (or opponent). Missing schedule team/opponent → row
dropped (fail closed). Book `event_start` is never used as the leakage
boundary.

### Role 1 artifact durability

Production scrapes catalog `weekly_props:{draftkings|fanduel}:{season}:{week}`
rows with S3 `artifact_uri`s. The write path
(`src.app.artifacts.store` via `persist_provider_snapshots`) **verifies after
upload** before marking a snapshot healthy/complete. A missing blob fails
closed **per provider** (no healthy catalog row for that book) without aborting
other providers in the same scrape. On S3, verify fails closed only on genuine
absence (404 / NoSuchKey); put-only IAM / transient HEAD errors match `_exists`
and do not fail the write. That keeps Role 2 `export_role2_live_props` from
trusting metadata whose body is gone, while preserving provider isolation.

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

- Live export needs S3 artifact credentials (`ARTIFACT_BACKEND=s3`,
  `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION`,
  `S3_BUCKET`) plus `DATABASE_URL` / `JOB_DATABASE_URL`. Without them,
  `scripts/export_role2_live_props.py` exits with a blocker (no fake rows).
- If Role 1 verify-after-upload finds a missing blob for one provider, that
  provider is not catalogued as healthy; surviving providers in the same scrape
  still persist. Re-run the scrape after fixing S3 durability rather than
  treating a prior healthy row as proof the blob still exists.
- Sealed-board player ids (`00-…`) do not match dry-run / book name keys
  unless identity is resolved **before** the CSV is handed to Role 2 (the
  exporter maps via `player_identity` / gsis-shaped quote ids).
- Kickoff timestamps come from the schedule CSV only (localized Eastern →
  UTC). Quotes whose team/opponent miss the schedule map are dropped — book
  `event_start` is not a fallback. Never use a later closing line as kickoff.
- Outcomes are absent until the week is final.
- **DK weekly TD hole (Phase 0b):** live DraftKings often omits `rush_tds` /
  `rec_tds` while FanDuel has them, so those Role 1 lines can be FD-only
  under `min_distinct_books_per_market=1`. Do not raise min books globally
  until a third source. See
  [`WEEKLY_PROPS_DK_TD_COVERAGE_PHASE0B.md`](WEEKLY_PROPS_DK_TD_COVERAGE_PHASE0B.md).

## 5. Still forbidden

Role 3 blend, `APP_PROJECTION_SOURCE` / PWA / sealed-pointer changes, treating
market agreement as a training target, and counting fixture weeks toward 6–8.
