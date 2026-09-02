# Live Sleeper shadow beta report

**Date:** 2026-08-31
**Environment:** Windows 11, Python 3.14.4, uv, SQLite shadow database
**Isolation:** `output/live_shadow/shadow_app.db`, `output/live_shadow/artifacts/`
**Owner config:** `config/sleeper_owner.json` (gitignored; schema in `config/sleeper_owner.example.json`)

## Commands executed

```powershell
uv run pytest -q
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/validate_compose_config.py
uv run python scripts/vertical_smoke.py
uv run python scripts/live_sleeper_smoke.py   # with LIVE_SLEEPER_SMOKE=1 and a configured SLEEPER_USERNAME
uv run python -m src.app.cli sleeper-shadow-sync `
  --season 2026 `
  --config config/sleeper_owner.json `
  --report output/live_shadow/sleeper_sync_report.json `
  --inject-failure
```

**Blocked on this machine (not claimed as verified):**

- `alembic upgrade head` / `alembic check` without an explicit `DATABASE_URL` (default env pointed at unavailable PostgreSQL driver path during batch run; migrations verified via app test suite and shadow sync migrate path)
- `npm test`, `npm run build`, `npm run test:e2e` (`npm` not on `PATH` in this shell)
- Docker compose runtime (`docker` not installed)
- PostgreSQL integration

## GET-only safety and connectivity

- Sleeper client asserted `GET`-only before any network access.
- Opt-in env `LIVE_SLEEPER_SHADOW=1` required for full shadow sync.
- Live smoke: user resolved, **six** 2026 leagues discovered, roster payload complete, no writes attempted.
- Shadow sync resolved Sleeper user ID `739931264659927040` and imported **only** the six configured league IDs.

## Six-league discovery

| Display name | League ID | Type | Rookie-pick rule |
|---|---|---|---|
| Redraft league 1 | `<redraft-league-1>` | redraft | — |
| Redraft league 2 | `<redraft-league-2>` | redraft | — |
| Dynasty league 1 | `<dynasty-league-1>` | dynasty | reverse_standings |
| Dynasty league 2 | `<dynasty-league-2>` | dynasty | max_pf |
| Dynasty league 3 | `<dynasty-league-3>` | dynasty | max_pf |
| Dynasty league 4 | `<dynasty-league-4>` | dynasty | reverse_standings |

- Configured: 6, discovered: 6, extra leagues ignored: 0.
- Owner roster identified in every league (no fixture roster-1 fallback).

## Dynasty rookie-pick rules persisted

| League ID | Rule | Notes |
|---|---|---|
| `<dynasty-league-1>` | reverse_standings | owner-confirmed |
| `<dynasty-league-2>` | max_pf | owner-confirmed |
| `<dynasty-league-3>` | max_pf | owner-confirmed |
| `<dynasty-league-4>` | reverse_standings | owner-confirmed |

Redraft leagues have **no** `league_draft_rule` rows. Re-sync is idempotent.

## Scoring / roster contracts

All six leagues compiled after adding live-discovered Sleeper keys:

- `fgm_50_59`, `fgm_60p` (kicker distance buckets)
- `def_pass_def`, `tkl_loss` (defensive bonus stats)

Each league now reports `publishable: true` with a distinct `contract_hash`. Superflex and yardage-bonus rule shapes are present on the live leagues that use them.

## Identity reconciliation (live)

Aggregate across in-season rosters (week 1):

| Metric | Count |
|---|---|
| Distinct rostered Sleeper player IDs | 1,320 |
| Resolved canonical IDs | 1,320 |
| Unresolved IDs | 0 |
| Ambiguous IDs | 0 |
| Unresolved starters (material) | 0 |

Pre-draft redraft leagues have empty rosters; starter slots are empty/null and are excluded from the starter gate.

Unresolved-ID artifact: `output/live_shadow/unresolved_player_ids.json` (empty after sync).

**Defect fixed during beta:** live `players/nfl` upsert failed on null `position` and duplicate canonical IDs; ingest now defaults position to `UNK`, strips GSIS whitespace, and upserts idempotently.

## Historical data completeness

From shadow database after sync:

- Source snapshots: 71
- Roster snapshots: 72
- Completed trades: 31
- Traded picks: 93
- NFL week: 1 (`season_week_source: nfl_state`)
- Prior-season chain walks logged `sleeper_league_chain_fetch_failed` for some histories (bounded, non-fatal)

## Shadow recommendation smoke (per league)

| League | Status | Result |
|---|---|---|
| Redraft league 1 | pre_draft | skipped — empty roster (expected) |
| Redraft league 2 | pre_draft | skipped — empty roster (expected) |
| Dynasty league 1 | in_season | lineup, waivers, trade, dynasty OK |
| Dynasty league 2 | in_season | lineup, waivers, trade, dynasty OK |
| Dynasty league 3 | in_season | lineup, waivers, trade, dynasty OK |
| Dynasty league 4 | in_season | lineup, waivers, trade, dynasty OK |

Redraft future-pick rejection verified on pre-draft leagues. Recommendations use fixture/fallback projections — **not** production football advice.

## Idempotency and failure injection

- Second shadow sync: table counts unchanged (`stable: true`).
- Injected publication failure: active weekly pointer unchanged (`pointer_unchanged: true`, `promoted: false`).

## Projection artifact mode

- `weekly_v2_state: fixture`
- `auto_publish_allowed: false`
- Missing trained model artifacts (9 weight files)
- Shadow sync loaded sealed preseason bundle for decision APIs only; no promotion to production pointers

Automatic publishing remains blocked by design until trained weekly v2 artifacts pass the existing readiness gate.

## Defects found and fixed

1. **S1** — Live identity upsert crashed on null positions and duplicate keys during bulk flush.
2. **S2** — Scoring compiler missing live Sleeper keys `fgm_50_59`, `fgm_60p`, `def_pass_def`, `tkl_loss`.
3. **S3** — Shadow report referenced nonexistent `ScoringContract.scoring_type`.
4. **S3** — `DraftBoardService` instantiated with session argument.
5. **S3** — Empty pre-draft starter slots incorrectly failed identity starter gate.

## Go / no-go

**Continued live read-only shadow use: GO (conditional).**

Evidence: six configured leagues sync completely, identity resolution is clean on in-season rosters, four dynasty rules match owner confirmation, scoring contracts compile, in-season decision APIs exercise successfully, idempotency and failure injection pass, and production publication remains blocked.

**Not GO for:** automatic projection publishing, PostgreSQL/Docker deployment, email, OpenAI assistant, or public internet deployment.

## Remaining blockers

1. Trained weekly v2 model weights and leakage-safe promotion
2. PostgreSQL migration/runtime verification (advisory locks, concurrency)
3. Docker compose build/smoke on a host with Docker installed
4. npm/web CI on a host with Node on `PATH`
5. Real email delivery and OpenAI spend limits for production assistant/research
6. TLS, DNS, and hosting for public deployment
7. Pre-draft redraft leagues will not produce lineup/waiver advice until rosters populate

Machine-readable evidence: `output/live_shadow/sleeper_sync_report.json`
