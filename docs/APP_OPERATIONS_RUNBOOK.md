# Fantasy Decision App — Operations Runbook

Commands are marked **[verified 2026-08-31]** when they were executed on a
developer machine during the readiness audit, and **[pending]** when they
require a provider account, credential, or runtime that was not available. See
[`PRODUCTION_READINESS_AUDIT.md`](PRODUCTION_READINESS_AUDIT.md) for the full
evidence.

## Local startup (fixture mode) — **[verified 2026-08-31]**

Runs with no PostgreSQL and no Docker: `.env.example` defaults `DATABASE_URL`
to a local SQLite file. Production refuses to start on SQLite, so this default
cannot leak into a deployment.

```powershell
uv sync --frozen --all-extras --dev
copy .env.example .env
# Set APP_SECRET_KEY to any 32+ character value for local use.
uv run python -m src.app.cli migrate
uv run python -m src.app.cli seed --email owner@example.com
uv run python -m src.app.cli api --port 8000
```

In a second terminal:

```powershell
cd web
npm ci
npm run dev
```

Open http://localhost:5173 and sign in with the email in `APP_ALLOWED_EMAIL`.
With `APP_ENABLE_DEV_AUTH=true`, the magic-link response includes
`development_link`, and the login screen renders it as a clickable link.

Seeding creates six leagues matching the product contract: two redraft and four
dynasty, two of them Superflex, one points-per-first-down, one with yardage
bonuses, five with a team defense, four with a kicker.

### First refresh

After signing in, `POST /api/v1/sync` (or **Run daily refresh** on the
Operations screen) imports all six leagues from the recorded Sleeper fixtures.
Check the job metadata: `unresolved_player_ids` must be empty. A non-empty list
means Sleeper ids could not be mapped onto known players, and those players will
have no projection.

## Docker Compose — **[pending: needs a Docker runtime]**

```powershell
copy .env.example .env
docker compose build
docker compose up -d
pwsh scripts/compose_smoke.ps1
docker compose ps
docker compose logs --no-color
```

Services: PostgreSQL (loopback `5432`), one-shot `migrate` and `seed`, `api`
(`8000`), the `scheduler` (the single authoritative timer), a one-shot `worker`
for manual runs, and static `web` (`5173`). The stack sets an explicit Compose
project name (`fantasy-decision-app`) and named volumes, so it cannot collide
with another project's containers or data.

Static validation of this configuration runs without Docker and is **not** a
substitute for the above:

```powershell
uv run python scripts/validate_compose_config.py    # [verified 2026-08-31]
```

## Scheduled jobs (America/Los_Angeles)

| Job | Schedule |
|---|---|
| `daily-refresh` | Daily 5:00 PM except Sunday |
| `sunday-early` | Sunday 8:45 AM |
| `sunday-afternoon` | Sunday 11:45 AM |
| `sunday-night` | Sunday 4:00 PM |
| `monday-night` | Monday 4:00 PM |
| `weekly-close-preliminary` | Tuesday 5:00 AM |
| `weekly-correction` | Wednesday 5:00 PM |
| `full-release` | On demand |

Schedules are wall-clock local and converted to UTC through `zoneinfo`, so the
two annual DST transitions are handled rather than assumed away. Weekly close
postpones itself when the NFL week is not final.

Run one job manually — **[verified 2026-08-31]**:

```powershell
uv run python -m src.app.jobs.scheduler run-once daily-refresh
```

Run every slot that is currently due (what the `scheduler` service loops on).
Each occurrence has a deterministic idempotency key, so invoking this more often
than the schedule cannot double-run a slot:

```powershell
uv run python -m src.app.jobs.scheduler run-due
uv run python -m src.app.jobs.scheduler list
```

Known deviation: `sunday-early`, `sunday-afternoon`, and `sunday-night` all run
`run_daily_refresh`. Each performs a full status sync, research on changed
players, and an affected-team publish; the blueprint's window-specific targeting
is not yet differentiated.

## Database backup — **[pending: needs a running Docker stack]**

```powershell
docker compose exec db pg_dump -U fantasy fantasy_app > backup.sql
```

Restore:

```powershell
docker compose exec -T db psql -U fantasy fantasy_app < backup.sql
```

There is no automated backup schedule. Restoring is a manual operator action,
and the procedure above has not been rehearsed against a real database.

For the local SQLite path, the database is a single file at the path in
`DATABASE_URL`; stop the API before copying it, because the connection runs in
WAL mode.

## Object retention

- Daily artifacts: 30 days
- Weekly season artifacts: through season end
- Named sealed releases: indefinite

Retention is a policy statement; no automated pruning job exists yet.

## Failure handling

Jobs take a PostgreSQL advisory lock (a no-op on SQLite, so mutual exclusion is
**not** exercised by the local path), write `job_run` records with a correlation
id, and fail closed. A failed job's database work is rolled back while its
failure row is committed on a separate short-lived session, so a crash still
leaves an audit trail. A failed promotion keeps the previous active projection
pointer.

## Health checks

- Liveness: `GET /health/live` — **[verified 2026-08-31]**
- Readiness: `GET /health/ready` (database connectivity) — **[verified 2026-08-31]**

## Operations visibility

`GET /api/v1/operations/status` and the Operations screen report, without
leaking secrets: the active release per horizon and per league, last successful
and failed job timestamps, data and evidence freshness, scheduler health and the
next due slot, artifact-store health, validation failures, the last rollback,
degraded dependencies, estimated month-to-date assistant cost, and — added in
this audit — the **Sleeper data source** (`fixture` or `live`), the weekly
artifact state (`fixture` / `fallback` / `trained`), and whether automatic
publishing is allowed.

If the Operations screen says *fixture — recorded payloads, not live league
data* or *not trained production output*, nothing on any other screen is a live
or trained result.

## MVP verification — **[verified 2026-08-31]**

```powershell
uv run pytest -q
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/vertical_smoke.py
uv run alembic upgrade head
uv run alembic check
cd web; npm ci; npm test; npm run build; npm run test:e2e
```

`npm run test:e2e` starts its own disposable seeded API through
`scripts/e2e_api.py` (a fresh SQLite file under `web/.e2e/`, migrated and
seeded per run) and drives a real owner journey against it. It never touches a
database an operator is using.

## Live Sleeper check (read-only, opt-in) — **[pending: needs the owner's username]**

```powershell
$env:LIVE_SLEEPER_SMOKE = "1"; $env:SLEEPER_USERNAME = "<username>"
uv run python scripts/live_sleeper_smoke.py
```

GET requests only; prints counts, never payloads, emails, or roster contents.
It skips with exit 0 when not configured, so it can sit in a pipeline without
becoming a network-dependent required test.

## Projection rollback

Restore the previous active weekly (or ROS/dynasty) pointer after a bad
promotion. Requires a CSRF token and an `Idempotency-Key` header:

```text
POST /api/v1/operations/projections/rollback?mode=weekly&season=2026&week=1
```

Rollback swaps the pointer only. Neither the superseded nor the restored bundle
is modified.

## Release checklist

1. `uv run pytest -q` and the web suite are green.
2. `uv run alembic check` reports no drift against the target database.
3. `operations/status` shows the expected `sleeper_source` and
   `weekly_v2_state` for the environment being deployed.
4. Production configuration validates: start the API with `APP_ENV=production`
   and confirm it boots rather than listing unsafe settings.
5. `docker compose build` and `pwsh scripts/compose_smoke.ps1` pass — **[pending]**.
6. A backup of the current database exists before migrating.
7. After deploy: sign in, run one refresh, confirm `unresolved_player_ids` is
   empty and every league still returns a lineup.
