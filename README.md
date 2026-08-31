# Fantasy Projections

NFL fantasy projections, Monte Carlo simulation, sealed release promotion, and a private mobile-first Sleeper decision application.

Architecture: [`docs/APP_IMPLEMENTATION_BLUEPRINT.md`](docs/APP_IMPLEMENTATION_BLUEPRINT.md)

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for locked dependency installs
- Node.js 22+ for the PWA (`web/`)

## Environment

| Variable | Purpose |
|---|---|
| `FANTASY_PROJECTIONS_DATA_DIR` | Root for `projections.db` / raw cache (see `src/paths.py`) |
| `FANTASY_PROJECTIONS_V2` | Path to the sibling v2 repo (default `../fantasy-projections-2`) |
| `APP_ALLOWED_EMAIL` | Authorized login email for the decision app |
| `APP_ENABLE_DEV_AUTH` | Print magic-link URLs in API responses (development only) |
| `DATABASE_URL` | Database URL for the decision app (SQLite locally, PostgreSQL in production) |
| `TRUSTED_HOSTS` | Hostnames the API answers to; production refuses `*` |
| `SLEEPER_USE_FIXTURES` | Leave unset to follow `APP_ENV`; production refuses fixtures |

Copy [`.env.example`](.env.example) and adjust values before running the app.

## Setup

```bash
uv sync --frozen --all-extras --dev
```

## Decision app quick start (fixture mode)

No PostgreSQL and no Docker required: `.env.example` defaults `DATABASE_URL` to
a local SQLite file. (Production refuses to start on SQLite, so this default
cannot reach a deployment.)

```powershell
copy .env.example .env
# Set APP_SECRET_KEY to any 32+ character value for local use.
uv run python -m src.app.cli migrate
uv run python -m src.app.cli seed --email owner@example.com
uv run python -m src.app.cli api --port 8000
```

Seeding creates the six leagues the product targets: two redraft and four
dynasty, two of them Superflex, one points-per-first-down, one with yardage
bonuses, five with a team defense, four with a kicker.

In another terminal:

```powershell
cd web
npm ci
npm run dev
```

Sign in at http://localhost:5173 using `APP_ALLOWED_EMAIL`. With dev auth
enabled, `POST /api/v1/auth/magic-link` returns `development_link` and the login
screen shows it as a link.

Then run one refresh — the **Run daily refresh** button on the Operations
screen, or `POST /api/v1/sync`. It imports all six leagues from the recorded
Sleeper fixtures. The job metadata's `unresolved_player_ids` should be empty; a
non-empty list means some rostered players have no projection.

The Operations screen states whether the data is `fixture` or `live` and whether
the weekly artifacts are `trained`. While it says fixture, nothing in the app is
a live or trained result.

Docker Compose (PostgreSQL + API + scheduler + web):

```powershell
copy .env.example .env
docker compose up --build
```

See [`docs/APP_OPERATIONS_RUNBOOK.md`](docs/APP_OPERATIONS_RUNBOOK.md) for
schedules, backups, and operations, and
[`docs/PRODUCTION_READINESS_AUDIT.md`](docs/PRODUCTION_READINESS_AUDIT.md) for
what has actually been verified versus what is still blocked on external
credentials or runtimes.

Verify the fixture MVP bundle:

```powershell
uv run python scripts/verify_mvp.py
uv run python scripts/audit_blueprint_mvp.py
uv run python scripts/vertical_smoke.py
uv run alembic check
cd web; npm test; npm run build; npm run test:e2e
```

`npm run test:e2e` starts its own disposable seeded API and drives a real owner
journey in a mobile browser; it never touches a database you are using.

## Common commands

```bash
# Full test suite
uv run pytest -q

# Build a promotion-eligible release bundle (clean git tree required)
uv run python -m src.projection.publish --season 2026 --simulation-profile publish --artifact-namespace <namespace>

# Validate and promote
uv run python scripts/validate_release_bundle.py --season 2026 --artifact-namespace <namespace>
uv run python -m src.projection.promote_release --season 2026 --artifact-namespace <namespace>

# Serve draft assistant locally
uv run python -m src.draft_assistant.serve --port 8766

# Browser verification (pointer-driven when --namespace is omitted)
uv run python scripts/verify_browser_surfaces.py --base-url http://127.0.0.1:8766 --season 2026
```

## Release model

New bundles use `release_bundle_manifest_v2` with six mandatory promotion invariants. Schema-v1 bundles remain readable but cannot be promoted. Rollback/restore uses tracked promotion receipts plus git ancestry — not the mutable validation sidecar. See `docs/PIPELINE_MAP.md` §8a/§10 and `docs/decisions/PROMOTION_PROVENANCE_2026-08-30.md`.
