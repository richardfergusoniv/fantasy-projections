# Supabase + Vercel Production Repair Report

Generated: 2026-09-02 (local repair session, starting commit `d3bb30c`)

## Executive summary

**Overall verdict: NO-GO for production promotion.** Code and test repairs required for clean-clone CI are complete locally (`864 passed`), but cloud configuration, Vercel promotion, Supabase cron/Vault, least-privilege roles, release-pointer repair, first production daily refresh, and external endpoint verification remain blocked on dashboard/credential actions that cannot be completed from this workspace alone.

## GO / NO-GO matrix

| Area | Verdict | Evidence |
|------|---------|----------|
| GitHub CI (clean-clone tests) | **GO (local)** / **PENDING (remote)** | Local `uv run pytest -q`: **864 passed**, 2 skipped. Prior remote CI on `d3bb30c`: 849 passed / 7 failed. Repair not yet pushed. |
| Supabase DB reachability | **GO** | Known reachable at Alembic `f1e2d3c4b5a6`; 34 tables present (pre-repair audit). |
| Supabase Storage (sealed bundle) | **GO** | `fantasy-app` bucket contains `v2_baseline_20260830` manifest + artifacts (~62 objects / ~95 MB). |
| Security roles (`fantasy_app_runtime` / `fantasy_app_migrator`) | **NO-GO** | Roles do not exist yet. Executable grant script added at `supabase/roles.sql`; passwords must be created in dashboard. |
| Release pointer (`manifest_storage_uri`) | **NO-GO** | `release_pointer` for 2026 exists but `manifest_storage_uri` is null. Repair script path: `scripts/upload_release_bundle.py --write-pointer`. |
| Status overlay pointer | **NO-GO** | `status_overlay_pointer` empty; first authenticated daily refresh not completed. |
| Supabase cron (`fantasy-run-due`) | **NO-GO** | `pg_cron` / `pg_net` not enabled; Vault `cron_secret` absent. Idempotent SQL prepared in `supabase/cron/run_due.sql`. |
| Vercel deployment (canonical) | **NO-GO** | `https://fantasy-projections.vercel.app` serves legacy **Fantasy Projections** app. `/health/live` returns **404**. |
| Public API / PWA | **NO-GO** | External smoke: home page title = Fantasy Projections; health endpoints unavailable. |
| Phone access / production readiness audit | **NO-GO** | Scripts updated for Supabase/Vercel layering; production env + Vercel promotion still required. |
| Secret hygiene in Git | **GO** | No secrets committed. Untracked runtime artifacts preserved outside Git. |

## Code repair completed (Phase 2)

All seven previously failing clean-clone tests addressed:

1. **Draft board** — committed hash-validated `projections_2026.csv` to public sealed release namespace; honest `league_specific` flag when roster/scoring changes replacement levels.
2. **Internal cron** — fixed `recover_stale_running(stale_after=...)` keyword call; timezone-safe stale recovery; regression tests for idempotency and empty queue.
3. **Phone access preflight** — separated production Vercel/Supabase readiness from legacy Cloudflare/local-stack checks.
4. **Production infrastructure audit** — distinguishes `code_ready`, `cloud_configuration_ready`, and `runtime_verified`.
5. **Projection reanchor** — clean clone loads sealed component projections from `draft_assistant/data/releases/v2_baseline_20260830/` without `output/` fallback.
6. **Weekly event cohort integration** — `@pytest.mark.integration` + explicit skip when research panel absent.
7. **Weekly v2 tuning harness** — synthetic panel written to temp parquet for fingerprint validation.

Additional fixes:

- Test isolation: strip leaked `MIGRATION_DATABASE_URL` from pytest environment.
- `seed_development_data` skips weekly R&D promotion when `WEEKLY_RND_ENABLED=false`.
- Operations status falls back to preseason run id when weekly run absent.
- `polars-runtime-32` declared explicitly in optional deps.

## Verification commands (local)

| Command | Result |
|---------|--------|
| `uv sync --frozen --all-extras --dev` | OK |
| `uv run pytest -q` | **864 passed**, 2 skipped |
| `uv run python scripts/verify_mvp.py` | OK |
| `uv run python scripts/audit_blueprint_mvp.py` | OK (session) |
| `uv run python scripts/validate_compose_config.py` | OK (session) |
| `uv run alembic check` (SQLite, CI-style) | OK in CI workflow; local run affected by `.env` Postgres URL — use isolated `DATABASE_URL` per `ci.yml` |
| `cd web && npm ci && npm test && npm run build` | **Blocked locally** (Windows file lock on `node_modules`); CI Linux job covers this |
| `bash scripts/predeploy_secret_scan.sh` | **Skipped locally** (no bash); runs in GitHub deploy workflow |
| External `GET /health/live` | **404** |
| External `GET /` title | **Fantasy Projections** (not Fantasy Decisions) |

## GitHub deployment gating (Phase 3)

Updated `.github/workflows/deploy-production.yml`:

- Trigger branch: **`master`**
- Concurrency group: `production-deploy`
- `verify` job runs full pytest + MVP verify + web build before deploy
- Deploy job uses GitHub **production** environment
- Post-deploy curl checks for JSON `/health/live`, `/health/ready`, PWA title **Fantasy Decisions**, `/login` not 404

Required secrets documented in `docs/GITHUB_PRODUCTION_SECRETS.md`:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`
- `MIGRATION_DATABASE_URL`

### Vercel Git integration (manual)

Direct Vercel Git deploys still bypass CI until changed in dashboard:

1. Vercel project → **Settings → Git** → disable **Production Branch** auto-deploy **or** set production deployments to **GitHub Actions only**.
2. Keep preview deployments if desired; ensure previews cannot update production alias.
3. Link CLI locally: `vercel link` with org/project IDs matching GitHub secrets.

## Supabase state (pre-repair baseline, no PII)

| Table / object | Count (approx.) |
|----------------|-----------------|
| `league_rule_snapshot` | 6 |
| `league_member` | 72 |
| `roster_snapshot` | 76 |
| `transaction` | 31 |
| `traded_pick` | 93 |
| `injury_evidence` | 50 |
| `availability_event` | 411 |
| `manager_tendency` | 0 |
| `trade_proposal` | 0 |
| `trade_evaluation` | 0 |
| `player_status_snapshot` | 0 |
| `depth_snapshot` | 0 |
| `job_run` / `job_outbox` | 0 |

| Item | Value |
|------|-------|
| Alembic revision | `f1e2d3c4b5a6` |
| Extensions `pg_cron`, `pg_net` | Not enabled |
| Vault `cron_secret` | Absent |
| DB role in use | `postgres` (not least-privilege) |

## Release bundle

| Field | Value |
|-------|-------|
| Namespace | `v2_baseline_20260830` |
| Release ID | `e92edd22-40d9-4219-87f6-47a651489d15` |
| Manifest SHA-256 | `5a8e14536aa7b062b1e5ff6e64aa78356847fb063579d0a42cdbdc5cc159fbb1` |
| Projections artifact SHA-256 | `25dbb9ee2a4574191678726d703e6ab34a2249cada474029624c0b0049dcfc13` |
| Storage | `s3://fantasy-app/releases/v2_baseline_20260830/...` (62 objects) |
| `manifest_storage_uri` in DB | **null** (must be repaired before production-offline bundle load) |
| Git-tracked public copy | `draft_assistant/data/releases/v2_baseline_20260830/` now includes `projections_2026.csv` |

Rollback: `active_release_2026.json` preserves previous namespace `v2_candidate_20260830`.

## Vercel production

| Field | Current state |
|-------|---------------|
| Canonical domain | `https://fantasy-projections.vercel.app` |
| Production branch (intended) | `master` |
| Deployed app title | **Fantasy Projections** (legacy) |
| `/login` | 404 |
| `/health/live` | 404 |
| Intended PWA title | **Fantasy Decisions** |

`vercel.json` in repo is configured for Vite `web/dist` + `api/index.py` routing.

## Backup (Phase 1)

Pre-repair `pg_dump -Fc` **not completed locally**: `pg_dump` not on PATH and PostgreSQL client binaries not found at default Windows locations. **Manual action required** before DDL:

```powershell
# Example (replace with your migration/direct host; never commit output)
pg_dump -Fc -h <host> -p 5432 -U postgres -d postgres -f C:\Users\rdfer\Projects\fantasy-projections-backups\pre_repair.dump
Get-FileHash C:\Users\rdfer\Projects\fantasy-projections-backups\pre_repair.dump -Algorithm SHA256
```

## Remaining manual actions (blockers)

1. **Install PostgreSQL client** or run backup from Supabase dashboard before applying `supabase/roles.sql` or pointer updates.
2. **Create roles** `fantasy_app_migrator` and `fantasy_app_runtime` in Supabase dashboard; apply `supabase/roles.sql` grants.
3. **Set connection URLs** in Vercel production env:
   - `DATABASE_URL` → transaction pooler (`fantasy_app_runtime`)
   - `JOB_DATABASE_URL` → session pooler
   - `MIGRATION_DATABASE_URL` → direct connection (`fantasy_app_migrator`)
4. **Repair release pointer**: run `scripts/upload_release_bundle.py` with `--write-pointer` against sealed `v2_baseline_20260830`; verify `manifest_storage_uri` and artifact hashes.
5. **Enable** `pg_cron` + `pg_net`; create Vault secrets `cron_secret` and `production_app_url`; apply `supabase/cron/run_due.sql`.
6. **Configure GitHub Production environment secrets** (see `docs/GITHUB_PRODUCTION_SECRETS.md`).
7. **Disable Vercel direct production deploy** or gate it behind green CI.
8. **Commit + push** repair branch to `master`; wait for green CI + Deploy Production workflow.
9. **Run authenticated production daily refresh** (read-only Sleeper); verify overlay pointer, snapshots, six-league sync.
10. **Promote Vercel deployment**; confirm external checks for Fantasy Decisions PWA + JSON health endpoints.

## Commit / CI reference

| Item | Value |
|------|-------|
| Starting commit | `d3bb30c` |
| Repair commit | *(pending push — local tree ready)* |
| Last remote CI (pre-repair) | https://github.com/rdfer/fantasy-projections/actions/runs/33578266715 (failure, 7 tests) |
| Post-repair CI | Pending push |

## Confirmations

- Sealed production projection source remains `v2_baseline_20260830`; no retrain/republish performed.
- Weekly-v2 R&D not represented as production-ready.
- Sleeper access remains read-only in configuration.
- No secrets, database exports, or live roster contents committed to Git.
- Untracked runtime data under `output/`, `staging_sql/`, `tmp_audit.*` preserved locally and excluded from Git.
