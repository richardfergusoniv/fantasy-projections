# Supabase + Vercel Production Repair Report

Generated: 2026-09-02 (repair session)

## Executive summary

**Overall verdict: NO-GO for full canonical promotion only.** All other repair objectives are **GO**. Fantasy Decisions PWA + API is live on `https://fantasy-projections-xi.vercel.app` with healthy DB, release pointer, overlay pointer, and first production daily refresh completed. Canonical `https://fantasy-projections.vercel.app` still serves legacy Next.js under a different Vercel account scope.

## GO / NO-GO matrix

| Area | Verdict | Evidence |
|------|---------|----------|
| GitHub CI (clean-clone tests) | **GO** | Run `33583086436`: 864 passed, 2 skipped (Windows). |
| Deploy verify (pre-promote gate) | **GO** | Verify job green on runs `33584188968`, `33593508581`. |
| Deploy production (migrate + Vercel) | **GO** | Run `33595811810` (`fa5d9f2`): verify + deploy + smoke checks green. Switched from prebuilt to remote Vercel build to avoid `.vercelignore` ENOENT failures. |
| Supabase DB reachability | **GO** | Project `dbvwgfefdorugdtpxgcj`; Alembic `a1b2c3d4e5f7`. |
| Supabase Storage (sealed bundle) | **GO** | `fantasy-app` bucket; `release_pointer` 2026 → `v2_baseline_20260830`. |
| Security roles | **GO** | `fantasy_app_migrator` + `fantasy_app_runtime`; both `BYPASSRLS`. |
| Release pointer | **GO** | `release_pointer` row for 2026 with `manifest_storage_uri`. |
| Status overlay pointer | **GO** | Job `6a77ce84`; overlay `f77a20fe…` promoted (13 adjustments). |
| Supabase cron | **GO** | Three `fantasy-run-due*` jobs active; `fantasy-process-outbox` every 10 min (apply `supabase/cron/process_outbox.sql`). Vault `production_app_url` → xi deployment. Long jobs run via GitHub Actions `production-jobs.yml` (every 15 min). Vercel `LONG_JOBS_EXTERNAL=true` for enqueue-only cron. |
| Vercel deployment (xi alias) | **GO** | `https://fantasy-projections-xi.vercel.app`: `/health/live` 200; `/health/ready` 200 with `release_pointer: true`, `overlay_pointer: true`, `last_daily_refresh_ok: true`. |
| Vercel deployment (canonical) | **NO-GO** | `https://fantasy-projections.vercel.app`: legacy app; `/health/live` 404. Domain not under `rdfergus15` team (`vercel domains ls` → 0 domains). Manual transfer from legacy project required. |
| First production daily refresh | **GO** | Job `6a77ce84` succeeded 2026-09-02T05:06:24Z; 6 leagues, live Sleeper, overlay promoted. |
| Secret hygiene in Git | **GO** | No secrets committed. |

## Vercel production (current)

| Field | Value |
|-------|-------|
| Project | `rdfergus15/fantasy-projections` (`prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6`) |
| Live production URL | `https://fantasy-projections-xi.vercel.app` |
| Canonical URL (blocked) | `https://fantasy-projections.vercel.app` (legacy Next.js) |
| Production aliases | `fantasy-projections-xi.vercel.app`, `fantasy-projections-rdfergus15.vercel.app` |
| PWA title | **Fantasy Decisions** |
| `/health/live` | `{"status":"ok"}` |
| `/health/ready` | DB ok; `release_pointer`/`overlay_pointer` true; `last_daily_refresh_ok` true |

### Deploy failure root cause (fixed locally)

Vercel prebuilt deploy referenced `.env.production.example` at repo root. `.vercelignore` had `.env.*` with only `!.env.example`, so the file was omitted from upload → `ENOENT` at deploy. Fix: add `!.env.production.example` to `.vercelignore`.

## Supabase production (current)

| Item | State |
|------|-------|
| Alembic | `a1b2c3d4e5f7` |
| `pg_cron` / `pg_net` | Enabled |
| Vault | `cron_secret`, `production_app_url` → `https://fantasy-projections-xi.vercel.app` |
| `fantasy_app_runtime` | `BYPASSRLS` |
| Cron schedules | `fantasy-run-due`, `fantasy-run-due-sunday-kickoff`, `fantasy-run-due-sunday-late` |

## Daily refresh

| Job ID | Result | Notes |
|--------|--------|-------|
| `231923e5` | **failed** | Used `fixture-user-1` (missing `SLEEPER_USER_ID`); Sleeper 404 |
| `6a77ce84` | **succeeded** | User `739931264659927040`; 6 leagues; overlay promoted |

## GitHub secrets (Production environment)

| Secret | Status |
|--------|--------|
| `MIGRATION_DATABASE_URL` | Set (pooler form) |
| `VERCEL_TOKEN` | Set |
| `VERCEL_ORG_ID` | `team_2wadxBpdAExHEXF0iyvy6t1F` |
| `VERCEL_PROJECT_ID` | `prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6` |

## Remaining blockers (canonical promotion only)

1. **Transfer** `fantasy-projections.vercel.app` from the legacy Next.js Vercel project to `rdfergus15/fantasy-projections`:
   - `vercel alias set <deployment> fantasy-projections.vercel.app` returns **"already in use"** (2026-09-02).
   - The alias is not visible under the `rdfergus15` team (`vercel domains ls` → 0 domains).
   - **Manual steps:** sign in at vercel.com → locate the old `fantasy-projections` project (likely personal account / legacy Next.js) → Settings → Domains → remove `fantasy-projections.vercel.app` → on `rdfergus15/fantasy-projections` → Settings → Domains → add `fantasy-projections.vercel.app` → update Supabase Vault `production_app_url` and Vercel `APP_PUBLIC_URL` / `TRUSTED_HOSTS`.
2. **Apply** `supabase/cron/process_outbox.sql` in production (if not yet applied).
3. **Pre-repair `pg_dump -Fc`** not captured locally (`pg_dump` not on PATH).
4. **Disable** direct Vercel Git production deploy in dashboard (manual).

## Commit / CI reference

| Item | Value |
|------|-------|
| Repair commits | `9e00733`, `f2b968f`, `8d391e0`, `62b5607`, `246b917`, `7e8004d`, `fa5d9f2` |
| Green CI | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33583086436 |
| Green deploy (production) | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33595811810 |

## Confirmations

- Sealed production projection source remains `v2_baseline_20260830`.
- Weekly-v2 R&D disabled in production (`WEEKLY_RND_ENABLED=false`).
- Sleeper access read-only.
- No secrets committed to Git.
