# Supabase + Vercel Production Repair Report

Generated: 2026-09-02 (repair session)

## Executive summary

**Overall verdict: GO.** Fantasy Decisions PWA + API is live on `https://fantasy-projections-xi.vercel.app` with healthy DB, release pointer, overlay pointer, and first production daily refresh completed. `fantasy-projections-xi.vercel.app` is the adopted canonical production URL: it is the alias `vercel deploy --prod` assigns automatically and is owned by the `rdfergus15` team. The prior target `fantasy-projections.vercel.app` was abandoned because its alias is held by a legacy Next.js project on a Vercel account this team does not control; chasing it is no longer part of the deploy flow.

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
| Supabase cron | **GO** | Four `fantasy-*` pg_cron jobs active. GitHub Actions `production-jobs.yml` green: run `33597842583` with `PRODUCTION_JOB_ENV` set. |
| Vercel deployment (canonical = xi) | **GO** | `https://fantasy-projections-xi.vercel.app`: `/health/live` 200; `/health/ready` 200 with `release_pointer: true`, `overlay_pointer: true`, `last_daily_refresh_ok: true`. Adopted as the canonical production URL. |
| First production daily refresh | **GO** | Job `6a77ce84` succeeded 2026-09-02T05:06:24Z; 6 leagues, live Sleeper, overlay promoted. |
| Secret hygiene in Git | **GO** | No secrets committed. |

## Vercel production (current)

| Field | Value |
|-------|-------|
| Project | `rdfergus15/fantasy-projections` (`prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6`) |
| Canonical production URL | `https://fantasy-projections-xi.vercel.app` (adopted) |
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
| Cron schedules | `fantasy-run-due`, `fantasy-run-due-sunday-kickoff`, `fantasy-run-due-sunday-late`, `fantasy-process-outbox` |

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
| `PRODUCTION_JOB_ENV` | Set (2026-09-02) |

## Canonical domain decision (resolved)

`fantasy-projections-xi.vercel.app` is the canonical production URL. The prior target `fantasy-projections.vercel.app` was **abandoned**: its alias is held by a legacy Next.js project on a Vercel account this team does not control (`vercel alias set` → "already in use"; the current `richardfergusoniv` / `rdfergus15` token cannot resolve it), and reclaiming it would require account/support access outside this team. Because `-xi` is the alias `vercel deploy --prod` assigns automatically and the runtime env (`APP_PUBLIC_URL`, `APP_CORS_ORIGINS`, `TRUSTED_HOSTS`) and Vault `production_app_url` already point at it, no separate promotion is needed. The foreign-domain promotion tooling (`scripts/promote_canonical_domain.{sh,ps1}`, `scripts/diagnose_canonical_domain.ps1`, and the "Promote canonical domain" step in `deploy-production.yml`) has been removed. If a nicer hostname is wanted later, add a domain you own to `prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6` and repoint `APP_PUBLIC_URL` / CORS / TRUSTED_HOSTS / Vault `production_app_url` to it.

## Remaining items (non-domain)

1. **Pre-repair `pg_dump -Fc`** not captured locally (`pg_dump` not on PATH).
2. **Disable** direct Vercel Git production deploy in dashboard (manual).

## Commit / CI reference

| Item | Value |
|------|-------|
| Repair commits | `9e00733` … `9093e2d` |
| Green CI | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33583086436 |
| Green deploy (production) | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33603536484 (latest, auto-promote skip verified), `33602005866`, `33601204545` |
| Green production jobs | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33603293313 (scheduled), `33601695404`, `33599358252` |

## Confirmations

- Sealed production projection source remains `v2_baseline_20260830`.
- Weekly-v2 R&D disabled in production (`WEEKLY_RND_ENABLED=false`).
- Sleeper access read-only.
- No secrets committed to Git.
