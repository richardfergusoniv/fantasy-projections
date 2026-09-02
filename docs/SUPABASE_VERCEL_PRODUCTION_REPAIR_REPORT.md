# Supabase + Vercel Production Repair Report

Generated: 2026-09-02 (repair session; latest local changes pending commit)

## Executive summary

**Overall verdict: NO-GO for full production promotion.** Clean-clone CI and deploy verify remain green. Supabase roles, release pointer, cron, and Vault are configured. **Fantasy Decisions PWA + API are live** on `https://fantasy-projections-xi.vercel.app` (`/health/live` → 200 JSON). Canonical `https://fantasy-projections.vercel.app` still serves the legacy Next.js app. First production daily refresh **in progress** (local run with real Sleeper user ID after fixing `fixture-user-1` fallback).

## GO / NO-GO matrix

| Area | Verdict | Evidence |
|------|---------|----------|
| GitHub CI (clean-clone tests) | **GO** | Run `33583086436` on `c19f6c0`/`9e00733`: 864 passed, 2 skipped (Windows). |
| Deploy verify (pre-promote gate) | **GO** | Run `33584188968` verify job green. |
| Deploy production (migrate + Vercel) | **PARTIAL** | Run `33585574535`: migrate green. Run `33586475165`: failed Vercel bundle size (412 MB > 225 MB). CLI deploy with `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` succeeded; GitHub secrets (`VERCEL_*`) now set. Workflow fixes pending commit/push. |
| Supabase DB reachability | **GO** | Project `dbvwgfefdorugdtpxgcj`; Alembic `a1b2c3d4e5f7`. |
| Supabase Storage (sealed bundle) | **GO** | `fantasy-app` bucket; `manifest_storage_uri` on `release_pointer` for 2026 → `v2_baseline_20260830`. |
| Security roles | **GO** | `fantasy_app_migrator` + `fantasy_app_runtime` created; migrator `BYPASSRLS`; runtime `BYPASSRLS` (fixes `job_outbox` RLS). |
| Release pointer | **GO** | `release_pointer` row for 2026 with `manifest_storage_uri` set (SQL-verified). |
| Status overlay pointer | **NO-GO** | `status_overlay_pointer` empty; daily refresh running / not yet promoted. |
| Supabase cron | **GO (scheduled)** / **PARTIAL (runtime)** | Three `fantasy-run-due*` jobs; Vault `cron_secret` + `production_app_url` → `https://fantasy-projections-xi.vercel.app`. Inline cron exceeds Vercel 300s; use local/background job or `LONG_JOBS_EXTERNAL` pattern. |
| Vercel deployment (xi alias) | **GO** | `https://fantasy-projections-xi.vercel.app`: title **Fantasy Decisions**; `/health/live` 200; `/health/ready` 200 (DB ok). |
| Vercel deployment (canonical) | **NO-GO** | `https://fantasy-projections.vercel.app`: legacy app; `/health/live` 404. Alias owned outside `rdfergus15` project — manual transfer required. |
| First production daily refresh | **IN PROGRESS** | Failed run `231923e5` (fixture user). Retry `6a77ce84` started with `SLEEPER_USER_ID=739931264659927040`. |
| Secret hygiene in Git | **GO** | No secrets committed. |

## Vercel production (current)

| Field | Value |
|-------|-------|
| Project | `rdfergus15/fantasy-projections` (`prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6`) |
| Live production URL | `https://fantasy-projections-xi.vercel.app` |
| Canonical URL (blocked) | `https://fantasy-projections.vercel.app` (legacy Next.js) |
| Latest CLI deploy | `dpl_HKg7APb2hcJpHTgHptUVQcyKqpf4` (bundle fixes + env) |
| PWA title | **Fantasy Decisions** |
| `/health/live` | `{"status":"ok"}` |
| `/health/ready` | DB ok; `release_pointer`/`overlay_pointer` flags false until overlay refresh completes |

### Vercel bundle / runtime fixes (local, pending push)

- `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` on build + Vercel project env
- `vercel.json`: `excludeFiles`, `--extra jobs` install trimmed for serverless
- `scipy` in main deps; lazy imports for weekly/polars/sklearn paths
- Hash helpers moved out of `accuracy_first.py` to avoid sklearn at import
- `TRUSTED_HOSTS` includes `fantasy-projections-xi.vercel.app`
- Production env: DB poolers, S3, `CRON_SECRET`, `SLEEPER_USERNAME`, `SLEEPER_USER_ID`

## Supabase production (current)

| Item | State |
|------|-------|
| Alembic | `a1b2c3d4e5f7` |
| `pg_cron` / `pg_net` | Enabled |
| Vault | `cron_secret`, `production_app_url` → xi deployment |
| `fantasy_app_runtime` | `BYPASSRLS` (applied 2026-09-02) |
| `job_outbox` | One queued `daily-refresh` from cron smoke |

## Daily refresh attempts

| Job ID | Result | Notes |
|--------|--------|-------|
| `231923e5` | **failed** | Used `fixture-user-1` (missing `SLEEPER_USER_ID`); Sleeper 404 |
| `6a77ce84` | **running** | Real user `739931264659927040`; started 2026-09-02T04:33:40Z |

Code fix: `_resolve_sleeper_user_id()` resolves from `SLEEPER_USER_ID`, `SLEEPER_USERNAME`, or owner config username (no silent fixture fallback in production).

## GitHub secrets (Production environment)

| Secret | Status |
|--------|--------|
| `MIGRATION_DATABASE_URL` | Set (pooler form) |
| `VERCEL_TOKEN` | Set |
| `VERCEL_ORG_ID` | `team_2wadxBpdAExHEXF0iyvy6t1F` |
| `VERCEL_PROJECT_ID` | `prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6` |

## Remaining blockers

1. **Commit + push** Vercel bundle/runtime fixes; green Deploy Production workflow on `master`.
2. **Transfer** `fantasy-projections.vercel.app` from legacy Vercel project to `rdfergus15/fantasy-projections`.
3. **Complete** first production daily refresh; verify `status_overlay_pointer` + six-league sync.
4. **Pre-repair `pg_dump -Fc`** still not captured locally (`pg_dump` not on PATH).
5. **Disable** direct Vercel Git production deploy (dashboard) so only GitHub Actions promotes.

## Commit / CI reference

| Item | Value |
|------|-------|
| Starting commit | `d3bb30c` |
| Repair commits on `master` | `11f61f9` … `9e00733` |
| Green CI | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33583086436 |
| Green deploy verify | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33584188968 |
| Migrate green / Vercel fail | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33585574535 |
| Vercel bundle fail | https://github.com/richardfergusoniv/fantasy-projections/actions/runs/33586475165 |

## Confirmations

- Sealed production projection source remains `v2_baseline_20260830`.
- Weekly-v2 R&D disabled in production (`WEEKLY_RND_ENABLED=false`).
- Sleeper access read-only.
- No secrets committed to Git.
