# Supabase + Vercel Deployment

See the approved deployment plan for architecture, cutover steps, and go/no-go gates.

## Pre-deploy checklist

1. Run `scripts/predeploy_secret_scan.sh`
2. Take manual `pg_dump -Fc` before migrations (Supabase Free has no PITR)
3. Upload release bundle: `uv run python scripts/upload_release_bundle.py --bundle-root ... --namespace v2_baseline_20260830 --write-pointer`
4. Rehearse selective import on staging
5. Benchmark daily refresh: `uv run python scripts/benchmark_daily_refresh.py --cold-start`

## Environment variables

Production runtime and CI variable names are listed in the deployment plan. Never place secrets in `VITE_*` variables.

## Cron

Apply `supabase/cron/run_due.sql` and `supabase/cron/process_outbox.sql` manually after storing `cron_secret` in Supabase Vault.

Long-running jobs (`daily-refresh`, Sunday slots, weekly close) execute via the **Production Jobs** GitHub Actions workflow (`.github/workflows/production-jobs.yml`), which pulls Vercel production env vars and runs `scheduler run-due` with `LONG_JOBS_EXTERNAL=false`. Vercel cron endpoints enqueue only and process short outbox jobs.

## Rollback

1. `vercel rollback`
2. Restore from manual `pg_dump` if schema migration applied
3. `SELECT cron.unschedule('fantasy-run-due');`
