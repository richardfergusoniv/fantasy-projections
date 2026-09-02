# Required GitHub Production environment secrets (values are not stored in Git):
# - VERCEL_TOKEN — from https://vercel.com/account/tokens
# - VERCEL_ORG_ID — team id for `rdfergus15` (starts with `team_`)
# - VERCEL_PROJECT_ID — `prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6` (`fantasy-projections`)
# - MIGRATION_DATABASE_URL — use the Supabase **session pooler** form from GitHub Actions
#   (IPv4). Example shape:
#   postgresql+psycopg://fantasy_app_migrator.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
#   Direct `db.<project-ref>.supabase.co:5432` is IPv6-only and fails on GitHub runners.
#
# Production job runner (`.github/workflows/production-jobs.yml`):
# - PRODUCTION_JOB_ENV — multiline env block copied from the Vercel production dashboard
#   (or a local production `.env` used for the first daily refresh). Must include at least:
#   APP_ENV=production
#   APP_SECRET_KEY=...
#   DATABASE_URL=postgresql+psycopg://fantasy_app_runtime...:6543/postgres
#   JOB_DATABASE_URL=postgresql+psycopg://fantasy_app_runtime...:5432/postgres
#   ARTIFACT_BACKEND=s3
#   S3_ENDPOINT_URL=...
#   S3_BUCKET=fantasy-app
#   S3_ACCESS_KEY_ID=...
#   S3_SECRET_ACCESS_KEY=...
#   SLEEPER_USER_ID=...
#   SLEEPER_USERNAME=...
#   SLEEPER_USE_FIXTURES=false
#   WEEKLY_RND_ENABLED=false
#   STATUS_OVERLAY_AUTO_PUBLISH=true
#   Note: `vercel env pull` cannot export sensitive Vercel secrets into CI (values become
#   `[SENSITIVE]`); paste the runtime block into this GitHub secret instead.
#
# Optional runtime verification secrets:
# - DATABASE_URL (transaction pooler for app)
# - JOB_DATABASE_URL (session pooler for jobs)
# - CRON_SECRET (must also exist in Supabase Vault as cron_secret)
