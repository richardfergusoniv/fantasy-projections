# Required GitHub Production environment secrets (values are not stored in Git):
# - VERCEL_TOKEN — from https://vercel.com/account/tokens
# - VERCEL_ORG_ID — team id for `rdfergus15` (starts with `team_`)
# - VERCEL_PROJECT_ID — `prj_TrOVfWAUKG2VHfvV7PHiROTkFWH6` (`fantasy-projections`)
# - MIGRATION_DATABASE_URL — use the Supabase **session pooler** form from GitHub Actions
#   (IPv4). Example shape:
#   postgresql+psycopg://fantasy_app_migrator.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
#   Direct `db.<project-ref>.supabase.co:5432` is IPv6-only and fails on GitHub runners.
#
# Optional runtime verification secrets:
# - DATABASE_URL (transaction pooler for app)
# - JOB_DATABASE_URL (session pooler for jobs)
# - CRON_SECRET (must also exist in Supabase Vault as cron_secret)
