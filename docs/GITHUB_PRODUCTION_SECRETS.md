# Required GitHub Production environment secrets (values are not stored in Git):
# - VERCEL_TOKEN
# - VERCEL_ORG_ID
# - VERCEL_PROJECT_ID
# - MIGRATION_DATABASE_URL
#
# Optional runtime verification secrets:
# - DATABASE_URL (transaction pooler for app)
# - JOB_DATABASE_URL (session pooler for jobs)
# - CRON_SECRET (must also exist in Supabase Vault as cron_secret)
