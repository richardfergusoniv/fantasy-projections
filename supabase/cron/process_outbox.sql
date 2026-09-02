-- Idempotent Supabase pg_cron schedule for short outbox jobs on Vercel.
-- Long jobs (daily-refresh, sunday-*, weekly-*) are skipped server-side when
-- VERCEL=1; run them via the GitHub Actions production-jobs workflow.

select cron.unschedule(jobid)
from cron.job
where jobname = 'fantasy-process-outbox';

select cron.schedule(
  'fantasy-process-outbox',
  '*/10 * * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret
      from vault.decrypted_secrets
      where name = 'production_app_url'
    ) || '/api/internal/cron/process-outbox',
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'Authorization', 'Bearer ' || (
        select decrypted_secret
        from vault.decrypted_secrets
        where name = 'cron_secret'
      )
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 250000
  );
  $$
);
