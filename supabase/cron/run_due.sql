-- Idempotent Supabase pg_cron + pg_net schedule for production cron dispatch.
-- Prerequisites:
--   create extension if not exists pg_cron with schema extensions;
--   create extension if not exists pg_net with schema extensions;
--   select vault.create_secret('<cron-secret>', 'cron_secret', 'Bearer token for Vercel cron');
--   select vault.create_secret('https://fantasy-projections.vercel.app', 'production_app_url', 'Canonical production URL');

select cron.unschedule(jobid)
from cron.job
where jobname = 'fantasy-run-due';

-- Daily at 5:00 PM America/Los_Angeles during PDT (17:00 Pacific = 00:00 UTC).
select cron.schedule(
  'fantasy-run-due',
  '0 0 * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret
      from vault.decrypted_secrets
      where name = 'production_app_url'
    ) || '/api/internal/cron/run-due',
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

-- Sunday refresh near the 10:00 AM Pacific kickoff window.
select cron.unschedule(jobid)
from cron.job
where jobname = 'fantasy-run-due-sunday-kickoff';

select cron.schedule(
  'fantasy-run-due-sunday-kickoff',
  '0 17 * * 0',
  $$
  select net.http_post(
    url := (
      select decrypted_secret
      from vault.decrypted_secrets
      where name = 'production_app_url'
    ) || '/api/internal/cron/run-due',
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

-- Additional Sunday pass for later-window inactive news.
select cron.unschedule(jobid)
from cron.job
where jobname = 'fantasy-run-due-sunday-late';

select cron.schedule(
  'fantasy-run-due-sunday-late',
  '0 20 * * 0',
  $$
  select net.http_post(
    url := (
      select decrypted_secret
      from vault.decrypted_secrets
      where name = 'production_app_url'
    ) || '/api/internal/cron/run-due',
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
