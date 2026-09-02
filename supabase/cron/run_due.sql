-- Reference cron job for Supabase pg_cron + pg_net.
-- Store CRON_SECRET in Supabase Vault; do not hardcode secrets in SQL.

-- Example (apply manually after approval):
-- select vault.create_secret('<cron-secret>', 'cron_secret', 'Bearer token for Vercel cron');

select cron.schedule(
  'fantasy-run-due',
  '*/5 * * * *',
  $$
  select net.http_post(
    url := 'https://<production-domain>/api/internal/cron/run-due',
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
