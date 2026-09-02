-- Least-privilege database roles for Supabase deployment.
-- Run as postgres / supabase_admin after creating passwords in the dashboard.
-- Do not commit passwords to Git.

\set ON_ERROR_STOP on

-- One-time manual step (dashboard or psql):
--   CREATE ROLE fantasy_app_migrator LOGIN PASSWORD '<secure-password>';
--   CREATE ROLE fantasy_app_runtime LOGIN PASSWORD '<secure-password>';

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fantasy_app_migrator') THEN
    RAISE NOTICE 'fantasy_app_migrator missing — create role manually before applying grants';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fantasy_app_runtime') THEN
    RAISE NOTICE 'fantasy_app_runtime missing — create role manually before applying grants';
  END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO fantasy_app_migrator;
GRANT CONNECT ON DATABASE postgres TO fantasy_app_runtime;

GRANT USAGE, CREATE ON SCHEMA public TO fantasy_app_migrator;
GRANT USAGE ON SCHEMA public TO fantasy_app_runtime;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fantasy_app_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fantasy_app_runtime;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fantasy_app_migrator;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fantasy_app_migrator;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fantasy_app_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO fantasy_app_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT ALL ON TABLES TO fantasy_app_migrator;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT ALL ON SEQUENCES TO fantasy_app_migrator;

REVOKE CREATE ON SCHEMA public FROM fantasy_app_runtime;

-- Alembic must read alembic_version under RLS; migrator also needs BYPASSRLS for migrations.
ALTER TABLE alembic_version DISABLE ROW LEVEL SECURITY;
ALTER ROLE fantasy_app_migrator BYPASSRLS;
ALTER ROLE fantasy_app_runtime BYPASSRLS;
GRANT SELECT ON alembic_version TO fantasy_app_migrator;

-- Canonical domain promotion updates Vault production_app_url from deploy CI.
GRANT USAGE ON SCHEMA vault TO fantasy_app_migrator;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vault TO fantasy_app_migrator;
--   SET ROLE fantasy_app_runtime;
--   CREATE TABLE public.__runtime_ddl_probe(id int);  -- must fail
--   RESET ROLE;
