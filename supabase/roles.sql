-- Least-privilege database roles for Supabase deployment.
-- Apply manually after creating the database; passwords via Vault / dashboard.

-- Migrator: Alembic only (CI and manual migrations).
-- CREATE ROLE fantasy_app_migrator LOGIN PASSWORD '...';
-- GRANT CONNECT ON DATABASE postgres TO fantasy_app_migrator;
-- GRANT USAGE, CREATE ON SCHEMA public TO fantasy_app_migrator;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fantasy_app_migrator;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fantasy_app_migrator;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO fantasy_app_migrator;

-- Runtime: Vercel API and job workers (no DDL).
-- CREATE ROLE fantasy_app_runtime LOGIN PASSWORD '...';
-- GRANT CONNECT ON DATABASE postgres TO fantasy_app_runtime;
-- GRANT USAGE ON SCHEMA public TO fantasy_app_runtime;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fantasy_app_runtime;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fantasy_app_runtime;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fantasy_app_runtime;
