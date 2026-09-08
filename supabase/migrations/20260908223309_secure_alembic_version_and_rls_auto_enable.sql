-- Clear Supabase advisor ERROR: public.alembic_version exposed via Data API without RLS.
-- App roles already have BYPASSRLS (see supabase/roles.sql), so Alembic/runtime keep working.
ALTER TABLE public.alembic_version ENABLE ROW LEVEL SECURITY;

-- Deny PostgREST roles even if table grants remain from default Supabase privileges.
REVOKE ALL ON TABLE public.alembic_version FROM anon, authenticated;

-- Clear advisor WARNs: SECURITY DEFINER event-trigger helper must not be RPC-callable.
REVOKE ALL ON FUNCTION public.rls_auto_enable() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.rls_auto_enable() FROM anon, authenticated;
