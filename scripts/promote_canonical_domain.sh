#!/usr/bin/env bash
# Promote fantasy-projections.vercel.app to the current PWA/API deployment.
# Exits 0 when canonical already serves Fantasy Decisions or promotion succeeds.
# Exits 0 with a skip message when the alias is still owned elsewhere (non-fatal in CI).
set -euo pipefail

DEPLOYMENT_URL="${DEPLOYMENT_URL:-https://fantasy-projections-xi.vercel.app}"
CANONICAL_HOST="${CANONICAL_HOST:-fantasy-projections.vercel.app}"
PUBLIC_URL="https://${CANONICAL_HOST}"

canonical_live="$(curl -sS -o /tmp/canonical_live.json -w "%{http_code}" "${PUBLIC_URL}/health/live" || true)"
curl -sS -o /tmp/canonical_home.html "${PUBLIC_URL}/" || true
if grep -q "Fantasy Decisions" /tmp/canonical_home.html 2>/dev/null && [ "${canonical_live}" = "200" ]; then
  echo "Canonical domain already serves Fantasy Decisions PWA."
  exit 0
fi

if grep -q "Fantasy Projections" /tmp/canonical_home.html 2>/dev/null; then
  echo "SKIP: ${CANONICAL_HOST} still serves legacy Next.js (health/live=${canonical_live})."
  echo "Remove the domain from the legacy Vercel project, then re-run deploy or this script."
  exit 0
fi

echo "Assigning alias ${CANONICAL_HOST} -> ${DEPLOYMENT_URL}"
if ! vercel alias set "${DEPLOYMENT_URL}" "${CANONICAL_HOST}" --token="${VERCEL_TOKEN:?VERCEL_TOKEN required}"; then
  echo "SKIP: could not assign canonical alias (likely still in use on another account)."
  exit 0
fi

CORS="${PUBLIC_URL},https://fantasy-projections-xi.vercel.app,https://fantasy-projections-rdfergus15.vercel.app"
HOSTS="${CANONICAL_HOST},fantasy-projections-xi.vercel.app,fantasy-projections-rdfergus15.vercel.app"

for pair in "APP_PUBLIC_URL:${PUBLIC_URL}" "APP_CORS_ORIGINS:${CORS}" "TRUSTED_HOSTS:${HOSTS}"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  vercel env rm "${name}" production --yes --token="${VERCEL_TOKEN}" >/dev/null 2>&1 || true
  printf '%s' "${value}" | vercel env add "${name}" production --token="${VERCEL_TOKEN}"
  echo "Updated Vercel env ${name}"
done

echo "Apply Supabase Vault SQL manually or via MCP:"
cat <<EOF
SELECT vault.update_secret(
  (SELECT id FROM vault.secrets WHERE name = 'production_app_url'),
  '${PUBLIC_URL}',
  'production_app_url',
  'Canonical production URL'
);
EOF

live_status="$(curl -sS -o /tmp/canonical_live.json -w "%{http_code}" "${PUBLIC_URL}/health/live")"
if [ "${live_status}" != "200" ]; then
  echo "Canonical /health/live returned ${live_status}"
  exit 1
fi
curl -sS -o /tmp/canonical_index.html "${PUBLIC_URL}/"
if ! grep -q "Fantasy Decisions" /tmp/canonical_index.html; then
  echo "Canonical home page does not serve Fantasy Decisions PWA"
  exit 1
fi

echo "Canonical promotion verified."
