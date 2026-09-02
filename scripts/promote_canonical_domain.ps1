# Promote fantasy-projections.vercel.app to the current PWA/API deployment.
# Prerequisite: remove the alias from the legacy Next.js project in the Vercel dashboard first.
param(
    [string]$DeploymentUrl = "https://fantasy-projections-xi.vercel.app",
    [string]$CanonicalHost = "fantasy-projections.vercel.app",
    [string]$SupabaseProjectRef = "dbvwgfefdorugdtpxgcj"
)

$ErrorActionPreference = "Stop"

Write-Host "Assigning alias $CanonicalHost -> $DeploymentUrl"
vercel alias set $DeploymentUrl $CanonicalHost

$publicUrl = "https://$CanonicalHost"
$cors = "$publicUrl,https://fantasy-projections-xi.vercel.app,https://fantasy-projections-rdfergus15.vercel.app"
$hosts = "$CanonicalHost,fantasy-projections-xi.vercel.app,fantasy-projections-rdfergus15.vercel.app"

foreach ($pair in @(
    @("APP_PUBLIC_URL", $publicUrl),
    @("APP_CORS_ORIGINS", $cors),
    @("TRUSTED_HOSTS", $hosts)
)) {
    $name, $value = $pair
    vercel env rm $name production --yes 2>$null | Out-Null
    $value | vercel env add $name production 2>&1 | Out-Null
    Write-Host "Updated Vercel env $name"
}

Write-Host "Updating Supabase Vault production_app_url -> $publicUrl"
$sql = @"
SELECT vault.update_secret(
  (SELECT id FROM vault.secrets WHERE name = 'production_app_url'),
  '$publicUrl',
  'production_app_url',
  'Canonical production URL'
);
"@
# Requires Supabase CLI or MCP; print SQL if not available.
Write-Host $sql

Write-Host "Verifying canonical health..."
$status = curl.exe -sS -o NUL -w "%{http_code}" "$publicUrl/health/live"
if ($status -ne "200") {
    throw "Expected HTTP 200 from $publicUrl/health/live, got $status"
}
$body = curl.exe -sS "$publicUrl/" 
if ($body -notmatch "Fantasy Decisions") {
    throw "Canonical home page does not serve Fantasy Decisions PWA"
}
Write-Host "Canonical promotion verified."
