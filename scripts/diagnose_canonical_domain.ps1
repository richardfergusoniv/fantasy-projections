# Diagnose why fantasy-projections.vercel.app is not on the current PWA deployment.
param(
    [string]$CanonicalHost = "fantasy-projections.vercel.app",
    [string]$TargetHost = "fantasy-projections-xi.vercel.app"
)

$ErrorActionPreference = "Continue"

Write-Host "=== Canonical domain diagnostic ===" -ForegroundColor Cyan

$liveStatus = curl.exe -sS -o NUL -w "%{http_code}" "https://$CanonicalHost/health/live"
$homeBody = curl.exe -sS "https://$CanonicalHost/" 2>$null
$title = "(unknown)"
if ($homeBody -and $homeBody -match "<title>([^<]+)</title>") {
    $title = $Matches[1]
}

Write-Host "Canonical: https://$CanonicalHost"
Write-Host "  /health/live -> HTTP $liveStatus"
Write-Host "  home title   -> $title"

if ($title -match "Fantasy Decisions" -and $liveStatus -eq "200") {
    Write-Host "GO: canonical already serves Fantasy Decisions PWA." -ForegroundColor Green
    exit 0
}

if ($title -match "Fantasy Projections") {
    Write-Host "BLOCKED: legacy Next.js still owns the canonical alias." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Current Vercel account aliases (rdfergus15):"
vercel alias ls 2>&1 | Select-Object -Skip 2

Write-Host ""
Write-Host "Alias assignment probe:"
vercel alias set "https://$TargetHost" $CanonicalHost 2>&1

Write-Host ""
Write-Host "Unblock steps:"
Write-Host "  1. Open https://vercel.com/login"
Write-Host "  2. Try BOTH 'Continue with GitHub' and 'Continue with Google' (separate accounts can share an email)"
Write-Host "  3. In each account, find a project serving 'Fantasy Projections' (Next.js)"
Write-Host "  4. Settings -> Domains -> remove $CanonicalHost"
Write-Host "  5. Run: powershell -File scripts/promote_canonical_domain.ps1"
