param(
    [string]$EnvFile = ".env.production.jobs"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$path = Join-Path $root $EnvFile

if (-not (Test-Path $path)) {
    Write-Error "Missing $path. Copy .env.production.jobs.example and fill production values from the Vercel dashboard."
}

$body = Get-Content -Path $path -Raw
if ($body -match '\[SENSITIVE\]|replace-me') {
    Write-Error "$path still contains placeholder values. Fill real production values before pushing the secret."
}

gh secret set PRODUCTION_JOB_ENV --env production --body $body
Write-Host "Set PRODUCTION_JOB_ENV on the GitHub production environment."
