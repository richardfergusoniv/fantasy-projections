# Expose the local web stack (Vite :5173 -> API :8000) via Cloudflare quick tunnel.
param(
    [int]$WebPort = 5173,
    [string]$ReportPath = "output/live_pg/cloudflare_tunnel.json"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")

function Find-Cloudflared {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "${env:ProgramFiles}\cloudflared\cloudflared.exe",
        (Join-Path $PSScriptRoot "..\tools\cloudflared.exe")
    )
    foreach ($path in $candidates) {
        $resolved = Resolve-Path $path -ErrorAction SilentlyContinue
        if ($resolved) { return $resolved.Path }
    }
    throw "cloudflared not found; install with: winget install Cloudflare.cloudflared"
}

$cloudflared = Find-Cloudflared
$logPath = Join-Path $root "output/live_pg/cloudflared.log"
$parent = Split-Path $logPath -Parent
if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
if (Test-Path $logPath) { Remove-Item $logPath -Force }

Write-Host "Starting Cloudflare quick tunnel -> http://127.0.0.1:$WebPort"
Write-Host "Log: $logPath"

$proc = Start-Process -FilePath $cloudflared -ArgumentList @(
    "tunnel", "--url", "http://127.0.0.1:$WebPort", "--logfile", $logPath, "--loglevel", "info"
) -PassThru -NoNewWindow

$deadline = (Get-Date).AddSeconds(45)
$url = $null
while ((Get-Date) -lt $deadline) {
    if (Test-Path $logPath) {
        $content = Get-Content $logPath -Raw -ErrorAction SilentlyContinue
        if ($content -match '(https://[a-z0-9-]+\.trycloudflare\.com)') {
            $url = $Matches[1]
        }
    }
    if ($url) { break }
    Start-Sleep -Seconds 1
}

if (-not $url) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for tunnel URL; see $logPath"
}

$publicHost = ([Uri]$url).Host
$payload = [ordered]@{
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    public_url = $url
    public_host = $publicHost
    web_port = $WebPort
    pid = $proc.Id
    log_path = $logPath
}
$reportFull = Join-Path $root $ReportPath
$reportParent = Split-Path $reportFull -Parent
if ($reportParent -and -not (Test-Path $reportParent)) {
    New-Item -ItemType Directory -Force -Path $reportParent | Out-Null
}
$payload | ConvertTo-Json | Set-Content -Path $reportFull -Encoding utf8

Write-Host "OK   tunnel URL: $url"
Write-Host "OK   use APP_PUBLIC_URL=$url"
Write-Host "OK   use TRUSTED_HOSTS=$publicHost"
Write-Host "PID  $($proc.Id) (stop with: Stop-Process -Id $($proc.Id))"
