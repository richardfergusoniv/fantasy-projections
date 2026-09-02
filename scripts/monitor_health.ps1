# Poll API health endpoints for uptime monitoring (Task Scheduler, cron, or external probe).
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSeconds = 10,
    [string]$ReportPath = ""
)

$ErrorActionPreference = "Stop"
$checks = @(
    @{ Name = "live"; Url = "$BaseUrl/health/live"; Expect = "ok" },
    @{ Name = "ready"; Url = "$BaseUrl/health/ready"; Expect = "ready" }
)

$results = @()
$failed = $false

foreach ($check in $checks) {
    try {
        $resp = Invoke-RestMethod -Uri $check.Url -TimeoutSec $TimeoutSeconds
        $status = $resp.status
        $ok = $status -eq $check.Expect
        if (-not $ok) { $failed = $true }
        $results += [ordered]@{
            name = $check.Name
            url = $check.Url
            status = $status
            ok = $ok
        }
        $label = if ($ok) { "OK  " } else { "FAIL" }
        Write-Host "$label $($check.Name) -> $status"
    } catch {
        $failed = $true
        $results += [ordered]@{
            name = $check.Name
            url = $check.Url
            status = "unreachable"
            ok = $false
            error = $_.Exception.Message
        }
        Write-Host "FAIL $($check.Name) -> $($_.Exception.Message)"
    }
}

$payload = [ordered]@{
    checked_at = (Get-Date).ToUniversalTime().ToString("o")
    base_url = $BaseUrl
    healthy = -not $failed
    checks = $results
}

if ($ReportPath) {
    $parent = Split-Path $ReportPath -Parent
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Path $ReportPath -Encoding utf8
}

if ($failed) { exit 1 }
Write-Host "monitor_health passed"
