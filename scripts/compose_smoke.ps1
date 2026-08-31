# Docker Compose smoke — waits for API readiness then hits key endpoints.
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)

Write-Host "Waiting for API at $BaseUrl ..."
do {
    try {
        $live = Invoke-RestMethod -Uri "$BaseUrl/health/live" -TimeoutSec 3
        $ready = Invoke-RestMethod -Uri "$BaseUrl/health/ready" -TimeoutSec 3
        if ($live.status -eq "ok" -and $ready.status -eq "ready") {
            Write-Host "OK   health endpoints"
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
} while ((Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) {
    Write-Error "API did not become ready within $TimeoutSeconds seconds"
}

$checks = @(
    @{ Name = "openapi"; Url = "$BaseUrl/openapi.json" },
    @{ Name = "auth-magic-link"; Url = "$BaseUrl/api/v1/auth/magic-link"; Method = "POST"; Body = @{ email = "owner@example.com" } }
)

foreach ($check in $checks) {
    if ($check.Method -eq "POST") {
        $resp = Invoke-RestMethod -Uri $check.Url -Method POST -ContentType "application/json" -Body ($check.Body | ConvertTo-Json)
    } else {
        $resp = Invoke-RestMethod -Uri $check.Url -TimeoutSec 10
    }
    if (-not $resp) {
        Write-Error "FAIL $($check.Name)"
    }
    Write-Host "OK   $($check.Name)"
}

Write-Host "compose smoke passed"
